"""
训练唇动检测模型
"""
import os
import argparse
import logging
import time
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.utils import to_categorical

# 导入自定义模块
import config
from data.data_loader import LipDataLoader, LipDataGenerator
from models.lip_model import LipMovementNet

# 设置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="训练唇动检测模型")
    
    parser.add_argument("-i", "--dataset", required=True,
                        help="训练数据目录路径")
    parser.add_argument("-e", "--epochs", type=int, default=config.NUM_EPOCHS,
                        help="训练轮数")
    parser.add_argument("-b", "--batch_size", type=int, default=config.BATCH_SIZE,
                        help="批次大小")
    parser.add_argument("-l", "--num_layers", type=int, default=1,
                        help="RNN层数")
    parser.add_argument("-n", "--neurons", type=int, default=64,
                        help="每层RNN神经元数量")
    parser.add_argument("-bi", "--bidirectional", type=str, default="True",
                        help="是否使用双向RNN ('True' 或 'False')")
    parser.add_argument("-g", "--use_gru", type=str, default="True",
                        help="是否使用GRU ('True' 或 'False')")
    parser.add_argument("-d", "--dropout", type=float, default=0.25,
                        help="Dropout率")
    parser.add_argument("-opt", "--optimizer", type=str, default="adam",
                        help="优化器类型 ('adam' 或 'rmsprop')")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001,
                        help="学习率")
    parser.add_argument("-seq", "--sequence_length", type=int, default=config.FRAME_SEQ_LEN,
                        help="序列长度")
    parser.add_argument("-f", "--features", type=int, default=config.NUM_FEATURES,
                        help="特征数量")
    parser.add_argument("-use_generator", action="store_true",
                        help="使用数据生成器而不是加载所有数据到内存")
    
    return parser.parse_args()

def str2bool(v):
    """将字符串转换为布尔值"""
    return v.lower() in ("yes", "true", "t", "1")

def train_model(args):
    """训练模型"""
    # 设置随机种子以确保结果可复现
    np.random.seed(int(time.time()))
    tf.random.set_seed(int(time.time()))
    
    # 创建模型目录和TensorBoard目录
    tensorboard_dir = os.path.join(args.dataset, 'tensorboard')
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    models_dir = os.path.join(args.dataset, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # 解析布尔参数
    is_bidirectional = str2bool(args.bidirectional)
    use_gru = str2bool(args.use_gru)
    
    # 创建模型
    logger.info("创建模型...")
    model = LipMovementNet(
        num_rnn_layers=args.num_layers,
        num_neurons_in_rnn_layer=args.neurons,
        is_bidirectional=is_bidirectional,
        use_gru=use_gru,
        dropout=args.dropout,
        optimizer=args.optimizer,
        lr=args.learning_rate,
        num_classes=len(config.CLASS_HASH),
        frames_n=args.sequence_length,
        num_features=args.features
    )
    
    # 打印模型参数和结构
    model.print_params()
    model.summary()
    
    # 设置回调函数
    log_dir = os.path.join(tensorboard_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    model_name = f"{args.num_layers}_{args.neurons}_{is_bidirectional}_{use_gru}_{args.dropout}_lip_motion_net_model.h5"
    model_path = os.path.join(models_dir, model_name)
    
    # 定义回调函数
    callbacks = [
        ModelCheckpoint(
            os.path.join(models_dir, f"checkpoint_epoch-{{epoch:02d}}_val_loss-{{val_loss:.4f}}.h5"),
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode='min'
        ),
        EarlyStopping(
            monitor='val_loss',
            min_delta=0.0005,
            patience=10,
            verbose=1
        ),
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True
        )
    ]
    
    # 训练模型：两种模式 - 内存加载或数据生成器
    if args.use_generator:
        # 使用数据生成器模式
        logger.info("使用数据生成器模式训练...")
        
        # 创建训练和验证数据生成器
        train_generator = LipDataGenerator(
            os.path.join(args.dataset, 'train'),
            config.CLASS_HASH,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            num_features=args.features
        )
        
        val_generator = LipDataGenerator(
            os.path.join(args.dataset, 'val'),
            config.CLASS_HASH,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            num_features=args.features
        )
        
        # 训练模型
        logger.info("开始训练...")
        history = model.model.fit(
            train_generator,
            epochs=args.epochs,
            validation_data=val_generator,
            callbacks=callbacks,
            workers=4,
            use_multiprocessing=True
        )
    else:
        # 使用内存加载模式
        logger.info("使用内存加载模式训练...")
        
        # 创建数据加载器
        data_loader = LipDataLoader(config.CLASS_HASH)
        
        # 加载训练数据
        logger.info("加载训练数据...")
        X_train, y_train = data_loader.load_sequences_into_memory(
            args.dataset, 
            'train',
            frame_seq_len=args.sequence_length
        )
        y_train = to_categorical(y_train, num_classes=len(config.CLASS_HASH))
        
        # 加载验证数据
        logger.info("加载验证数据...")
        X_val, y_val = data_loader.load_sequences_into_memory(
            args.dataset, 
            'val',
            frame_seq_len=args.sequence_length
        )
        y_val = to_categorical(y_val, num_classes=len(config.CLASS_HASH))
        
        # 计算训练步数
        steps_per_epoch = len(X_train) // args.batch_size
        validation_steps = len(X_val) // args.batch_size
        
        logger.info(f"每个epoch的步数={steps_per_epoch}")
        logger.info(f"验证步数={validation_steps}")
        
        # 训练模型
        logger.info("开始训练...")
        history = model.model.fit(
            X_train, 
            y_train, 
            batch_size=args.batch_size, 
            epochs=args.epochs, 
            verbose=1,
            callbacks=callbacks, 
            validation_data=(X_val, y_val), 
            shuffle=True
        )
    
    # 保存最终模型
    logger.info(f"训练完成，保存模型到 {model_path}")
    model.model.save(model_path)
    
    return model, history

def main():
    """主函数"""
    args = parse_args()
    
    logger.info(f"开始训练过程，数据集路径: {args.dataset}")
    start_time = time.time()
    
    train_model(args)
    
    end_time = time.time()
    training_time = (end_time - start_time) / 60  # 转换为分钟
    logger.info(f"训练完成！总时间: {training_time:.2f} 分钟")

if __name__ == "__main__":
    main()