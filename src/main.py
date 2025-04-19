"""
唇动检测项目主入口
"""
import os
import sys
import argparse
import logging
import importlib

# 设置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="唇动检测项目")
    
    # 子命令解析器
    subparsers = parser.add_subparsers(dest="command", help="选择要执行的命令")
    
    # 训练子命令
    train_parser = subparsers.add_parser("train", help="训练模型")
    train_parser.add_argument("-i", "--dataset", required=True,
                             help="训练数据目录路径")
    train_parser.add_argument("-e", "--epochs", type=int, default=500,
                             help="训练轮数")
    train_parser.add_argument("-b", "--batch_size", type=int, default=16,
                             help="批次大小")
    train_parser.add_argument("-l", "--num_layers", type=int, default=1,
                             help="RNN层数")
    train_parser.add_argument("-n", "--neurons", type=int, default=64,
                             help="每层RNN神经元数量")
    train_parser.add_argument("-bi", "--bidirectional", type=str, default="True",
                             help="是否使用双向RNN ('True' 或 'False')")
    train_parser.add_argument("-g", "--use_gru", type=str, default="True",
                             help="是否使用GRU ('True' 或 'False')")
    train_parser.add_argument("-d", "--dropout", type=float, default=0.25,
                             help="Dropout率")
    train_parser.add_argument("-use_generator", action="store_true",
                             help="使用数据生成器而不是加载所有数据到内存")
    
    # 评估子命令
    eval_parser = subparsers.add_parser("evaluate", help="评估模型")
    eval_parser.add_argument("-i", "--dataset", required=True,
                            help="数据集路径")
    eval_parser.add_argument("-m", "--model", required=True,
                            help="模型文件路径")
    eval_parser.add_argument("-o", "--output_dir", default="evaluation_results",
                            help="评估结果输出目录")
    eval_parser.add_argument("--visualize", action="store_true",
                            help="是否可视化评估结果")
    
    # 预测子命令
    predict_parser = subparsers.add_parser("predict", help="使用模型进行预测")
    predict_parser.add_argument("-v", "--video", required=True,
                               help="视频文件路径或摄像头索引（0表示默认摄像头）")
    predict_parser.add_argument("-p", "--shape_predictor", required=True,
                               help="dlib面部特征点预测器模型路径")
    predict_parser.add_argument("-m", "--model", required=True,
                               help="训练好的模型路径")
    predict_parser.add_argument("-d", "--display_size", type=str, default="640x480",
                               help="显示窗口大小，格式为'宽x高'")
    predict_parser.add_argument("--show_landmarks", action="store_true",
                               help="是否显示所有面部特征点")
    
    # 网格搜索子命令
    grid_parser = subparsers.add_parser("grid_search", help="执行网格搜索优化模型参数")
    grid_parser.add_argument("-i", "--dataset", required=True,
                            help="训练数据目录路径")
    grid_parser.add_argument("-go", "--grid_options", required=True,
                            help="网格搜索参数配置文件路径")
    grid_parser.add_argument("-gr", "--grid_results", required=True,
                            help="网格搜索结果输出文件路径")
    
    # 数据准备子命令
    prep_parser = subparsers.add_parser("prepare_data", help="准备训练数据")
    prep_parser.add_argument("-i", "--input", required=True,
                            help="输入视频目录路径")
    prep_parser.add_argument("-o", "--output", required=True,
                            help="输出序列数据集目录路径")
    prep_parser.add_argument("-p", "--shape_predictor", required=True,
                            help="dlib面部特征点预测器模型路径")
    prep_parser.add_argument("-s", "--sequence_length", type=int, default=25,
                            help="序列长度")
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    if not args.command:
        logger.error("请指定要执行的命令。使用 -h 查看帮助信息。")
        return
    
    if args.command == "train":
        from train import main as train_main
        sys.argv = [sys.argv[0]] + [arg for arg in sys.argv[2:]]
        train_main()
    
    elif args.command == "evaluate":
        from evaluate import main as eval_main
        sys.argv = [sys.argv[0]] + [arg for arg in sys.argv[2:]]
        eval_main()
    
    elif args.command == "predict":
        from predict import main as predict_main
        sys.argv = [sys.argv[0]] + [arg for arg in sys.argv[2:]]
        predict_main()
    
    elif args.command == "grid_search":
        try:
            import lip_movement_net
            with open(args.grid_options, 'r') as f:
                if not f.readline().strip():
                    logger.info("生成网格搜索配置文件...")
                    lip_movement_net.generate_grid_data(args.grid_options)
            
            logger.info("执行网格搜索...")
            lip_movement_net.train_in_grid_search_mode(args.grid_options, args.grid_results, args.dataset)
        except Exception as e:
            logger.error(f"执行网格搜索时出错: {e}")
    
    elif args.command == "prepare_data":
        try:
            from prepare_training_dataset import prepare as prepare_main
            sys.argv = [sys.argv[0]] + [arg for arg in sys.argv[2:]]
            prepare_main()
        except Exception as e:
            logger.error(f"准备数据时出错: {e}")

if __name__ == "__main__":
    main()