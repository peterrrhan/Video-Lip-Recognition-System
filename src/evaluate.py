"""
评估训练好的模型性能
"""
import os
import argparse
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import seaborn as sns

# 导入自定义模块
import config
from data.data_loader import LipDataLoader

# 设置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="评估唇动检测模型性能")
    
    parser.add_argument("-i", "--dataset", required=True,
                        help="数据集路径")
    parser.add_argument("-m", "--model", required=True,
                        help="模型文件路径")
    parser.add_argument("-s", "--sequence_length", type=int, default=config.FRAME_SEQ_LEN,
                        help="序列长度")
    parser.add_argument("-o", "--output_dir", default="evaluation_results",
                        help="评估结果输出目录")
    parser.add_argument("--visualize", action="store_true",
                        help="是否可视化评估结果")
    
    return parser.parse_args()

def evaluate_model(args):
    """评估模型性能"""
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    logger.info(f"加载模型: {args.model}")
    model = tf.keras.models.load_model(args.model)
    
    # 创建数据加载器
    data_loader = LipDataLoader(config.CLASS_HASH)
    
    # 加载测试数据
    logger.info("加载测试数据...")
    X_test, y_test = data_loader.load_sequences_into_memory(
        args.dataset, 
        'test',
        frame_seq_len=args.sequence_length
    )
    
    if len(X_test) == 0:
        logger.error("没有找到测试数据！")
        return
    
    # 进行预测
    logger.info("进行预测...")
    y_pred_probs = model.predict(X_test)
    
    # 转换为单标签
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # 计算评估指标
    logger.info("计算评估指标...")
    
    # 确保y_test和y_pred具有相同的形状
    y_test = np.array(y_test).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    
    # 计算各项指标
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # 有些情况下，ROC AUC只适用于二分类
    try:
        roc_auc = roc_auc_score(y_test, y_pred_probs[:, 1] if y_pred_probs.shape[1] == 2 else y_pred_probs, multi_class='ovr')
    except:
        roc_auc = "不适用（多分类）"
    
    # 生成详细分类报告
    class_report = classification_report(y_test, y_pred, target_names=list(config.CLASS_HASH.keys()))
    
    # 打印评估结果
    logger.info("\n==== 评估结果 ====")
    logger.info(f"测试样本数: {len(X_test)}")
    logger.info(f"精确率: {precision:.4f}")
    logger.info(f"召回率: {recall:.4f}")
    logger.info(f"F1分数: {f1:.4f}")
    if isinstance(roc_auc, str):
        logger.info(f"ROC AUC: {roc_auc}")
    else:
        logger.info(f"ROC AUC: {roc_auc:.4f}")
    
    logger.info("\n混淆矩阵:")
    logger.info(cm)
    
    logger.info("\n分类报告:")
    logger.info(class_report)
    
    # 保存评估结果到文件
    results_file = os.path.join(args.output_dir, "evaluation_results.txt")
    with open(results_file, 'w') as f:
        f.write("==== 唇动检测模型评估结果 ====\n\n")
        f.write(f"模型文件: {args.model}\n")
        f.write(f"测试数据集: {args.dataset}\n")
        f.write(f"测试样本数: {len(X_test)}\n\n")
        f.write(f"精确率: {precision:.4f}\n")
        f.write(f"召回率: {recall:.4f}\n")
        f.write(f"F1分数: {f1:.4f}\n")
        if isinstance(roc_auc, str):
            f.write(f"ROC AUC: {roc_auc}\n\n")
        else:
            f.write(f"ROC AUC: {roc_auc:.4f}\n\n")
        
        f.write("混淆矩阵:\n")
        f.write(str(cm) + "\n\n")
        
        f.write("分类报告:\n")
        f.write(class_report)
    
    logger.info(f"评估结果已保存到: {results_file}")
    
    # 可视化结果
    if args.visualize:
        logger.info("生成可视化评估结果...")
        
        # 创建一个新的图形
        plt.figure(figsize=(15, 10))
        
        # 1. 绘制混淆矩阵
        plt.subplot(1, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=list(config.CLASS_HASH.keys()),
                   yticklabels=list(config.CLASS_HASH.keys()))
        plt.title('混淆矩阵')
        plt.xlabel('预测类别')
        plt.ylabel('真实类别')
        
        # 2. 绘制类别分布
        plt.subplot(1, 2, 2)
        class_counts = np.bincount(y_test)
        plt.bar(range(len(class_counts)), class_counts)
        plt.xticks(range(len(class_counts)), list(config.CLASS_HASH.keys()))
        plt.title('测试数据类别分布')
        plt.xlabel('类别')
        plt.ylabel('样本数')
        
        # 保存图形
        viz_file = os.path.join(args.output_dir, "evaluation_visualization.png")
        plt.tight_layout()
        plt.savefig(viz_file)
        logger.info(f"可视化结果已保存到: {viz_file}")
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'classification_report': class_report
    }

def main():
    """主函数"""
    args = parse_args()
    evaluate_model(args)

if __name__ == "__main__":
    main()
