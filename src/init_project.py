"""
初始化项目目录结构
"""
import os
import sys
import argparse
import logging
import shutil

# 设置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="初始化唇动检测项目目录结构")
    
    parser.add_argument("-o", "--output_dir", default="lip_movement_detection",
                       help="项目输出目录")
    parser.add_argument("--force", action="store_true",
                       help="如果目录已存在，强制覆盖")
    
    return parser.parse_args()

def create_directory_structure(output_dir, force=False):
    """创建项目目录结构"""
    # 检查输出目录是否已存在
    if os.path.exists(output_dir):
        if not force:
            logger.error(f"输出目录 {output_dir} 已存在。使用 --force 参数覆盖或指定不同的输出目录。")
            return False
        logger.warning(f"强制模式：将覆盖输出目录 {output_dir}")
    
    # 创建主目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建子目录结构
    directories = [
        os.path.join(output_dir, "src"),
        os.path.join(output_dir, "src", "models"),
        os.path.join(output_dir, "src", "data"),
        os.path.join(output_dir, "src", "utils"),
        os.path.join(output_dir, "data", "dataset_source"),
        os.path.join(output_dir, "data", "dataset_source", "train", "speaking"),
        os.path.join(output_dir, "data", "dataset_source", "train", "silent"),
        os.path.join(output_dir, "data", "dataset_source", "val", "speaking"),
        os.path.join(output_dir, "data", "dataset_source", "val", "silent"),
        os.path.join(output_dir, "data", "dataset_source", "test", "speaking"),
        os.path.join(output_dir, "data", "dataset_source", "test", "silent"),
        os.path.join(output_dir, "data", "dataset_dest"),
        os.path.join(output_dir, "models"),
        os.path.join(output_dir, "evaluation_results"),
        os.path.join(output_dir, "scripts"),
        os.path.join(output_dir, "videos")
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"创建目录: {directory}")
    
    # 创建README文件
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, 'w') as f:
        f.write("# 唇动检测项目\n\n")
        f.write("通过监测嘴唇动作判断人是否在说话的深度学习项目。\n\n")
        f.write("## 目录结构\n\n")
        f.write("- `src/`: 源代码\n")
        f.write("  - `models/`: 模型定义\n")
        f.write("  - `data/`: 数据加载和处理\n")
        f.write("  - `utils/`: 工具函数\n")
        f.write("- `data/`: 数据目录\n")
        f.write("  - `dataset_source/`: 原始视频数据\n")
        f.write("  - `dataset_dest/`: 处理后的序列数据\n")
        f.write("- `models/`: 保存训练好的模型\n")
        f.write("- `evaluation_results/`: 模型评估结果\n")
        f.write("- `scripts/`: 脚本文件\n")
        f.write("- `videos/`: 测试视频文件\n\n")
        f.write("## 使用方法\n\n")
        f.write("### 准备数据\n\n")
        f.write("```bash\n")
        f.write("python src/main.py prepare_data -i data/dataset_source -o data/dataset_dest -p models/shape_predictor_68_face_landmarks.dat\n")
        f.write("```\n\n")
        f.write("### 训练模型\n\n")
        f.write("```bash\n")
        f.write("python src/main.py train -i data/dataset_dest -e 100 -b 16 -l 2 -n 64 -bi True -g True -d 0.25\n")
        f.write("```\n\n")
        f.write("### 评估模型\n\n")
        f.write("```bash\n")
        f.write("python src/main.py evaluate -i data/dataset_dest -m models/2_64_True_True_0.25_lip_motion_net_model.h5 --visualize\n")
        f.write("```\n\n")
        f.write("### 预测\n\n")
        f.write("```bash\n")
        f.write("python src/main.py predict -v videos/test_video.mp4 -p models/shape_predictor_68_face_landmarks.dat -m models/2_64_True_True_0.25_lip_motion_net_model.h5\n")
        f.write("```\n\n")
        f.write("### 使用网格搜索优化模型\n\n")
        f.write("```bash\n")
        f.write("python src/main.py grid_search -i data/dataset_dest -go csv/grid_options.csv -gr csv/grid_results.csv\n")
        f.write("```\n")
    
    logger.info(f"创建README文件: {readme_path}")
    
    # 创建模型占位符文件
    model_placeholder = os.path.join(output_dir, "models", "put_shape_predictor_68_face_landmarks.dat_here.txt")
    with open(model_placeholder, 'w') as f:
        f.write("请从dlib项目下载shape_predictor_68_face_landmarks.dat文件并放在这个目录。\n")
        f.write("下载链接: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2\n")
    
    logger.info(f"创建模型占位符文件: {model_placeholder}")
    
    # 创建requirements.txt文件
    requirements_path = os.path.join(output_dir, "requirements.txt")
    with open(requirements_path, 'w') as f:
        f.write("tensorflow>=2.4.0\n")
        f.write("numpy>=1.19.2\n")
        f.write("opencv-python>=4.5.1\n")
        f.write("dlib>=19.21.0\n")
        f.write("scikit-learn>=0.24.1\n")
        f.write("matplotlib>=3.3.4\n")
        f.write("seaborn>=0.11.1\n")
        f.write("progressbar2>=3.53.1\n")
    
    logger.info(f"创建requirements文件: {requirements_path}")
    
    # 创建示例脚本文件
    # Linux/Mac 训练脚本
    train_sh_path = os.path.join(output_dir, "scripts", "train.sh")
    with open(train_sh_path, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("cd ..\n")
        f.write("python src/main.py train -i data/dataset_dest -e 100 -b 16 -l 2 -n 64 -bi True -g True -d 0.25\n")
    
    # Windows 训练脚本
    train_bat_path = os.path.join(output_dir, "scripts", "train.bat")
    with open(train_bat_path, 'w') as f:
        f.write("@echo off\n")
        f.write("cd ..\n")
        f.write("python src\\main.py train -i data\\dataset_dest -e 100 -b 16 -l 2 -n 64 -bi True -g True -d 0.25\n")
    
    logger.info(f"创建示例脚本文件: {train_sh_path}, {train_bat_path}")
    
    return True

def main():
    """主函数"""
    args = parse_args()
    
    if create_directory_structure(args.output_dir, args.force):
        logger.info(f"成功初始化项目目录结构于: {args.output_dir}")
        logger.info(f"接下来请安装依赖项: pip install -r {os.path.join(args.output_dir, 'requirements.txt')}")
    else:
        logger.error("初始化项目目录结构失败")

if __name__ == "__main__":
    main()
