"""
数据加载和处理模块
"""
import os
import numpy as np
import csv
import logging
from typing import List, Tuple, Optional, Dict
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# 设置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LipDataLoader:
    """嘴唇数据加载器类"""
    
    def __init__(self, class_hash: Dict[str, int]):
        """
        初始化数据加载器
        
        Args:
            class_hash: 类别名称到索引的映射
        """
        self.class_hash = class_hash
    
    def load_sequences_into_memory(self, dataset_top_dir: str, type_name: str, frame_seq_len: int = 25):
        """
        将序列加载到内存中
        
        Args:
            dataset_top_dir: 数据集顶级目录
            type_name: 数据集类型名称（'train', 'val', 'test'）
            frame_seq_len: 序列长度
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 特征和标签数据
        """
        X_data = []
        y_data = []

        # 统计序列目录数量和类别分布
        num_seq_dirs = 0
        class_wise_totals = {}

        # 第一步：计算要处理的序列总数
        data_set_type_dir = os.path.join(dataset_top_dir, type_name)
        logger.info(f"处理目录 {data_set_type_dir}")
        
        class_names = os.listdir(data_set_type_dir)
        for class_name in class_names:
            num_sequences_for_class = 0
            class_dir = os.path.join(data_set_type_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            data_set_names = os.listdir(class_dir)
            for data_set_name in data_set_names:
                data_set_dir = os.path.join(class_dir, data_set_name)
                if not os.path.isdir(data_set_dir):
                    continue
                    
                person_dir_names = os.listdir(data_set_dir)
                for person_dir_name in person_dir_names:
                    person_dir = os.path.join(data_set_dir, person_dir_name)
                    if not os.path.isdir(person_dir):
                        continue

                    sequence_dir_names = os.listdir(person_dir)
                    n = len(sequence_dir_names)
                    num_seq_dirs += n
                    num_sequences_for_class += n

            class_wise_totals[class_name] = num_sequences_for_class

        logger.info(f"正在为 {data_set_type_dir} 加载 {num_seq_dirs} 个序列到内存")
        logger.info(f"类别分布: {class_wise_totals}")

        # 创建进度条
        progress_step = max(1, num_seq_dirs // 100)  # 每完成1%更新一次进度
        progress_count = 0
        
        # 第二步：加载每个序列
        for class_name in class_names:
            class_dir = os.path.join(data_set_type_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            data_set_names = os.listdir(class_dir)
            for data_set_name in data_set_names:
                data_set_dir = os.path.join(class_dir, data_set_name)
                if not os.path.isdir(data_set_dir):
                    continue
                    
                person_dir_names = os.listdir(data_set_dir)
                for person_dir_name in person_dir_names:
                    person_dir = os.path.join(data_set_dir, person_dir_name)
                    if not os.path.isdir(person_dir):
                        continue

                    sequence_dir_names = os.listdir(person_dir)

                    for sequence_dir_name in sequence_dir_names:
                        progress_count += 1
                        if progress_count % progress_step == 0:
                            logger.info(f"进度: {progress_count}/{num_seq_dirs} ({progress_count/num_seq_dirs*100:.1f}%)")
                            
                        sequence_dir = os.path.join(person_dir, sequence_dir_name)
                        if not os.path.isdir(sequence_dir):
                            continue
                            
                        # 获取并排序所有CSV文件
                        try:
                            facial_landmark_file_names = sorted(os.listdir(sequence_dir))
                            # 检查序列长度
                            if len(facial_landmark_file_names) < frame_seq_len:
                                logger.warning(f"警告: 忽略序列目录 {sequence_dir}，序列长度为 {len(facial_landmark_file_names)}，需要 {frame_seq_len}")
                                continue
                                
                            # 仅使用指定长度的序列
                            facial_landmark_file_names = facial_landmark_file_names[:frame_seq_len]
                            
                            # 提取唇部分离序列
                            lip_separation_sequence = []
                            for facial_landmark_file_name in facial_landmark_file_names:
                                facial_landmark_file_path = os.path.join(sequence_dir, facial_landmark_file_name)
                                with open(facial_landmark_file_path, 'r') as f_obj:
                                    reader = csv.reader(f_obj)
                                    for coords in reader:
                                        # 提取嘴唇关键点
                                        if len(coords) >= 136:  # 确保有足够的坐标点 (68 点 * 2 坐标)
                                            part_61 = (int(coords[2 * 61]), int(coords[2 * 61 + 1]))
                                            part_67 = (int(coords[2 * 67]), int(coords[2 * 67 + 1]))
                                            part_62 = (int(coords[2 * 62]), int(coords[2 * 62 + 1]))
                                            part_66 = (int(coords[2 * 66]), int(coords[2 * 66 + 1]))
                                            part_63 = (int(coords[2 * 63]), int(coords[2 * 63 + 1]))
                                            part_65 = (int(coords[2 * 65]), int(coords[2 * 65 + 1]))

                                            # 计算嘴唇间距
                                            A = self._dist(part_61, part_67)
                                            B = self._dist(part_62, part_66)
                                            C = self._dist(part_63, part_65)

                                            avg_gap = (A + B + C) / 3.0
                                            
                                            # 将特征添加到序列中
                                            lip_separation_sequence.append([avg_gap])
                                            break
                            
                            # 确保序列长度正确
                            if len(lip_separation_sequence) != frame_seq_len:
                                logger.warning(f"警告: 序列 {sequence_dir} 的特征长度 {len(lip_separation_sequence)} 不等于 {frame_seq_len}")
                                continue
                                
                            # 归一化序列
                            scaler = MinMaxScaler()
                            arr = scaler.fit_transform(lip_separation_sequence)
                            
                            # 添加到数据集
                            X_data.append(arr)
                            y_data.append(self.class_hash[class_name])
                        except Exception as e:
                            logger.error(f"处理序列 {sequence_dir} 时出错: {e}")

        # 转换为NumPy数组
        X_data = np.array(X_data)
        y_data = np.array(y_data)
        logger.info(f"数据加载完成。X_data.shape={X_data.shape}, y_data.shape={y_data.shape}")

        return X_data, y_data

    def _dist(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """
        计算两点之间的欧几里得距离
        
        Args:
            p1: 第一个点的(x,y)坐标
            p2: 第二个点的(x,y)坐标
            
        Returns:
            float: 两点之间的距离
        """
        p1_x, p1_y = p1
        p2_x, p2_y = p2
        return np.sqrt((p2_x - p1_x) ** 2 + (p2_y - p1_y) ** 2)


class LipDataGenerator(tf.keras.utils.Sequence):
    """嘴唇数据序列生成器，用于TensorFlow训练"""
    
    def __init__(
        self, 
        data_dir: str, 
        class_hash: Dict[str, int],
        batch_size: int = 16, 
        sequence_length: int = 25, 
        num_features: int = 1, 
        shuffle: bool = True
    ):
        """
        初始化数据生成器
        
        Args:
            data_dir: 数据目录
            class_hash: 类别映射
            batch_size: 批次大小
            sequence_length: 序列长度
            num_features: 特征数量
            shuffle: 是否在每个epoch结束时洗牌
        """
        self.data_dir = data_dir
        self.class_hash = class_hash
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.shuffle = shuffle
        
        # 扫描目录，找到所有序列
        self.sequences = []
        self.labels = []
        self._scan_directories()
        
        # 初始化索引
        self.indexes = np.arange(len(self.sequences))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def _scan_directories(self):
        """扫描目录，找到所有序列"""
        class_names = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]
        
        for class_name in class_names:
            if class_name not in self.class_hash:
                logger.warning(f"忽略未知类别 {class_name}")
                continue
                
            class_dir = os.path.join(self.data_dir, class_name)
            dataset_names = [d for d in os.listdir(class_dir) if os.path.isdir(os.path.join(class_dir, d))]
            
            for dataset_name in dataset_names:
                dataset_dir = os.path.join(class_dir, dataset_name)
                person_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
                
                for person_dir in person_dirs:
                    person_path = os.path.join(dataset_dir, person_dir)
                    sequence_dirs = [d for d in os.listdir(person_path) if os.path.isdir(os.path.join(person_path, d))]
                    
                    for seq_dir in sequence_dirs:
                        seq_path = os.path.join(person_path, seq_dir)
                        
                        # 检查序列是否完整
                        files = os.listdir(seq_path)
                        if len(files) >= self.sequence_length:
                            self.sequences.append(seq_path)
                            self.labels.append(self.class_hash[class_name])
        
        logger.info(f"找到 {len(self.sequences)} 个序列")
        
        # 检查类别分布
        label_counts = {}
        for label in self.labels:
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1
            
        for class_name, class_idx in self.class_hash.items():
            count = label_counts.get(class_idx, 0)
            logger.info(f"类别 '{class_name}' (索引 {class_idx}): {count} 个序列")
    
    def __len__(self):
        """返回每个epoch中的批次数"""
        return len(self.sequences) // self.batch_size
    
    def __getitem__(self, index):
        """获取一个批次数据"""
        # 获取当前批次的索引
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        # 创建批次数据和标签
        batch_x = np.zeros((len(indexes), self.sequence_length, self.num_features))
        batch_y = np.zeros((len(indexes), len(self.class_hash)))
        
        # 填充批次
        for i, idx in enumerate(indexes):
            seq_path = self.sequences[idx]
            label = self.labels[idx]
            
            # 读取序列数据
            sequence_data = self._load_sequence(seq_path)
            
            # 存储数据和标签
            batch_x[i] = sequence_data
            batch_y[i, label] = 1.0  # One-hot编码
        
        return batch_x, batch_y
    
    def on_epoch_end(self):
        """每个epoch结束时调用"""
        self.indexes = np.arange(len(self.sequences))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def _load_sequence(self, seq_path):
        """加载一个序列的数据"""
        sequence_data = np.zeros((self.sequence_length, self.num_features))
        
        try:
            # 获取并排序所有CSV文件
            files = sorted(os.listdir(seq_path))[:self.sequence_length]
            
            for i, file_name in enumerate(files):
                if i >= self.sequence_length:
                    break
                    
                file_path = os.path.join(seq_path, file_name)
                
                with open(file_path, 'r') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        # 提取嘴唇特征
                        part_61 = (int(row[2 * 61]), int(row[2 * 61 + 1]))
                        part_67 = (int(row[2 * 67]), int(row[2 * 67 + 1]))
                        part_62 = (int(row[2 * 62]), int(row[2 * 62 + 1]))
                        part_66 = (int(row[2 * 66]), int(row[2 * 66 + 1]))
                        part_63 = (int(row[2 * 63]), int(row[2 * 63 + 1]))
                        part_65 = (int(row[2 * 65]), int(row[2 * 65 + 1]))

                        # 计算嘴唇间距
                        def dist(p1, p2):
                            return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
                            
                        A = dist(part_61, part_67)
                        B = dist(part_62, part_66)
                        C = dist(part_63, part_65)

                        avg_gap = (A + B + C) / 3.0
                        
                        sequence_data[i, 0] = avg_gap
                        break
            
            # 归一化序列
            scaler = MinMaxScaler()
            sequence_data = scaler.fit_transform(sequence_data)
            
        except Exception as e:
            logger.error(f"加载序列 {seq_path} 时出错: {e}")
        
        return sequence_data
