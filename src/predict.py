"""
使用训练好的模型进行预测
"""
import os
import cv2
import dlib
import numpy as np
import argparse
import logging
import time
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from typing import List, Tuple, Optional

# 导入配置
import config

# 设置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="使用训练好的模型进行预测")
    
    parser.add_argument("-v", "--video", required=True,
                        help="视频文件路径或摄像头索引（0表示默认摄像头）")
    parser.add_argument("-p", "--shape_predictor", required=True,
                        help="dlib面部特征点预测器模型路径")
    parser.add_argument("-m", "--model", required=True,
                        help="训练好的模型路径")
    parser.add_argument("-s", "--sequence_length", type=int, default=config.FRAME_SEQ_LEN,
                        help="序列长度")
    parser.add_argument("-d", "--display_size", type=str, default="640x480",
                        help="显示窗口大小，格式为'宽x高'")
    parser.add_argument("--show_landmarks", action="store_true",
                        help="是否显示所有面部特征点")
    
    return parser.parse_args()

def resize_frame(frame, target_size):
    """调整帧大小"""
    if target_size:
        width, height = map(int, target_size.split('x'))
        return cv2.resize(frame, (width, height))
    return frame

def get_facial_landmark_vectors_from_frame(frame, detector, shape_predictor):
    """从帧中提取面部特征点"""
    # 转换为灰度图像以加速处理
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 检测人脸
    dets = detector(gray, 1)
    if not dets:
        return None, None
    
    # 获取第一个检测到的人脸
    facial_points = []
    for k, d in enumerate(dets):
        shape = shape_predictor(gray, d)
        if shape is None:
            continue

        # 提取所有68个特征点
        for i in range(68):
            part = shape.part(i)
            facial_points.append(part.x)
            facial_points.append(part.y)

        # 只处理第一个人脸
        if len(facial_points) > 0:
            break

    return dets, facial_points

def extract_lip_features(facial_points):
    """提取唇部特征"""
    if not facial_points or len(facial_points) < 136:  # 68点 * 2坐标
        return None
    
    # 提取唇部关键点
    part_61 = (facial_points[2 * 61], facial_points[2 * 61 + 1])
    part_67 = (facial_points[2 * 67], facial_points[2 * 67 + 1])
    part_62 = (facial_points[2 * 62], facial_points[2 * 62 + 1])
    part_66 = (facial_points[2 * 66], facial_points[2 * 66 + 1])
    part_63 = (facial_points[2 * 63], facial_points[2 * 63 + 1])
    part_65 = (facial_points[2 * 65], facial_points[2 * 65 + 1])

    # 计算唇部间距
    def dist(p1, p2):
        return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
    
    A = dist(part_61, part_67)
    B = dist(part_62, part_66)
    C = dist(part_63, part_65)
    
    avg_gap = (A + B + C) / 3.0
    
    return avg_gap

def test_video(args):
    """在视频上测试模型"""
    # 初始化face detector和shape predictor
    detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor(args.shape_predictor)
    
    # 加载模型
    logger.info(f"加载模型 {args.model}")
    model = tf.keras.models.load_model(args.model)
    
    # 确定视频源
    video_source = args.video
    try:
        video_source = int(video_source)  # 尝试转换为摄像头索引
        logger.info(f"使用摄像头索引 {video_source}")
    except ValueError:
        logger.info(f"使用视频文件 {video_source}")
    
    # 打开视频源
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        logger.error(f"无法打开视频源 {args.video}")
        return
    
    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:  # 如果是摄像头可能返回0
        fps = 30  # 设置默认值
    logger.info(f"视频FPS: {fps}")
    
    # 初始化状态和序列
    state = "等待足够的帧..."
    input_sequence = []
    predictions = []  # 用于平滑预测结果
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # 主循环
    while True:
        # 读取一帧
        ret, frame = cap.read()
        if not ret:
            if isinstance(video_source, str):
                # 如果是文件，循环播放
                cap = cv2.VideoCapture(video_source)
                ret, frame = cap.read()
                if not ret:
                    break
            else:
                # 如果是摄像头，继续尝试
                continue
        
        # 调整帧大小
        frame = resize_frame(frame, args.display_size)
        
        # 创建显示帧的拷贝
        display_frame = frame.copy()
        
        # 提取面部特征点
        dets, facial_points = get_facial_landmark_vectors_from_frame(frame, detector, shape_predictor)
        
        # 如果检测到人脸
        if dets and facial_points:
            # 绘制人脸框
            for i, d in enumerate(dets):
                cv2.rectangle(display_frame, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 0), 2)
                cv2.rectangle(display_frame, (d.left(), d.bottom()), (d.right(), d.bottom() + 30), (0, 0, 255), cv2.FILLED)
                cv2.putText(display_frame, state, (d.left() + 5, d.bottom() + 20), font, 0.6, (255, 255, 255), 2)
                
                # 如果需要，绘制所有面部特征点
                if args.show_landmarks:
                    for i in range(68):
                        x = facial_points[2 * i]
                        y = facial_points[2 * i + 1]
                        cv2.circle(display_frame, (x, y), 1, (0, 0, 255), -1)
                
                # 只绘制嘴唇特征点
                lip_indices = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]
                for i in lip_indices:
                    x = facial_points[2 * i]
                    y = facial_points[2 * i + 1]
                    cv2.circle(display_frame, (x, y), 2, (0, 255, 255), -1)
            
            # 提取唇部特征
            lip_gap = extract_lip_features(facial_points)
            if lip_gap is not None:
                # 添加到序列并保持序列长度
                input_sequence.append([lip_gap])
                if len(input_sequence) > args.sequence_length:
                    input_sequence.pop(0)
                
                # 当序列足够长时进行预测
                if len(input_sequence) == args.sequence_length:
                    # 归一化序列
                    scaler = MinMaxScaler()
                    arr = scaler.fit_transform(input_sequence)
                    
                    # 进行预测
                    X_data = np.array([arr])
                    y_pred = model.predict(X_data, verbose=0)[0]
                    
                    # 获取预测类别
                    y_pred_max = y_pred.argmax()
                    
                    # 添加到预测历史并保持长度
                    predictions.append(y_pred_max)
                    if len(predictions) > 5:  # 保留最近5个预测
                        predictions.pop(0)
                    
                    # 平滑预测结果（简单的多数投票）
                    from collections import Counter
                    most_common = Counter(predictions).most_common(1)[0][0]
                    
                    # 更新状态
                    for k, v in config.CLASS_HASH.items():
                        if v == most_common:
                            confidence = y_pred[most_common]
                            state = f"{k.capitalize()} ({confidence:.2f})"
                            break
                    
                    # 更新显示
                    for i, d in enumerate(dets):
                        cv2.rectangle(display_frame, (d.left(), d.bottom()), (d.right(), d.bottom() + 30), (0, 0, 255), cv2.FILLED)
                        cv2.putText(display_frame, state, (d.left() + 5, d.bottom() + 20), font, 0.6, (255, 255, 255), 2)
        
        # 显示帧
        cv2.imshow('Lip Movement Detector', display_frame)
        
        # 检查按键
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # 按'q'退出
            break
        elif key == ord('s'):  # 按's'切换显示特征点
            args.show_landmarks = not args.show_landmarks
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

def main():
    """主函数"""
    args = parse_args()
    test_video(args)

if __name__ == "__main__":
    main()
