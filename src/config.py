"""
配置参数模块：集中管理所有配置参数
"""

# 模型参数
FRAME_SEQ_LEN = 25
NUM_FEATURES = 1
BATCH_SIZE = 16
NUM_EPOCHS = 500
LEARNING_RATE = 0.001
NUM_CLASSES = 2

# 类别定义
CLASS_HASH = {
    'silent': 0,
    'speaking': 1
}

# 数据处理参数
VIDEO_START_OFFSETS = {
    'GRID': 25,
    'HMDB': 0
}

VIDEO_END_OFFSETS = {
    'GRID': 50,
    'HMDB': 2147483647
}

CROSS_FILE_BOUNDARIES = {
    'GRID': False,
    'BBC': True
}
