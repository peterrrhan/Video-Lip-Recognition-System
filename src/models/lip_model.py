"""
唇动检测模型定义
"""
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, Bidirectional
from tensorflow.keras.optimizers import Adam, RMSprop
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LipMovementNet(Model):
    """唇部运动检测网络模型"""
    
    def __init__(
        self, 
        num_rnn_layers=1,
        num_neurons_in_rnn_layer=64,
        is_bidirectional=True,
        use_gru=True,
        dropout=0.25,
        num_output_dense_layers=0,
        num_neurons_in_output_dense_layers=0,
        activation_output_dense_layers='relu',
        optimizer='adam',
        lr=0.001,
        num_classes=2,
        frames_n=25,
        num_features=1
    ):
        """
        初始化唇部运动检测网络
        
        Args:
            num_rnn_layers: RNN层数
            num_neurons_in_rnn_layer: 每层RNN的神经元数量
            is_bidirectional: 是否使用双向RNN
            use_gru: 是否使用GRU（否则使用LSTM）
            dropout: Dropout比率
            num_output_dense_layers: 输出密集层数量
            num_neurons_in_output_dense_layers: 输出密集层神经元数量
            activation_output_dense_layers: 输出密集层激活函数
            optimizer: 优化器类型 ('adam' 或 'rmsprop')
            lr: 学习率
            num_classes: 输出类别数
            frames_n: 输入序列长度
            num_features: 特征维度
        """
        super(LipMovementNet, self).__init__()
        self.frames_n = frames_n
        self.num_features = num_features
        self.num_classes = num_classes
        
        self.num_rnn_layers = num_rnn_layers
        self.num_neurons_in_rnn_layer = num_neurons_in_rnn_layer
        self.is_bidirectional = is_bidirectional
        self.use_gru = use_gru
        self.dropout = dropout
        
        self.num_output_dense_layers = num_output_dense_layers
        self.num_neurons_in_output_dense_layers = num_neurons_in_output_dense_layers
        self.activation_output_dense_layers = activation_output_dense_layers
        self.optimizer_type = optimizer
        self.lr = lr
        
        self.model = self._build_model()
        self._compile_model()
    
    def _build_model(self):
        """构建模型架构"""
        model = Sequential()
        input_shape = (self.frames_n, self.num_features)
        
        # 构建RNN层
        for i in range(self.num_rnn_layers):
            return_sequences = i < self.num_rnn_layers - 1  # 最后一层不返回序列
            
            # 根据配置选择RNN类型
            rnn_layer = GRU if self.use_gru else LSTM
            layer_name = f'rnn-{i}'
            
            # 第一层需要指定输入形状
            if i == 0:
                if self.is_bidirectional:
                    model.add(Bidirectional(
                        rnn_layer(
                            self.num_neurons_in_rnn_layer, 
                            return_sequences=return_sequences,
                            name=layer_name
                        ),
                        input_shape=input_shape
                    ))
                else:
                    model.add(rnn_layer(
                        self.num_neurons_in_rnn_layer,
                        return_sequences=return_sequences,
                        name=layer_name,
                        input_shape=input_shape
                    ))
            else:
                if self.is_bidirectional:
                    model.add(Bidirectional(
                        rnn_layer(
                            self.num_neurons_in_rnn_layer,
                            return_sequences=return_sequences,
                            name=layer_name
                        )
                    ))
                else:
                    model.add(rnn_layer(
                        self.num_neurons_in_rnn_layer,
                        return_sequences=return_sequences,
                        name=layer_name
                    ))
        
        # 添加Dropout层
        if self.dropout > 0.0:
            model.add(Dropout(self.dropout))
        
        # 添加输出密集层
        for i in range(self.num_output_dense_layers):
            name = f'dense-{i}'
            model.add(
                Dense(
                    self.num_neurons_in_output_dense_layers, 
                    activation=self.activation_output_dense_layers,
                    name=name
                )
            )
        
        # 最终分类层
        model.add(Dense(self.num_classes, activation='softmax', name='output'))
        
        return model
    
    def _compile_model(self):
        """编译模型"""
        # 设置优化器
        if self.optimizer_type.lower() == 'adam':
            optimizer = Adam(learning_rate=self.lr)
        elif self.optimizer_type.lower() == 'rmsprop':
            optimizer = RMSprop(learning_rate=self.lr)
        else:
            logger.warning(f"未知的优化器类型: {self.optimizer_type}，使用默认的Adam")
            optimizer = Adam(learning_rate=self.lr)
        
        # 编译模型
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
        )
    
    def call(self, inputs, training=None):
        """前向传播"""
        return self.model(inputs, training=training)
    
    def get_config(self):
        """获取模型配置"""
        config = super().get_config()
        config.update({
            'num_rnn_layers': self.num_rnn_layers,
            'num_neurons_in_rnn_layer': self.num_neurons_in_rnn_layer,
            'is_bidirectional': self.is_bidirectional,
            'use_gru': self.use_gru,
            'dropout': self.dropout,
            'num_output_dense_layers': self.num_output_dense_layers,
            'num_neurons_in_output_dense_layers': self.num_neurons_in_output_dense_layers,
            'activation_output_dense_layers': self.activation_output_dense_layers,
            'optimizer': self.optimizer_type,
            'lr': self.lr,
            'num_classes': self.num_classes,
            'frames_n': self.frames_n,
            'num_features': self.num_features
        })
        return config

    def summary(self):
        """显示模型摘要"""
        return self.model.summary()
    
    def print_params(self):
        """打印模型参数"""
        logger.info(f"模型配置:")
        logger.info(f"  RNN层数: {self.num_rnn_layers}")
        logger.info(f"  每层神经元数: {self.num_neurons_in_rnn_layer}")
        logger.info(f"  是否双向: {self.is_bidirectional}")
        logger.info(f"  使用GRU: {self.use_gru}")
        logger.info(f"  Dropout率: {self.dropout}")
        logger.info(f"  输出密集层数量: {self.num_output_dense_layers}")
        logger.info(f"  输出密集层神经元数量: {self.num_neurons_in_output_dense_layers}")
        logger.info(f"  输出密集层激活函数: {self.activation_output_dense_layers}")
        logger.info(f"  优化器: {self.optimizer_type}")
        logger.info(f"  学习率: {self.lr}")
        logger.info(f"  序列长度: {self.frames_n}")
        logger.info(f"  特征维度: {self.num_features}")
        logger.info(f"  类别数量: {self.num_classes}")
