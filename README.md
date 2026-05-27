# Neural-Signal-Decoding

## 项目简介
本项目基于多种经典算法对神经信号数据进行解码预测，实现神经信号到行为变量的映射。

## 使用方法
- GRU、LSTM、RNN：基于循环神经网络的时序神经信号解码
- XGBoost：基于梯度提升树的回归预测
- KalmanFilter：卡尔曼滤波线性解码

## 数据集
1. **cori 数据集**：神经信号解码任务
2. **crcns hc2 数据集**：位置预测任务（分别预测 x 坐标、y 坐标）
   注：本项目仅使用每个数据集的 **一个 session 数据**

## 文件结构
每种算法单独一个文件夹，内含对应数据集的运行代码：
## 文件结构
```text
## 文件结构
├── GRU/
│   ├── cori.py
│   └── hc2.py
├── LSTM/
│   ├── cori.py
│   └── hc2.py
├── RNN/
│   ├── cori.py
│   └── hc2.py
├── XGBoost/
│   ├── cori.py
│   └── hc2.py
├── KalmanFilter/
│   ├── cori.py
│   └── hc2.py
├── main_hc.py         
├── main_cori.py        
├── .gitignore
├── LICENSE
└── README.md
```

## 运行环境
- Python 3.12
- TensorFlow / Keras
- XGBoost
- NumPy, Pandas, Scipy

## 使用说明
直接运行对应文件夹下的 .py 文件即可执行对应模型的解码任务。