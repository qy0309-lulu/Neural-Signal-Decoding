import numpy as np

verbose = 0
# 加载数据集
binned_spikes = np.load(r'E:\Neural_Decoding-master\binned_spikes.npy')
choices = np.load(r'E:\Neural_Decoding-master\choices.npy')+1

# 查看数据形状和前10个标签
print(binned_spikes.shape, choices.shape)
print(choices[:10])

# 80/20 划分训练集和验证集
n_trials = binned_spikes.shape[0]
split = int(n_trials * 4/5)

training_spikes = binned_spikes[:split]
validation_spikes = binned_spikes[split:]

training_choices = choices[:split]
validation_choices = choices[split:]

# =============================================================================
from Neural_Decoding import decoders
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# XGBoost不支持时序维度，需展平数据
flat_train_data = np.reshape(training_spikes, (len(training_spikes),-1))
flat_val_data = np.reshape(validation_spikes, (len(validation_spikes),-1))

# 初始化并训练XGBoost（启用GPU）
my_XGBoost_classifier = decoders.XGBoostClassification(gpu=1)
my_XGBoost_classifier.fit(flat_train_data, training_choices)

# 预测并计算准确率
predictions = my_XGBoost_classifier.predict(flat_val_data)
accuracy = np.mean(predictions == validation_choices)
print("\n[XGBoost] validation accuracy: {} %".format(100*accuracy))