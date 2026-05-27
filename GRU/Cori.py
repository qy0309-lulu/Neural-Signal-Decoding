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
# 初始化并训练GRU
my_GRU_classifier = decoders.GRUClassification(units = 100,
                                               dropout = 0.1,
                                               num_epochs = 10,
                                               verbose = verbose)
my_GRU_classifier.fit(training_spikes, training_choices)

# 预测并计算准确率
predictions = my_GRU_classifier.predict(validation_spikes)
accuracy = np.mean(predictions == validation_choices)
print("\n[GRU] validation accuracy: {} %".format(100*accuracy))