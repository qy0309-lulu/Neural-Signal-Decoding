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
from sklearn.decomposition import PCA

# PCA降维（展平数据后）
pca = PCA(n_components=2)
pca.fit(flat_train_data)
X_train = pca.transform(flat_train_data)
X_test = pca.transform(flat_val_data)

# 重构时序维度（需匹配LSTM输入格式，示例仅为参考，需根据实际维度调整）
X_train = np.reshape(X_train, (X_train.shape[0], 1089, 50))  # 需根据实际维度适配
X_test = np.reshape(X_test, (X_test.shape[0], 1089, 50))

# 训练LSTM
LSTM_classifier = decoders.LSTMClassification(units = 100,
                                              dropout=0.1,
                                              num_epochs = 10,
                                              verbose = verbose)
LSTM_classifier.fit(X_train, training_choices)
predictions = LSTM_classifier.predict(X_test)
accuracy = np.mean(predictions == validation_choices)
print("[PCA+LSTM] accuracy: {} %".format(100*accuracy))