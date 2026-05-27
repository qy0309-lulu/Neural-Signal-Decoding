# 安装依赖
# pip install googledrivedownloader
import numpy as np

# import googledrivedownloader as gdd
#
# # 下载并解压预处理后的Cori数据集
# gdd.download_file_from_google_drive(file_id='1W3TwEtC0Z6Qmbfuz8_AWRiQHfuDb9FIS',
#                                     dest_path='./Binned_data.zip',
#                                     unzip=True)

verbose = 0
# 加载数据集
binned_spikes = np.load('binned_spikes.npy')
choices = np.load('choices.npy')+1

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

# 初始化并训练SimpleRNN
my_RNN_classifier = decoders.SimpleRNNClassification(units = 100,
                                                    dropout = 0.2,
                                                    num_epochs =10,
                                                    verbose = verbose)
my_RNN_classifier.fit(training_spikes, training_choices)

# 预测并计算准确率
predictions = my_RNN_classifier.predict(validation_spikes)
accuracy = np.sum(predictions == validation_choices) / float(len(predictions))
print("\n[simple RNN] validation accuracy: {} %".format(100*accuracy))

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

# ==============================================================================
# 初始化并训练LSTM
my_LSTM_classifier = decoders.LSTMClassification(units = 100,
                                                 dropout=0.1,
                                                 num_epochs = 10,
                                                 verbose = verbose)
my_LSTM_classifier.fit(training_spikes, training_choices)

# 预测并计算准确率
predictions = my_LSTM_classifier.predict(validation_spikes)
accuracy = np.mean(predictions == validation_choices)
print("\n[LSTM] validation accuracy: {} %".format(100*accuracy))

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

# ==============================================================================
def get_test_train_splits(data, decisions, n_folds=5):
    """生成k折交叉验证的训练/验证集"""
    fold_size = len(data) // n_folds
    training_sets = [np.roll(data, fold_size * i, axis=0)[fold_size:] for i in range(n_folds)]
    val_sets = [np.roll(data, fold_size * i, axis=0)[:fold_size] for i in range(n_folds)]
    training_Y = [np.roll(decisions, fold_size * i, axis=0)[fold_size:] for i in range(n_folds)]
    val_Y = [np.roll(decisions, fold_size * i, axis=0)[:fold_size] for i in range(n_folds)]
    return (training_sets, training_Y), (val_sets, val_Y)


# 执行5折交叉验证（以LSTM为例）
(training_sets, training_Ys), (val_sets, val_Ys) = get_test_train_splits(training_spikes, training_choices)
scores = []

for fold in range(5):
    print("Fold {} of 5".format(fold))
    training_X = training_sets[fold]
    training_Y = training_Ys[fold]
    validation_X = val_sets[fold]
    validation_Y = val_Ys[fold]

    # 初始化并训练LSTM
    LSTM_classifier = decoders.LSTMClassification(units=100,
                                                  dropout=0.1,
                                                  num_epochs=10,
                                                  verbose=verbose)
    LSTM_classifier.fit(training_X, training_Y)

    # 计算当前折的准确率
    predictions = LSTM_classifier.predict(validation_X)
    accuracy = np.mean(predictions == validation_Y)
    scores.append(accuracy)
    print(f"Fold {fold} acc is {accuracy:.2f}")

print(f"[5 fold LSTM] Aaverage Accuracy: {np.mean(scores):.2f}")