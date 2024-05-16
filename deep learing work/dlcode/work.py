import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv(r"C:\Users\94506\Desktop\deep learing work\dlcode\波士顿房价数据集.csv")
# print(data.describe())
# print(data.info())
# print(data)

# 检查缺失值
missing_values = data.isnull().sum()
# print("缺失值情况：\n", missing_values)

# 如果存在缺失值，则填充缺失值
if missing_values.sum() > 0:
    data.fillna(data.mean(), inplace=True)
    # print(data.head())

# 检测重复值
duplicate_rows = data[data.duplicated()]
if not duplicate_rows.empty:
    data.drop_duplicates(inplace=True)

# # 对每个特征进行异常值检测和处理，并画出箱型图
# plt.figure(figsize=(10, 6))
# for i, feature in enumerate(data.columns):
#     plt.subplot(3, 5, i + 1)
#     plt.boxplot(data[feature], vert=False)
#     plt.title(feature)
# plt.tight_layout()
# # plt.show()

for feature in data.columns:
    # 根据箱线图检测异常值
    Q1 = data[feature].quantile(0.25)
    Q3 = data[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # 处理异常值
    data[feature] = data[feature].apply(
        lambda x: lower_bound if x < lower_bound
        else (upper_bound if x > upper_bound else x))
# print(data.head())

# 归一化处理
scaler = StandardScaler()
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
X = scaler.fit_transform(X)
y = y.reshape(-1, 1)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print("训练集大小:", X_train.shape)
# print("测试集大小:", X_test.shape)

import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 定义神经网络模型
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(1))

# 自定义损失函数
def custom_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# 自定义优化器
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# 编译模型
model.compile(optimizer=optimizer, loss=custom_loss)

# 训练模型
history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, verbose=1)

# 打印训练阶段的loss值
loss = history.history['loss']
print("训练阶段的Loss值:")
print(loss)

# 预测
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

# 计算均方误差
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
print("训练集均方误差:", mse_train)
print("测试集均方误差:", mse_test)


import matplotlib.pyplot as plt

# 可视化训练损失的变化情况
plt.plot(history.history['loss'], label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 可视化实际值与预测值的比较
plt.scatter(y_test, y_pred_test, color='blue', label='Predicted')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='True')
plt.title('Actual vs Predicted')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.legend()
plt.show()


