import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf

# 读取数据
data = pd.read_csv(r"C:\Users\94506\Desktop\deep learing work\dlcode\波士顿房价数据集.csv")

# 检查缺失值
missing_values = data.isnull().sum()
if missing_values.sum() > 0:
    data.fillna(data.mean(), inplace=True)

# 检测重复值
duplicate_rows = data[data.duplicated()]
if not duplicate_rows.empty:
    data.drop_duplicates(inplace=True)

# 处理异常值
for feature in data.columns:
    Q1 = data[feature].quantile(0.25)
    Q3 = data[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data[feature] = data[feature].apply(
        lambda x: lower_bound if x < lower_bound
        else (upper_bound if x > upper_bound else x))

# 归一化处理
scaler = StandardScaler()
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
X = scaler.fit_transform(X)
y = y.reshape(-1, 1)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def train_and_evaluate_model(X_train, X_test, y_train, y_test, hidden_units, learning_rate, epochs=100, batch_size=32):
    # 定义神经网络模型
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(hidden_units[0], activation='relu', input_shape=(X_train.shape[1],)))
    model.add(tf.keras.layers.Dense(hidden_units[1], activation='relu'))
    model.add(tf.keras.layers.Dense(hidden_units[2], activation='relu'))
    model.add(tf.keras.layers.Dense(1))

    # 自定义损失函数
    def custom_loss(y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))

    # 自定义优化器
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # 编译模型
    model.compile(optimizer=optimizer, loss=custom_loss)

    # 训练模型
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    # 评估模型
    train_loss = history.history['loss'][-1]
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)

    return train_loss, mse_train, mse_test, model


# 定义一组可能的隐藏层单元数量和学习率
hidden_units_list = [[128, 64, 32], [64, 32, 16], [256, 128, 64]]
learning_rate_list = [0.001, 0.01]

# 初始化最佳测试集MSE和对应的参数组合
best_mse = float('inf')
best_params = None
best_model = None

# 遍历可能的参数组合
for hidden_units in hidden_units_list:
    for learning_rate in learning_rate_list:
        # 训练和评估模型
        train_loss, mse_train, mse_test, model = train_and_evaluate_model(X_train, X_test, y_train, y_test,
                                                                          hidden_units,
                                                                          learning_rate)

        # 更新最佳测试集MSE和对应的参数组合
        if mse_test < best_mse:
            best_mse = mse_test
            best_params = {'hidden_units': hidden_units, 'learning_rate': learning_rate}
            best_model = model

# 打印最佳参数的模型以及其测试集MSE
print("Best Model Parameters:", best_params)
print("Best Test MSE:", best_mse)

# 保存最佳参数模型
best_model.save("best_model")

plt.figure(figsize=(10, 5))

# 训练集预测结果
plt.subplot(1, 2, 1)
plt.scatter(y_train, best_model.predict(X_train), color='blue')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=4)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Training Set Predictions')

# 测试集预测结果
plt.subplot(1, 2, 2)
plt.scatter(y_test, best_model.predict(X_test), color='red')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Test Set Predictions')

plt.tight_layout()
plt.show()
