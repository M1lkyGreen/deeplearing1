import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# 读取数据
data = pd.read_csv(r"C:\Users\94506\Desktop\deep learing work\dlcode\波士顿房价数据集.csv")

# 归一化处理
scaler = StandardScaler()
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
X = scaler.fit_transform(X)
y = y.reshape(-1, 1)

# 加载最佳模型并指定自定义损失函数
def custom_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

best_model = tf.keras.models.load_model(r"C:\Users\94506\Desktop\deep learing work\dlcode\best_model", custom_objects={'custom_loss': custom_loss})

# 对新数据进行预测
new_data = np.array([[0.02731, 0.0, 7.07, 0.0, 0.469, 6.421, 78.9, 4.9671, 2.0, 242.0, 17.8, 396.90, 9.14]])
new_data_scaled = scaler.transform(new_data)
predicted_price = best_model.predict(new_data_scaled)

print("Predicted Price:", predicted_price[0][0])
