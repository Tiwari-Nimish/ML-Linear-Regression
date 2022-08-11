import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

X, y = datasets.make_regression(n_samples=1000, n_features=3, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=1234)

print(X_train.shape)
print(y_train.shape)

from liner_reg import LinearRegression
regressor = LinearRegression()

def mse(y_true, y_predicted):
    return np.mean((y_true-y_predicted)**2)

x1_train = X_train[:,1]
x1_test = X_test[:,1]
x1_train = x1_train.reshape(800,1)
x1_test = x1_test.reshape(200,1)
print(x1_train.shape)

regressor.fit(x1_train, y_train)
y_pred_line = regressor.predict(x1_train)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8,6))
m1 = plt.scatter(x1_train, y_train, color=cmap(0.9), s=10)
plt.plot(x1_train, y_pred_line, color='black', linewidth=2, label="Prediction")
plt.show()