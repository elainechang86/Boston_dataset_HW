import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("poster")

# special matplotlib argument for improved plots
from matplotlib import rcParams

from sklearn.datasets import load_boston
boston = load_boston()

# print(boston.keys()) #Keys的名字
# print(boston.data.shape) #data的大小
# print(boston.feature_names) #Column的名字
print(boston.DESCR) #decription of the dataset

bos = pd.DataFrame(boston.data) #變成表格

#print(bos.head()) #head(5)就是列出前五列資料

bos.columns = boston.feature_names

# print(bos.head())
# print(boston.target.shape)

bos['PRICE'] = boston.target

# print(bos.head())
# print(bos.describe())

X = bos['LSTAT']
Y = bos['PRICE']


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33, random_state = 1)
# print(X_train.shape)
# print(X_test.shape)
# print(Y_train.shape)
# print(Y_test.shape)


from sklearn.linear_model import LinearRegression
X_test=np.array(X_test).reshape(len(X_test), 1)
Y_test=np.array(Y_test).reshape(len(Y_test), 1)
X_train=np.array(X_train).reshape(len(X_train), 1)
Y_train=np.array(Y_train).reshape(len(Y_train), 1)
lm = LinearRegression()
lm.fit(X_train, Y_train)
Y_pred = lm.predict(X_test)
plt.close('all')
# pic1=plt.figure(1)
# plt.scatter(Y_test, Y_pred) #點點圖
# plt.ylabel("Prices: $Y_i$")
# plt.xlabel("Predicted prices: $\hat{Y}_i$")
# plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")
# pic1.show()
pic2=plt.figure(2)
plt.scatter(X_test, Y_test, s=20)
plt.plot(X_test, Y_pred, "r-", linewidth=5)
for idx, m in enumerate(X_test):
    plt.plot([m, m], [Y_test[idx], Y_pred[idx]], 'g-')
plt.ylabel("Prices")
plt.xlabel("LSTAT")
plt.show()
mse = sklearn.metrics.mean_squared_error(Y_test, Y_pred)
print(mse)
