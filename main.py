import numpy as np
import pandas as pd

df = pd.read_csv("../input/population-vs-profit-from-andrew-ng-coursera/pop_profit.txt")
df.head()

df.columns = df.columns.str.strip()
df.shape
df.describe()

from matplotlib import pyplot
df.hist()
pyplot.show()
df.head(20)

df.shape
df.describe()
import matplotlib.pyplot as plt
plt.scatter(df['population'], df['profit'], color='r')

x = df[['population']].values
y = df['profit'].values
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x, y)
m = lr.coef_
c = lr.intercept_
y = m*x + c
yx = df[['population']]
y = df['profit']
plt.scatter(x, y, color='b')
plt.plot(x, lr.predict(x), color='r')
plt.xlabel('population')
plt.ylabel('profit')

lr.predict([[15]])
