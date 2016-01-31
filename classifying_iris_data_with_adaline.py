# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mystuff import adaline
from mystuff import plotting_stuff

plt.close('all')

df = pd.read_csv('./iris.data', header=None)
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values
fig1 = plt.figure()
initial_data_plot = fig1.add_subplot(111)
initial_data_plot.set_title('initial data')
initial_data_plot.set_xlabel('petal length')
initial_data_plot.set_ylabel('sepal length')
initial_data_plot.legend(loc='upper left')
initial_data_plot.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
initial_data_plot.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')

fig2, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

ada1 = adaline.AdalineGD(n_iter=10, eta=0.01).fit(X, y)
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
ax[0].set_title('Adaline -- learning rate 0.01')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-Squared-Error)')

ada2 = adaline.AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.cost_) + 1), np.log10(ada2.cost_), marker='o')
ax[1].set_title('Adaline -- learning rate 0.0001')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('log(Sum-Squared-Error)')

X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

ada = adaline.AdalineGD(n_iter=15, eta=0.01)
ada.fit(X_std, y)
plotting_stuff.plot_decision_regions(X_std, y, classifier=ada)

fig3 = plt.figure()
cost_plot = fig3.add_subplot(111)
cost_plot.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
cost_plot.set_xlabel('Epochs')
cost_plot.set_ylabel('Sum-Squared-Error')

plt.show()
