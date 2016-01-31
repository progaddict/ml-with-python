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

plt.show()
