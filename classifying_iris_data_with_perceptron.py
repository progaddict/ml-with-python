# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mystuff import perceptron
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

ppn = perceptron.Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
fig2 = plt.figure()
errors_plot = fig2.add_subplot(111)
errors_plot.set_title('errors during fitting')
errors_plot.set_xlabel('Epochs')
errors_plot.set_ylabel('Number of misclassifications')
errors_plot.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')

plotting_stuff.plot_decision_regions(X, y, classifier=ppn)

plt.show()
