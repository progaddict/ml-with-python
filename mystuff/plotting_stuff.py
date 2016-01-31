# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    fig = plt.figure()
    decision_boundaries_plot = fig.add_subplot(111)
    decision_boundaries_plot.set_title('Decision Boundaries')
    decision_boundaries_plot.set_xlabel('petal length')
    decision_boundaries_plot.set_ylabel('sepal length')
    decision_boundaries_plot.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    decision_boundaries_plot.set_xlim(xx1.min(), xx1.max())
    decision_boundaries_plot.set_ylim(xx2.min(), xx2.max())
    for idx, cl in enumerate(np.unique(y)):
        decision_boundaries_plot.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx],
                                         label=cl)
    return fig
