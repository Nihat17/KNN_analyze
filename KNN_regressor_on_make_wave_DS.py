#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 17:17:58 2018

@author: Nihat Allahverdiyev
"""
from sklearn.neighbors import KNeighborsRegressor
import mglearn.datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
# load the data
X, y = mglearn.datasets.make_wave(n_samples = 40)

#plt.plot(X, y, 'o', label = "Data")

# split the data 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

# instantiate the model and set the number of neighbors to 3
reg = KNeighborsRegressor(n_neighbors = 3)

# train the model
reg.fit(X_train, y_train)

# Now we can make predictions
print("Test set predictions: \n{}".format(reg.predict(X_test)))

# display the model accuracy
print("Accuracy: {:.2f}".format(reg.score(X_test,y_test)))

fig, axes = plt.subplots(1, 3, figsize = (20, 4))

# create 1000 data points , evenly spaced between -3, 3
line = np.linspace(-3, 3, 1000).reshape(-1, 1)

for n_neighbors, ax in zip([1, 3, 9], axes):
    # make predictions using 1, 3, or 9 neighbors
    reg = KNeighborsRegressor(n_neighbors = n_neighbors)
    reg.fit(X_train, y_train)
    ax.plot(line, reg.predict(line))
    ax.plot(X_train, y_train, '^', c=mglearn.cm2(0), markersize = 8)
    ax.plot(X_test, y_test, 'v', c=mglearn.cm2(0), markersize=8)
    
    ax.set_title("{} neighbor(s)\n train score: {:.2f} test score".format(
            n_neighbors, reg.score(X_train, y_train),
            reg.score(X_test, y_test)))
    ax.set_xlabel("Feature")
    ax.set_ylabel("Target")
    
axes[0].legend(["Model predictions", "Training data/target",
"Test data/target"], loc="best")
    
