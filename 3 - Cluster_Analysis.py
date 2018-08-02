# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 10:18:20 2018

@author: p695032
"""

import pandas as pd
import math as mt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sklearn.datasets as dt
import hdbscan
import os
from scipy.stats import itemfreq

#define the folder and file
path="//aur.national.com.au/User_Data/AU-VIC-MELBOURNE-3COL1-02/UserData/P695032/Python/Study/Rides"
os.chdir(path)


file_name = "prep_train_data.csv"

#import the file
df = pd.read_csv(file_name, sep='\t', encoding='utf-8')
df = df.drop(df.columns[[0,1]], axis=1)

##### Model 1

clusterer_1 = hdbscan.HDBSCAN(min_cluster_size=2000)
cluster_labels_m1 = clusterer_1.fit_predict(df)

itemfreq(cluster_labels)

# Create a KMeans instance with 3 clusters: model
model = KMeans(n_clusters = 3)

# Fit model to points
model.fit(df)

# Determine the cluster labels of new_points: labels
labels = model.predict(df)

itemfreq(labels)