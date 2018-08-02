# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 08:18:04 2018

@author: Italo Paiva
"""


import pandas as pd
import random
import os

#working directory

os.getcwd()

path="//aur.national.com.au/User_Data/AU-VIC-MELBOURNE-3COL1-02/UserData/P695032/Python/Study/Rides"
os.chdir(path)

#load the data
f = "train.csv"

#count the line
num_lines = sum(1 for l in open(f))

#sample size - 0.5%
size = int(num_lines/200)

skip_idx = random.sample(range(1,num_lines), num_lines - size)

# read the data
data = pd.read_csv(f,skiprows=skip_idx)

# export the data to csv
file_name = "train_sample.csv"
data.to_csv(file_name, sep='\t', encoding='utf-8')
