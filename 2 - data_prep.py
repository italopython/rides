# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 10:49:36 2018

@author: p695032
"""

import pandas as pd
import math as mt
import numpy as np
from math import radians, sin, cos, acos
import matplotlib.pyplot as pl
from sklearn import preprocessing
import os

#define the folder and file
path="//aur.national.com.au/User_Data/AU-VIC-MELBOURNE-3COL1-02/UserData/P695032/Python/Study/Rides"
os.chdir(path)

file_name = "train_sample.csv"

#import the file
data = pd.read_csv(file_name, sep='\t', encoding='utf-8')

#define 3 geo reference points
#==============================================================================
# ref_geo_1 = pd.DataFrame({'lat' : [40.996335],'long' :  [-73.979714]})
# ref_geo_2 = pd.DataFrame({'lat' : [40.639078],'long' :  [-74.318230]})
# ref_geo_3 = pd.DataFrame({'lat' : [40.641162],'long' :  [-73.933709]})
# 
#==============================================================================

data['distance'] = 6371.01 * 2 * np.arcsin(np.sqrt(
        np.sin((np.radians(data['dropoff_latitude'])-np.radians(data['pickup_latitude']))/2.0)**2 + \
        np.cos(np.radians(data['pickup_latitude'])) * np.cos(np.radians(data['dropoff_latitude'])) *
        np.sin((np.radians(data['dropoff_longitude'])-np.radians(data['pickup_longitude']))/2.0)**2))


# delete invalid distances

data = data.loc[(data['distance'] > 0)]

# calculate supporting distances

data['distance_p_1'] = 6371.01 * 2 * np.arcsin(np.sqrt(
        np.sin((np.radians(40.996335)-np.radians(data['pickup_latitude']))/2.0)**2 + \
        np.cos(np.radians(data['pickup_latitude'])) * np.cos(np.radians(40.996335)) *
        np.sin((np.radians(-73.979714)-np.radians(data['pickup_longitude']))/2.0)**2))


data['distance_p_2'] = 6371.01 * 2 * np.arcsin(np.sqrt(
        np.sin((np.radians(40.639078)-np.radians(data['pickup_latitude']))/2.0)**2 + \
        np.cos(np.radians(data['pickup_latitude'])) * np.cos(np.radians(40.639078)) *
        np.sin((np.radians(-74.318230)-np.radians(data['pickup_longitude']))/2.0)**2))


data['distance_p_3'] = 6371.01 * 2 * np.arcsin(np.sqrt(
        np.sin((np.radians(40.641162)-np.radians(data['pickup_latitude']))/2.0)**2 + \
        np.cos(np.radians(data['pickup_latitude'])) * np.cos(np.radians(40.641162)) *
        np.sin((np.radians(-73.933709)-np.radians(data['pickup_longitude']))/2.0)**2))
        
#normalise the data
        
RobustScaler = preprocessing.RobustScaler()

data['ns_distance'] = RobustScaler.fit_transform(data['distance'].values.reshape(-1,1))
data['ns_distance_p_1'] = RobustScaler.fit_transform(data['distance_p_1'].values.reshape(-1,1))        
data['ns_distance_p_2'] = RobustScaler.fit_transform(data['distance_p_2'].values.reshape(-1,1))        
data['ns_distance_p_3'] = RobustScaler.fit_transform(data['distance_p_3'].values.reshape(-1,1))        
                        


#==============================================================================
# data['distance'] = np.arccos(
#                            np.sin(np.radians(90-(data['pickup_latitude'])))
#                            np.sin(np.radians(90-(data['dropoff_latitude'])))
#                            +
#                            (np.cos(np.radians(90-(data['pickup_latitude'])))*
#                            np.cos(np.radians(90-(data['dropoff_latitude']))))
#                            *
#                            (np.cos(np.radians(data['pickup_longitude']-data['dropoff_longitude'])))
#                          )*6371.01
#==============================================================================

#==============================================================================
# data.describe()
#==============================================================================

data['date'] = pd.DatetimeIndex(pd.to_datetime(data['pickup_datetime'].str.replace(' UTC',''))).date

data['year'] = pd.DatetimeIndex(data['date']).year

data['month'] = pd.DatetimeIndex(data['date']).month

data['day'] = pd.DatetimeIndex(data['date']).day

data['weekday'] = pd.DatetimeIndex(data['date']).weekday

data['hour'] = pd.DatetimeIndex(pd.to_datetime(data['pickup_datetime'].str.replace(' UTC',''))).hour

data['pickup_range'] = np.where((data['hour'] >= 6) & (data['hour'] < 8),'BH',
               np.where((data['hour'] >= 8) & (data['hour'] <= 11), 'MW',
               np.where((data['hour'] > 11) & (data['hour'] <= 13), 'LB',
               np.where((data['hour'] > 13) & (data['hour'] <= 17), 'AW','AH'))))          
               

       
#==============================================================================
# cr_hr = pd.get_dummies(data['hour'], drop_first=True)    
# cr_wd = pd.get_dummies(data['weekday'], drop_first=True)   
# cr_pr = pd.get_dummies(data['pickup_range'], drop_first=True)   
#==============================================================================

list(data)
#==============================================================================
#==============================================================================
# ['Unnamed: 0',
#  'key',
#  'fare_amount',
#  'pickup_datetime',
#  'pickup_longitude',
#  'pickup_latitude',
#  'dropoff_longitude',
#  'dropoff_latitude',
#  'passenger_count',
#  'distance',
#  'distance_p_1',
#  'distance_p_2',
#  'distance_p_3',
#  'ns_distance',
#  'ns_distance_p_1',
#  'ns_distance_p_2',
#  'ns_distance_p_3',
#  'date',
#  'year',
#  'month',
#  'day',
#  'weekday',
#  'hour',
#  'pickup_range']
#==============================================================================
# 
#==============================================================================
var = [
 'fare_amount',
 'passenger_count',
 'ns_distance',
 'ns_distance_p_1',
 'ns_distance_p_2',
 'ns_distance_p_3',
#==============================================================================
#  'date',
#==============================================================================
 'year'
]

#concactenate the data and exclude NaN
#==============================================================================
# data = pd.concat([data[var], cr_wd,cr_pr], axis=1, join_axes=[data.index]).loc[~np.isnan(data['dropoff_latitude'])]
#==============================================================================
data = pd.concat([data[var], 
                  pd.get_dummies(data['hour'], drop_first=True) ,
                  pd.get_dummies(data['weekday'], drop_first=True),
                  pd.get_dummies(data['month'], drop_first=True),
                  pd.get_dummies(data['pickup_range'], drop_first=True)],
                  axis=1, join_axes=[data.index]).loc[(data['distance'] > 0)]

# export the data to csv
file_name = "prep_train_data.csv"
data.to_csv(file_name, sep='\t', encoding='utf-8')

pd.crosstab(data['hour'],data['pickup_range'])

data['hour'].value_counts()
#==============================================================================
# 
# Out[255]: 
# 19    17187
# 18    16627
# 20    16304
# 21    15841
# 22    15297
# 14    13961
# 23    13842
# 13    13664
# 17    13642
# 15    13362
# 12    13286
# 11    13156
# 9     12809
# 8     12777
# 10    12539
# 16    11469
# 0     10971
# 7      9948
# 1      8140
# 2      6101
# 6      5792
# 3      4556
# 4      3147
# 5      2700
#==============================================================================

pd.crosstab(data['hour'],data['pickup_range'])

#==============================================================================
#==============================================================================
# Out[3]: 
# pickup_range     AH     AW    BH     LB     MW
# hour                                          
# 0             10971      0     0      0      0
# 1              8140      0     0      0      0
# 2              6101      0     0      0      0
# 3              4556      0     0      0      0
# 4              3147      0     0      0      0
# 5              2700      0     0      0      0
# 6                 0      0  5792      0      0
# 7                 0      0  9948      0      0
# 8                 0      0     0      0  12777
# 9                 0      0     0      0  12809
# 10                0      0     0      0  12539
# 11                0      0     0      0  13156
# 12                0      0     0  13286      0
# 13                0      0     0  13664      0
# 14                0  13961     0      0      0
# 15                0  13362     0      0      0
# 16                0  11469     0      0      0
# 17                0  13642     0      0      0
# 18            16627      0     0      0      0
# 19            17187      0     0      0      0
# 20            16304      0     0      0      0
# 21            15841      0     0      0      0
# 22            15297      0     0      0      0
# 23            13842      0     0      0      0
#==============================================================================
#==============================================================================




