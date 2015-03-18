from __future__ import division
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from scipy.special import stdtr

main_dir = "C:/Users/J/Desktop/data/"
root = main_dir + "data_06_agg_plot" + "/"

#IMPORT AND MERGE DATA--------------------------------------
df = pd.read_csv(root + "sample_30min.csv", header=0, parse_dates=[1])
df_assign = pd.read_csv(root + "sample_assignments.csv", usecols=[0,1])

df = pd.merge(df, df_assign)

#data is over 30 min intervals instead of for a whole day. we want m/d/y to aggregate, not 30 min level 

df['date'][0]
df['date'][0].day
df['date'][0].month
df['date'][0].year

#NEW VARIABLES  and anonymous functions--------------------------------
#lambda is starting anon fxn. apply it to everyting in date series. call it x. what to do to x?
#take every internal value
 
df['year'] = df['date'].apply(lambda x: x.year)
df['month'] = df['date'].apply(lambda x: x.month)
df['day'] = df['date'].apply(lambda x: x.day)

df                 ##created new variable and columns. now we can aggregate/collapse on the day or month. 

# AGGREGATION (DAILY) collapsing ------------------------------------------
grp = df.groupby(['year', 'month', 'day', 'panid', 'assignment'])
grp.groups                     #this is looking into a dictionary. don't do this with big data!
agg = grp['kwh'].sum()         #added up usage for a day for panid 1. then again for next panid. 

#RESET THE INDEX ---------------
agg = agg.reset_index()
agg.head()

# let's collapse all Control or Treatment group per day to look at that. 
grp1 = agg.groupby(['year', 'month', 'day', 'assignment'])
grp1.head()
grp1.groups

##split up treatment and control 
trt = {(k[0], k[1], k[2]): 
agg.kwh[v].values for k, v in grp1.groups.iteritems() if k[3]=='T'}

ctrl = {(k[0], k[1], k[2]): 
agg.kwh[v].values for k, v in grp1.groups.iteritems() if k[3]=='C'}
keys = ctrl.keys()

# tstats and pvals - PLOTTING-----------------

tstats = DataFrame([(k, np.abs(float(ttest_ind(trt[k], ctrl[k], equal_var=False)[0]))) 
    for k in keys], columns=['ymd', 'tstat'])
pvals = DataFrame([(k, np.abs(float(ttest_ind(trt[k], ctrl[k], equal_var=False)[1]))) 
    for k in keys], columns=['ymd', 'pval'])
t_p = pd.merge(tstats, pvals)

##sort and reset index
t_p.sort(['ymd'], inplace=True)
t_p.reset_index(inplace=True, drop=True)

## PLOTTING -----------------------------------------------------
fig1 = plt.figure()
ax1 = fig1.add_subplot(2,1,1)
ax1.plot(t_p['tstat'])
ax1.axhline(2, color='r', linestyle='--')
ax1.axvline(14, color='g', linestyle='-')
ax1.set_title('T-Stats Over Time')

ax2 = fig1.add_subplot(2,1,2)
ax2.plot(t_p['pval'])
ax2.axhline(0.05, color='r', linestyle='--')
ax2.axvline(14, color='g', linestyle='-')
ax2.set_title('P-Values Over Time')
plt.show()