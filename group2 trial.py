from __future__ import division
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import csv
import xlrd
import time
import matplotlib.pyplot as plt
import pytz
from datetime import datetime, timedelta

print(time.ctime())
main_dir = "C:/Users/J/Desktop/data"
root = main_dir + "/" + "cerdata/"
assignfile = "SME and Residential allocations.xlsx"
timeseriescorr = "timeseries_correction.csv"
paths = [os.path.join(root, v) for v in os.listdir(root) if v.startswith("File")]

#How is File1 delimited?  A: space delimited
N = 10
with open(root + "/" "File1.txt") as myfile:
        head = [next(myfile) for x in xrange(N)]
print head

#If I didn't do the above and just wanted to know what each column was
dfpeek = pd.read_table(os.path.join(root + "File1.txt"))
dfpeek.head()

#IMPORT AND STACK THE FILES
df = pd.concat([pd.read_table(v, names = ['panid', 'date', 'kwh'], sep = " ", nrows = 1.5*10**6) 
    for v in paths], ignore_index = True)

# GET ASSIGNMENT AND TREATMENT INFORMATION 
assignment = pd.read_excel(root + assignfile, sep = ' ', na_values=[' ', '-', 'NA'], usecols=range(0,4))
assignment = assignment[assignment.Code==1] #I only want residential, which is code ==1.
assignment = assignment[[0,2,3]] #Don't need "Code" column anymore 
assignment.columns = ['panid', 'tariff', 'stimulus']

# We only want Control Residential users (E,E) or Bi-monthly bill and Stimulus A users (A, 1)
keeprows = ((assignment.tariff =='E') & (assignment.stimulus=='E')) | ((assignment.tariff=='A') & (assignment.stimulus=='1'))
assignment = assignment[keeprows]


# Merge with panel data.
df = pd.merge(df,assignment, on = ['panid'])
#df_monthly = pd.merge(df_monthly,assignment, on = ['panid'])
del [assignment, keeprows]

# Group variables on panid and day, then sum consumption across each day.
df['hour_cer'] = (df.time % 100)
df['day_cer'] = (df.time - df['hour_cer'])/100
del df['time']
# Pull in timestamps
tscorr = pd.read_csv(main_dir+"/"+timeseriescorr, header=0, parse_dates=[1])
tscorr = tscorr[['year','month','day','hour_cer','day_cer']]

df = pd.merge(df,tscorr, on=['day_cer','hour_cer'])
del tscorr
del [[df['day_cer'],df['hour_cer']]]

# Aggregate on day
daygrp = df.groupby(['panid','tariff','year','month','day'])
df_daily= daygrp['kwh'].sum().reset_index()

# Aggregate on month
monthgrp = df.groupby(['panid','tariff','year','month'])
df_monthly = monthgrp['kwh'].sum().reset_index()
del df
del [daygrp, monthgrp]

# Group on treatment status and day
grp_daily = df_daily.groupby(['tariff','year','month','day'])
trt_daily = {(k[1],k[2],k[3]): df_daily.kwh[v].values for k,v in grp_daily.groups.iteritems() if k[0]=="A"} 
ctrl_daily = {(k[1],k[2],k[3]): df_daily.kwh[v].values for k,v in grp_daily.groups.iteritems() if k[0]=="E"}
del [df_daily, grp_daily]

# Group on treatment status and month
grp_monthly = df_monthly.groupby(['tariff','year','month'])
trt_monthly = {(k[1],k[2]): df_monthly.kwh[v].values for k,v in grp_monthly.groups.iteritems() if k[0]=="A"} 
ctrl_monthly = {(k[1],k[2]): df_monthly.kwh[v].values for k,v in grp_monthly.groups.iteritems() if k[0]=="E"}
del [df_monthly, grp_monthly]

keys_daily = trt_daily.keys()
keys_monthly = trt_monthly.keys()

# create dataframes of tstats over time
tstats_daily = DataFrame([(k[0], k[1], k[2], np.abs(ttest_ind(trt_daily[k],ctrl_daily[k], equal_var=False)[0])) for k in keys_daily], columns=['year','month','day','tstat'])
pvals_daily  = DataFrame([(k[0], k[1], k[2], np.abs(ttest_ind(trt_daily[k],ctrl_daily[k], equal_var=False)[1])) for k in keys_daily], columns=['year','month','day','pval'])


