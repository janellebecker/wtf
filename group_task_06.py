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

# MERGE CONSUMPTION DATA WITH ASSIGNMENT INFO
df = pd.merge(df, assignment, on = ['panid'])
df.head()
del [assignment, keeprows]


# GROUP VARIABLES ON PANID AND DAY. SUM UP DAILY CONSUMPTION.

# Make day and hour variables
df['hour_cer'] = (df.date % 100)
df['day_cer'] = (df.date - df['hour_cer'])/100
del df['date']

# bring in the time correction csv so that when we inner merge df with it, we keep the corrected versions
tscorr = pd.read_csv(root + timeseriescorr, header=0, parse_dates=[1]) 
tscorr.head()

#make year and month variables
tscorr['year'] = tscorr['ts'].apply(lambda x: x.year)
tscorr['month'] = tscorr['ts'].apply(lambda x: x.month)

#keep only these 
tscorr = tscorr[['year', 'month', 'day', 'hour_cer', 'day_cer']]
tscorr.head()
df.head()
df = pd.merge(df, tscorr, on=['hour_cer', 'day_cer'])

df.head()


# AGGREGATE DAILY CONSUMPTION (note: A or E indicates treatment or control....lets keep that)
daily_grp = df.groupby(['panid', 'tariff', 'year', 'month', 'day'])
keys1 = daily_grp.groups.keys()
daily_agg = daily_grp['kwh'].sum().reset_index()
daily_agg.head()

daily_agg['kwh_daily'] = daily_agg['kwh']
df_daily = daily_agg[[0,1,2,3,4,6]]
del daily_agg
df_daily.head()


#AGGREGATE MONTHLY CONSUMPTION
monthly_grp = df.groupby(['panid', 'tariff', 'year', 'month'])
keys2 = monthly_grp.groups.keys()
monthly_agg = monthly_grp['kwh'].sum().reset_index()
monthly_agg.head()

monthly_agg['kwh_monthly'] = monthly_agg['kwh']
df_monthly = monthly_agg[[0,1,2,3,5]]
del monthly_agg
df_monthly.head()


#GROUP ON TREATMENT STATUS AND DAY
#first, group on day. second, group on treatment status

daygroup = df_daily.groupby(['year', 'month', 'day', 'tariff'])
keys3 = daygroup.groups.keys()

trt_daily = {(k[0], k[1], k[2]): df_daily.kwh_daily[v].values for k, v in daygroup.groups.iteritems() 
    if k[1]=='A'}
ctrl_daily = {(k[0], k[1], k[2]): df_daily.kwh_daily[v].values for k, v in daygroup.groups.iteritems() 
    if k[1]=='E'}


#GROUP ON TREATMENT STATUS AND MONTH 
monthgroup = df_monthly.groupby(['year', 'month', 'tariff'])
keys4 = monthgroup.groups.keys()
trt_monthly = {(k[0], k[1]): df_daily.kwh_monthly[v].values for k, v in daygroup.groups.iteritems() 
    if k[2]=='A'}
ctrl_monthly = {(k[0], k[1]): df_daily.kwh_monthly[v].values for k, v in daygroup.groups.iteritems() 
    if k[2]=='E'}








#why is keys_daily blank? 


keys_daily = trt_daily.keys()
keys_monthly = trt_monthly.keys()
#the control groups have the exact same keys...so it doesn't matter which we use. 

# CREATE DATAFRAMES OF T-STATS AND P-VALUES 
# t = ttest_ind(list a, list b, equal_var = False)  #output (t stat, twotail pvalue)


tstats_daily = DataFrame([(k[0], k[1], k[2], np.abs(float(ttest_ind(trt_daily[k], ctrl[k], equal_var=False)[0])))
    for k in keys_daily], columns=['year', 'month', 'day', 'tstat'])

pvals_daily = DataFrame([(k[0], k[1], k[2], np.abs(float(ttest_ind(trt_daily[k], ctrl[k], equal_var=False)[1]))) 
    for k in keys_daily], columns=['year', 'month', 'day', 'pvals'])
t_p_daily = pd.merge(tstats_daily, pvals_daily)





