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
import statsmodels.api as sm #package developed to do stats analyses
from datetime import datetime, timedelta
from dateutil import parser #use this to ensure dates are parsed correctly 


main_dir = "C:/Users/J/Desktop/data/"
root = main_dir + "data08/"
paths = [root + v for v in os.listdir(root) if v.startswith("08_")]

## logit will be a tool to check for balance - we don't want your demographics to predict your T/C status 

# IMPORT AND ADD/DROP VARIABLES -----------------------
df = pd.read_csv(paths[1], header=0, parse_dates=[1], date_parser=np.datetime64)  #this date parser is much faster than before.
df.head()
df_assign = pd.read_csv(paths[0], header=0)

df['year'] = df['date'].apply(lambda x: x.year)
df['month'] = df['date'].apply(lambda x: x.month)

# MONTHLY AGGREGATION ---------------------------------
grp = df.groupby(['year', 'month', 'panid',])
df = grp['kwh'].sum().reset_index()

## groupby will make the year, month, panid the index. the only thing is a series which is kwh. 
##reset index will make those(grouped by things) into series and create a row index.
##turns a series into a dataframe

## merge time invariant information after pivoting! 

# PIVOT THE DATA --------------------------------------------
df['mo_str'] = ['0' + str(v) if v < 10 else str(v) for v in df['month']] 
## add 0's to numbers 0-09 so that numbers get ordered correctly. don't want 1 (jan) next to 10 (october)
df.head()
df['kwh_ym'] = 'kwh_' + df.year.apply(str) + "_" + df.mo_str.apply(str) ## make column 
df.head()

df_piv = df.pivot('panid', 'kwh_ym', 'kwh')  #recall df.pivot(i, j, value in the cells) i is the person, j is the columns names, value for (i, j) pairing. 
df_piv.head()
df_piv.reset_index(inplace=True)
df_piv.columns.name = None #get rid of "kwh_ym" name of the columns over the index column...
df_piv.head()

## MERGE THE STATIC VALUES AKA TIME INVARIANT INFORMATION ----------------
df = pd.merge(df_assign, df_piv)
del df_piv, df_assign

# GENERATE DUMMY VARIABLES FROM QUALITATIVE DATA (e.g. categories)

df1 = pd.get_dummies(df, columns= ['gender']) # will make dummy vectors for all object or category types without having to 
## tell it sepcifically for which variables. default is for all strings. be caferful!  also, it removes the original with F's and M's
df1.head()

# avoid the dummy variable trap--> you don't need both male and female columns
df1.drop(['gender_M'], axis=1, inplace=True)

## SET UP THE DATA FOR LOGIT ---------------------------------------------
## pretend it started in the third month.
kwh_cols = [v for v in df1.columns.values if v.startswith('kwh')]
kwh_cols = [v for v in kwh_cols if int(v[-2:]) < 4]

cols = ['gender_F'] + kwh_cols

## SET UP Y, X
y = df1['assignment']
X = df1[cols]
X = sm.add_constant(X)

# LOGIT ------------------------------------------------------------
logit_model = sm.Logit(y, X)
logit_results = logit_model.fit()
print(logit_results.summary())

## good news: none of these are significant. this means that none of these would determine the T or C status. this means good randomization. 

