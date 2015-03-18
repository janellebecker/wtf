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
from dateutil import parser #use this to ensure dates are parsed correctly 

main_dir = "C:/Users/J/Desktop/data/"


#import data----------------------- ----------header = row 0. if not header, header=null
df = pd.read_csv(main_dir + "sample_30min.csv", header=0, parse_dates=[1], date_parser=parser.parse)

df_assign = pd.read_csv(main_dir + "sample_assignments.csv", usecols =[0,1])

#merge--------------------------------------------
df = pd.merge(df, df_assign)
df.head()

# add/drop variables -----------------------------
type('date')#str
type(df['date'][0]) # datetime.datetime...so we can use lambda do check the date and take the year from it

df['year'] = df['date'].apply(lambda x: x.year)
df['month'] = df['date'].apply(lambda x: x.month)
df['day'] = df['date'].apply(lambda x: x.day)
df['ymd'] = df['date'].apply(lambda x: x.date())

df.head()

# daily aggregation AGGREGATE ON DAY
# grp = df.groupby(['year', 'month', 'day', 'panid', 'assignment'])
grp = df.groupby(['ymd', 'panid', 'assignment']) # equivalent 

df1 = grp['kwh'].sum().reset_index() #daily agg of kwh - COLLAPSING hourly data to the data level

# PIVOT DATA: RESHAPING DATA IN PYTHON (super easy in stata)-------------------------
#go from long to wide

##1 create column names for wide data
# create string names and denote consumption and date
# use ternery expression: [true-expr(x) if condition else false-exp(x) for x in list] 
# this is for the long version of grouping
#df1['day_str'] = ['0' + str(v) if v < 10 else str(v) for v in df1['day']] #add 0 to less than 10
#df1['kwh_ymd'] = 'kwh_' + df1.year.apply(str) + '_' + df1.month.apply(str) + '_' + df1.day_str.apply(str)

df1['kwh_ymd'] = 'kwh_' + df1['ymd'].apply(str)
df1.head()


# 2. PIVOT aka long to wide  #we lost everything that is time invariant (demographics) gotta re-assign
#from the long dataset, pivot(i, j, value) the i is the consumer, the j is the new variable, the value you want in that cell

df1_piv = df1.pivot('panid', 'kwh_ymd', 'kwh')

# clean up for making things pretty :) 
#right now panid is acting as index. we want it to be its own variable and reset index
df1_piv.reset_index(inplace=True) #this makes panid its own variable
df1_piv
df1_piv.columns.name = None
df1_piv

# MERGE TIME INVARIANT DATA TO OUR WIDE DATASET NOW-----------------------
#df2 = pd.merge(df1_piv, df_assign) # this will tack on the assignments last
df2 = pd.merge(df_assign, df1_piv)
df2.head()

## EXPORT THIS DATA SET (FOR REGRESSION LATER ON!)
df2.to_csv(main_dir + "07_kwh_wide.csv", sep = ",", index=False)
