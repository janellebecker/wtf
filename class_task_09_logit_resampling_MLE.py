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
from dateutil import parser

print(time.ctime())
main_dir = "C:/Users/J/Desktop/data/"
root = main_dir + "logit/"
assign = "allocation_subsamp.csv"
consump = "kwh_redux_pretrial.csv"
df_assign = pd.read_csv(root + assign, header=0)

# Create a vector with the IDs for the control group, and the IDs for each treatment group
df_E = df_assign.ID[(df_assign.tariff == 'E') & (df_assign.stimulus == 'E')]
df_A1 = df_assign.ID[(df_assign.tariff == 'A') & (df_assign.stimulus == '1')]
df_A3 = df_assign.ID[(df_assign.tariff == 'A') & (df_assign.stimulus == '3')]
df_B1 = df_assign.ID[(df_assign.tariff == 'B') & (df_assign.stimulus == '1')]
df_B3 = df_assign.ID[(df_assign.tariff == 'B') & (df_assign.stimulus == '3')]

#Seed the random number generator
np.random.seed(1789)

#Randomly draw from each vector, drawing without replacement
df_E_300 = np.random.choice(df_E, size=300, replace=False, p=None)
df_A1_150 = np.random.choice(df_A1, size=150, replace=False, p=None)
df_A3_150 = np.random.choice(df_A3, size=150, replace=False, p=None)
df_B1_50 = np.random.choice(df_B1, size=50, replace=False, p=None)
df_B3_50 = np.random.choice(df_B3, size=50, replace=False, p=None)

# Create a DataFrame with all the sampled IDs
# Combine these series to form a dataframe
#first, make these arrays into lists. then combine lists 
df_E_300 = np.random.choice(df_E, size=300, replace=False, p=None).tolist()
df_A1_150 = np.random.choice(df_A1, size=150, replace=False, p=None).tolist()
df_A3_150 = np.random.choice(df_A3, size=150, replace=False, p=None).tolist()
df_B1_50 = np.random.choice(df_B1, size=50, replace=False, p=None).tolist()
df_B3_50 = np.random.choice(df_B3, size=50, replace=False, p=None).tolist()

df_SampID = pd.DataFrame(df_E_300 + df_A1_150 + df_A3_150 + df_B1_50 + df_B3_50, columns=['ID'])
df_SampID.head()

# Import consumption data and merge with sampled ID's - Strips away all other consumption data
df_cons = pd.read_csv(root + consump, header=0, parse_dates=[2], date_parser=np.datetime64)
df_cons.head()

# Merge sampled ID's dataframe with consumption data
df = pd.merge(df_SampID, df_cons, on = ['ID'], how = 'inner')
df.reset_index(drop=True, inplace=True) #drop=true will get rid of old index. inplace will replace old dataframe
df.head()

# Compute aggregate monthly consumption for each panel ID.
    #create month variable 
df['month'] = df['date'].apply(lambda x: x.month)
df['year'] = df['date'].apply(lambda x: x.year)
df.head()

    #aggregate consumption for each ID by month 
grp = df.groupby(['year', 'month', 'ID'])
df_agg = grp['kwh'].sum().reset_index()
df_agg.head()

#dont need this, since we're going to pivot and create month specific column names
#df_agg['kwh_month'] = df_agg['kwh']
#df_agg.drop('kwh', axis=1, inplace=True)
#df_agg.head()

# Pivot the data
    # prep to pivot: make kwh_[month] variables for wide dataframe
df_agg['month_str'] = ['0' + str(v) if v < 10 else str(v) for v in df_agg['month']]
df_agg['kwh_ym'] = 'kwh_' + df_agg.year.apply(str) + "_" + df_agg.month_str.apply(str)
df_agg.head()

    # pivot the data from long to wide df.pivot(i, j, value)
df_piv = df_agg.pivot('ID', 'kwh_ym', 'kwh')
df_piv.head()
df_piv.reset_index(inplace=True) #to make the ID not the index 
df_piv.columns.name = None #get reid of kwh_ym over the index 

# Merge the wide dataset with the allocation dataset on ID
df2 = pd.merge(df_assign, df_piv)
df2.head() # df2 is now my dataframe with monthly aggregate info, wide, with T/C info. 
df3 = df2.copy()
# Run a logit model to see if consumption could predict T/C status for each of the 4 T groups
    # Set up variables for logit regression 
    
#how to get a column of 0/1 based on if statements (if E, E, i want Control column of 0/1's) 