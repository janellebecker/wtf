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

main_dir = "C:/Users/J/Desktop/data/"
root = main_dir + "grouptask4/"

# CHANGE WORKING DIRECTORY (wd)-------------------------------------
os.chdir(root)

from logit_functions import *  #import all functions with their names in tact

# do_logit() # see that its there
# in the bottom section, do_logit?? # tells you what you can do, the arguments, etc. 

# IMPORT DATA ---------------------------------------------------
df = pd.read_csv(root + "14_B3_EE_w_dummies.csv")
df = df.dropna(axis=0, how='any') #this is default setting 
# dropped from 1033 to 851 rows

# GET TARIFFS ------------------------------------------------------

tariffs = [v for v in df['tariff'].unique() if v != 'E']
stim = [v for v in df['stimulus'].unique() if v != 'E']

tariffs.sort()
stim.sort()

# LOGIT MODELS ---------------------------------------------
# 2009 values are pre-trial. treatment begins 2010. 
#i dont want to use 2010 for checking for balance, just the 2009 values. 
df.head()
# keep the dataframe, drop the values from treatment 2010

df_drop = [v for v in df.columns if v.startswith("kwh_2010")]
df_pretrial = df.drop(df_drop, axis=1)
df_pretrial.head()

for i in tariffs:
    for j in stim:
        logit_results, df_logit = do_logit(df_pretrial, i, j, add_D=None, mc=False)
        
##logit results, make a new df with the results
##if you want to add dummies, you could put it there with add_D. otherwise it'll search itself
##dummies must start with "D_" and consmp vars with "kwh_"
##specifically for our class
##mc is for multicollinearity. problem for significance, not unbiased

#df_logit has all the information that was kept, 837 instead of 851, original was 1033.

#if there were high MC...? time 21 min

#QUICK MEANS COMPARISON WITH T-TEST BY HAND -----------------------------------
grp = df_logit.groupby('tariff')
df_mean = df_logit.groupby('tariff').mean()
df_mean = grp.mean().transpose() #much nicer to look at

df_mean.B - df_mean.E #can see the differences between means, but lets do t-tests

# DO T-TEST "BY HAND" AKA HARD CODE ---------------------------------
df_s = grp.std().transpose()
df_n = grp.count().transpose().mean() #mean by column
top = df_mean['B'] - df_mean['E']
bottom = np.sqrt(df_s['B']**2/df_n['B'] + df_s['E']**2/df_n['E'])
tstat = top/bottom
sig = tstat[np.abs(tstat) > 2]
sig.name = 't-stats'

print logit_results.summary()





