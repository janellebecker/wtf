## I am continuing on from the script using in class on March 4th (before spring break)

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


# IMPORT DATA --------------------------------------------------------------
df = pd.read_csv(main_dir + "07_kwh_wide.csv", header=0)

# SIMPLE LINEAR PROBABILITY MODEL (LPM) ------------------------------------
## crude. not logic or probit.  fails in the extremes. something about negative values? 
## good easing into probabilty analysis
## Let's see if consumption before a certain date "determines" your assignment

df['T'] = 0 + (df['assignment'] == 'T')  #cool way to make 1's and 0's for T/C

df[['assignment', 'T']] #checking it

# SET UP DATA ----------------------------------------------------------------

## get X matric (left hand variables for our regression)
## want a list of all the values that have to do with consumption for regression
type(df.columns)
type(df.columns.values) #in case you need the values and it isn't working (both might work  in this case)

kwh_cols = [v for v in df.columns.values if v.startswith('kwh')]

## degrees of freedom? 5 ppl, 4 on RHS. but column of ones. 
## so up to 3 variables for regression to work

## pretend that the treatment occured in 2015-01-04. we want dates before.
## how to pull info if the ending is 01-01, 01-02, or 01-03?

v = 'kwh_2015-01-31'
v[0:3]
v[0:]
v[-2:] #start from the back. -2 is the third one in. then it'll count forward
v[-6: -1]
v[-6:]

kwh_cols = [v for v in kwh_cols if int(v[-2:]) < 4] #if youw ant to specifically pull (esp if things aren't in order)

kwh_cols[0:3] # would work if it's in order 

## set up y and x variables in regression
y = df['T']
X = df[kwh_cols]
X = sm.add_constant(X)

# RUN OLS ---------------------------------------------------------
ols_model = sm.OLS(y, X) #lpm. stored as an object. get stuff? run functions
ols_results = ols_model.fit()

print(ols_results.summary())  #if you dont use "print", it looks stupid 






