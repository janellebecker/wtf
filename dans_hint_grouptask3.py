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
df = pd.read_csv(root + assign, header=0)


df['tariff'].values # I see its an array?

# Extract Only Control Group using Boolean Indexing 
    #vector has to be the same length as the dataframe. 
df.head()

df['tariff'] == 'E'   # returns vector of true/false when tariff equals E

indx = df['tariff'] == 'E' 

df[indx]  ############this will only extract the times where it was true (what i wanted)

# Extract Control or Treatment A1 (EE or A and 1) how to link boolean statements 
indx1 = df['tariff'] == 'A'

indx2 = ((df['tariff'] == 'A') & (df['stimulus'] == '1'))

indx3 = ((df['tariff'] == 'E') | ((df['tariff'] == 'A') & (df['stimulus'] == '1')))

indx | indx2  #yields the same as the indx 3 combination statement 
df[indx | indx2]

## SHORTCUT 
df['treatment'] = df['tariff'] + df['stimulus'] #concat'ing strings
df.head()

#easier boolean statements now

indx4 = ((df['treatment'] == 'E') | (df['treatment'] == 'B3'))
df[indx4]  ##########this returns the rows from the dataframe only where that logic is true


#### other questions i had  - why we do [df[df stuff

dfEE = df[df['treatment']=='E']]  
# this is a boolean statement like df[indx4], where its returning the parts of 
# df that only have true for that boolean statement !!!!! ooohhhhhh



## to concatenate arrays - combine lists. with .tolist() 

#masterlist = array1.tolist() + array2.tolist() ... + arrayk.tolist()
# turn this into a dataframe:
# DataFrame(masterlist, columns = ['columname']

## EXAMPLE OF LOGIT (for example, with EE and A1)

dfEEA1 = pd.concat([dfEE, dfA1], axis=0) #would stack them together 

dfEEA1['T'] = 0 + dfEEA1['treatment'] != 'EE' #create column of 1/0's for T/C


#-----------------------------------------------------------------------------
#video # 2 
















