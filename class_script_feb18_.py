from __future__ import division
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os

main_dir = "C:/Users/J/Desktop/data"
root = main_dir + "/" "demo0507" + "/"

#PATHING---------------
paths = [os.path.join(root, v) for v in os.listdir(root) if v.startswith("file_")]

#IMPORT AND STACK ALL IN ONE GO----------
df = pd.concat([pd.read_csv(v, names = ['panid', 'date', 'kwh']) for v in paths], 
ignore_index = True)

#MERGE--------------
df_assign = pd.read_csv(root + "sample_assignments.csv", usecols = [0,1])
df.head()
df_assign.head()
df = pd.merge(df, df_assign)

#dates as strings vs. dates as dates

type(df['date'].values[0])  #string! strings dont have orders. sorts alpha-numeric
df.sort(['date']) #this will not sort by date, but goes 1, 10, 11, 12, 13..4, 
#luckily, CER is integer format, so you can sot them

#help with sorting dates - parse dates 
df = pd.concat([pd.read_csv(v, names = ['panid', 'date', 'kwh'], parse_dates = [1], 
    header = None) for v in paths], ignore_index = True)
df_assign = pd.read_csv(root + "sample_assignments.csv", usecols = [0,1])
df = pd.merge(df, df_assign)

df.sort(['date'])
df.sort(['date', 'panid'])
df.sort(['panid', 'date']) #order matters

grp1 = df.groupby(['assignments'])
keys = grp1.groups.keys()
grp1.groups
gd1 = grp1.groups #dictionary
gd1.keys()
gd1.viewvalues() #each value that appears 

grp1.mean() #pooled, ignoring time, over T and C. 

#(tuple) set in stone together
#[list] super malleable and you can add subrtract, change, etc



grp2 = df.groupby(['assignments', 'date']) 
grp2.groups.keys()
gd2 = grp2.groups
gd2.keys() #equivalent to grp2.groups.keys()

gd2 

#switched over to walking through livedemo07 instead of copy pasting that here again (line 85)

#CREATING THE TREATMENT AND CONTROL GROUP WITH DICTIONARIES
gd2 #first row output {('C', Timestamp('2015-01-01 00:00:00')): [0L, 90L, 120L],
#that first row is being assigned to k and returning that entire type. if k[0] means look if its a T
grp2 = df.groupby(['assignments', 'date']) 
trt = {k[1]: df.kwh[v].values for k, v in grp.groups.iteritems() if k[0] == 'T'}
ctrl = {k[1]: df.kwh[v].values for k, v in grp.groups.iteritems() if k[0] == 'C'}

#in a dictionary, the object on the right side cannot be a series. it has to be a list. 
#this is why we use .values, so that it extracts an array structured thing

grp2 = df.groupby(['date','assignments']) #if the order was diff, my 0/1s would be diff with k[]
trt = {k[0]: df.kwh[v].values for k, v in grp.groups.iteritems() if k[1] == 'T'}


#--------------------went to livedemo07 to edit


