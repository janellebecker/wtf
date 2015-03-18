from __future__ import division
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os

main_dir = "C:\Users\J\Desktop\data"

# ADVANCED PATHING ---------------------------------------------------
##constructing paths using patterns

##writing a loop to import all the names for you

root = main_dir + "\demo06\\"
root

paths0 = [root + "file_rand_" + str(v) + ".csv" for v in range(1,5)]
paths0

## %s is place holder %s for string. 
paths1 = [os.path.join(root, "file_rand_%s.csv") % v for v in range(1,5)]
paths1

##could do a combo of the two by using the + sign and the % thing 
paths2 = [root + "file_rand_%s.csv" % v for v in range(1,5)]
paths2

##super pro way to import file based on patterns

##pathing based on starts with 
[v for v in os.listdir(root)] #this lists the file names 
[os.path.join(root, v) for v in os.listdir(root)]
[root + v for v in os.listdir(root)]
[root + v for v in os.listdir(root) if v.startswith("file_")]
[v for v in os.listdir(root) if v.startswith("file_")]

paths3 = [root + v for v in os.listdir(root) if v.startswith("file_")]
paths3

##help function kind of: str. hit tab it'll go through all the functions you can use 


#IMPORT DATA---------------------------------------------------------------
list_of_dfs = [pd.read_csv(v, names = ['panid', 'date', 'kwh']) for v in paths3]

#the columns were missing titles/column names. don't want to use first row as name.
#look at the data. assign column names as a header so that we have a key!

len(list_of_dfs)
type(list_of_dfs)
type(list_of_dfs[0])

## assignment data
## only use the columns you want!!!!
df_assign = pd.read_csv(main_dir + "\sample_assignments.csv", usecols = [0,1])

#usecols is the same as telling  what columns to use 
df_assign = df_assign[[0,1]]

##STACK AND MERGE -----------------------------------------------------
#default is row binds (stacking) auto on key names
#ill renumber the index
##stacking
df = pd.concat(list_of_dfs, ignore_index = True)

##merge
df = pd.merge(df, df_assign) #default will drop anything not intersected

# DROPPING AND CHANGING ROW VALUES --------------------------------
df.drop(9)              #doesn't copy it, will actually change the values of df!!!
df.drop(range(0,9))     #will drop rows 0-8.

df1 = df.copy()
df1.drop(range(0,9), inplace = True)   #will change the dataframe
df1

## changing row values

df.kwh[0]
df.kwh[2:10]
df.kwh[range(0,5)]
df.kwh[[1,4,10]] = 3        #will not work
df['kwh'][[1,4,10]] = 3     #this is how you replace stuff
df.kwh[[1,4,10]]
df