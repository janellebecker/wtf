#leaving things as nuimbers/floats is better. numpy goes faster than converting it to a string and searching for values. 
# he'll post his "solutions"  to the group task online
#-----------------------------
from __future__ import division
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os

main_dir = "C:/Users/J/Desktop/data"
root = main_dir + "/" "demo0607" + "/"

#PATHING ----------------------------------
paths = [os.path.join(root, v) for v in os.listdir(root) if v.startswith("file_")]

#IMPORT AND STACK ALL IN ONE LINE-------------------------------------
df = pd.concat([pd.read_csv(v, names = ['panid', 'date','kwh']) for v in paths], 
    ignore_index = True)
    
df_assign = pd.read_csv(main_dir + "/" "sample_assignments.csv", usecols = [0,1])    


#MERGING---------------------------------------------------
df = pd.merge(df, df_assign)

#GROUPBY aka split apply and combine #like collapsing in stata = split and apply
## see more at http://pandas.pydata.org/pandas-docs/stable/groupby.html


# split by control/treatment, pooled w/o time
groups1 = df.groupby(['assignments'])  #splitting by assignment
groups1.groups

# apply the mean
groups1['kwh'].apply(np.mean)  #.apply is to any type of function
groups1['kwh'].mean()   #if its internal function, itll go faster to use that the call from numpy. 

%timeit -n 100 groups1['kwh'].apply(np.mean) # %timeit -n 100 is a magic function for ipython? 
%timeit -n 100 groups1['kwh'].mean()        #this is going to tell us how long it'd take to do 100 loops. itll tell me for 3 tries, the best of 3. 

#SPLIT BY THE CONTROL AND TREATMENT, POOLING WITH TIME--this time we care about the time. don't care who, but we do care about the time

groups2 = df.groupby(['assignments', 'date'])  #splitting by assignment
groups2.groups

# apply the mean
groups2['kwh'].apply(np.mean)  #.apply is to any type of function
groups2['kwh'].mean()   #if its internal function, itll go faster to use that the call from numpy. 

groups3 = df.groupby([ 'date', 'assignments']) #ORDER MATTERS FOR HOW TO COLLAPSE DATA
groups3.groups

##UNSTACK UNSTACKING DATA ----------------------------------------------------
gp_mean = groups2['kwh'].mean() 
type(gp_mean)
gp_unstack = gp_mean.unstack('assignments')  #reshapes the data. C/T was running down the columns. now they're the column titles

gp_unstack['T'] #mean, over time, of all treated panids

