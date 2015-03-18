from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os

from __future__ import division #from python3, letting us divide integers

main_dir = "C:\Users\J\Desktop\data"
# git_dir - will use later 
csv1 = "small_data_w_missing_duplicated.csv"
csv2 = "sample_assignments.csv" 

#IMPORT DATA ----------------------------------------------------------
df1 = pd.read_csv(os.path.join(main_dir, csv1), na_values = ['-', 'NA'])
df2 = pd.read_csv(os.path.join(main_dir, csv2), na_values = ['-', 'NA'])


#CLEAN DATA ------------------------------------------------------------

##clean df1  of duplicates across all columns (see demo 03)
df1 = df1.drop_duplicates()
df1 = df1.drop_duplicates(['panid', 'date'], take_last = True)

##clean df2  - from an excel, the right 3 columns were just describing the data. 
##we dont want those
df2[[0,1]]

## df1 doesn't tell us who is treatment or control.
## we need to assign the correct T/C code, tags to the consumption data

df2 = df2[[0,1]] #reassigning df2 to a subset 

# COPY DATAFRAMES--------------------------------------------------------
df3 = df2         #creates a link/reference (alter df2 DOES affect df3)
df4 = df2.copy()  #creating a copy so that altering df2 does not affet df4

# REPLACING DATA --------------------------------------------------------
df2.group.replace(['T', 'C'], ['1', '0']) #group was the column name
df2.group = df2.group.replace(['T', 'C'], ['1', '0']) # this is actually changing the dataframe

df3 #should be the same as current df2
df4 #should be the old df2 frozen in time (like saving a copy at that moment in time)

# MERGING ---------------------------------------------------------------
df1 #the data
df2 #the assignments

##default is a "many to one" merge. uses intersection between the two 
##automatically finds the keys it has in common (e.g. panid)
##would fail if there were duplicates in df2 cause it wouldnt know how to assign it
pd.merge(df1, df2) 

##merge based on specific column / key
pd.merge(df1, df2, on = ['panid']) #needs to be a list of the keys you want to merge on 

pd.merge(df1, df2, on = ['panid'], how = 'inner') #default state. intersection (inner join)
## there will be no 5th panid since there was no intersection, it drops that

pd.merge(df1, df2, on = ['panid'], how = 'outer') #union of the two keys
## a way of seeing what was left out after merging - subsets that didnt overlap right 

df5 = pd.merge(df1, df2, on = ['panid'], how = 'inner') #merge creates a new object 

## ROW BINDS AND COLUMN BINDS (combining and stacking)-------------------------------------

##row bind (aka stacking)
pd.concat([df2, df4]) ##the default is to "row bind", meaning it adds on rows, including the indices
pd.concat([df2, df4], axis = 0) ## this is the same becasue axis=0 is the row
## how to reset the indices when you row bind 
pd.concat([df2, df4], axis = 0, ignore_index = True) ## ignore index = false is default


##column bind 
pd.concat([df2, df4], axis = 1) ##attaches it by column with the other one


#For the upcoming assignment: import, clean, stack. import assignments, clean, merge. 
