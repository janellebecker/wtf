from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os

main_dir = "C:\Users\J\Desktop\data" 
git_dir = "C:\Users\J\Desktop\DUKE1\bigdata590\GitHub" 
csvfile = "sample_missing.csv"

# IMPORTING DATA: Set the value for missing values (SENTINELS)----------------------

df = pd.read_csv(os.path.join(main_dir, csvfile))
df

## easily pull up initial values - df.head
df.head() ## top five values
df.head(10) ## top 10 rows (0-9)
df[:10] # slicing same thing

df.tail(10) ## gives bottom 10 rows
##look at the column/series consump. of that series, give me the first 10 values. 
##then apply a function (type) to those particular values (top ten of consump)

df['consump'].head(10).apply(type)  ##output: the values are considered strings

##We DON'T want string data for missing values. "." is commong sentintel for missing, but
## we need to create new sentinels to adjust for this

##change the sentinel type (missing value type) using na_values to define
## what text is considered a missing value. any column with below, itll be tagged missing.

missing = ['.','NA', 'NULL', ' '] # recall, this is a list  #recall 999999 may also be one. look at notes
df = pd.read_csv(os.path.join(main_dir, csvfile), na_values = missing)

df.head(10) #double check 
df['consump'].head(10).apply(type) 

# MISSING DATA (USING SMALLER DATAFRAME)--------------------------------------

#quick tip: you can REPEAT lists by multiplication!

[1,2,3]
[1,2,3]*3  output will be 1 1 2 3 1 2 3 1 2 3 

# types of missing data
None
np.nan #from numpy dataset
type(None) #missing is a none (object)
type(np.nan) #float (numeric)   # we want to use numpy value/float (not object). 
## faster to use numeric or numpy style data 

##Create a small sample data set! 

zip1 = zip([2, 4, 8], [np.nan, 5, 7], [np.nan, np.nan, 22])

df1 = DataFrame(zip1, columns = ['a', 'b', 'c'])

df1

## Search for missing data using: 
df1.isnull()  #this is the pandas method to find missing data
np.isnan(df1) #numpy way # returns boolean ways. true is missing. 
## how to find out the "verbs"/a list of all the options. that DataFrame contains. Type in DataFrame. then hit tab

## subset of columns to find missing values
cols = ['a', 'c']
df1[cols]
df1[cols].isnull()

## For series
df1['b'].isnull()  #this will actually help us extract data

## find non-missing values
df1.notnull() #now true means it is not a missing value
df1.isnull()

## FILLING IN OR DROPPING MISSING VALUES -----------------------------------

##FILLING
## pandas method fillna

df1.fillna(9999) #if missing, itll fill it with 9999
df1
df2 = df1.fillna(999)
df2

## DROPPING empty/missing values
##pandas method dropna

df1.dropna() # default: this drops the entire row with ANY missing value!!!!
df1.dropna(axis = 0, how = 'any') #default : this is saying drop ROWS with ANY missing values
df1.dropna(axis = 1, how = 'any') # (axis=1 is columns) #drop columns with any missing values 
df1.dropna(axis = 0, how = 'all') #will only drop row if ALL values are missing

df.tail
df.dropna(how = 'all') #there were a lot of empty rows! 

#SEEING ROWS WITH MISSING DATA------------------------------------------------

#understand why things are missing, get a better picture of the data

df3 = df.dropna(how = 'all') #df3 is now the dataframe cleaned of all missing rows
df3.isnull() #can't see the data


df3['consump'].isnull()

rows = df3['consump'].isnull()
df3[rows]

