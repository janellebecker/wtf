from pandas import Series, DataFrame
import pandas as pd
import numpy as np

# IMPORTING DATA-------------------------------------------

##assigning file paths 
main_dir = "C:\Users\J\Desktop\data"
csvfile = "\sample_data_clean.csv"
textfile = "\sample_data_clean.txt"

main_dir + csvfile ##full system path for this file 

##tab completion  - type m, tab, and it'll show you  stuff. only one? itll take it

##import the two files

#read_csv and read_table  read_table is more general. read_table defaults to tab delimiter
pd.read_csv(main_dir + csvfile)

df = pd.read_csv(main_dir + csvfile)
df
df2 = pd.read_table(main_dir + textfile)
df2

##how to test a type of an object
type(df)

##EXPLORING YOUR DATAFRAME----------------------------------------------------

## find the names of your dataframe
list(df)

## extracting columns of data (aka series)

df.consump ##extracts data as a series

#dictionary-like key: column names can be thought of as a key
df['consump']

c =df.consump  ##key? attribute?
c2 = df['consump']

type(c)

## BOOLEAN (LOGICAL) OPERATORS-------------------------------------------------

## compare c to c2 ##output: true for everything

c==c2
c > c2
c >= c2

##other boolean operators. same as stata. !=

# ROW EXTRACTION--------------------------------------------------

##row slicing from dataframe
df[5:10]  #to look at rows. [m, n] yields m to n-1 because start at zero. always go one beyond what you want.
df[:10] shorthand to go up to a number 

df[:10] == df[0:10]

## cannot put single digit to extract a single row. df[10]
##if i want row 10, 
df[10:11]

## row slicing from series (aka a column)

c[5:10]
df.panid[0:10]  ## you can use the name directly if you didn't name it. must use df.---

## extraction by boolean indexing
# if its a 0, dont want it. if it's a 1, i want it.
## say i only want to look at participant #4 in the data 
df

df.panid == 4
df[df.panid == 4] #extracts subset of df where panid ==4

df[df.consump > 2] ## where is a column variable greater than 2 for anybody? 

##indexing vector has same length as dataframe. this is important

df.panid[df.panid > 2]