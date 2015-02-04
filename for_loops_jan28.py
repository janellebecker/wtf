from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os

main_dir = "C:/Users/J/Desktop/data590" 
git_dir = "C:/Users/J/Desktop/DUKE1/590bigdata/GitHub/JMB"
csvfile = "sample_data_clean.csv"

# FOR LOOPS---------------------------------------------------------------
df = pd.read_csv(os.path.join(main_dir, csvfile))

list1 = range(10, 15)
list2 = ['a', 'b', 'c']
list3 = [1, 'a', True]

##ITERATING OVER ELEMENTS (FOR LOOPS)
for v in list1:
    v

## want to see what's happening? use the print function
for v in list2:
    print(v)
    
     
for v in list3: 
    print(v, type(v)) # it will tell me the type of the object that it spits out. 1 is an int, a is a string, true is a bool.

for v in list3: 
    print(v, type(v), 'hahaha')
    
##  to square everything in a list
list4 = range(11)
list4


for v in list4:
    print(v**3) # cannot use ^, instead use **
    
## to continually add
jenny = 0
for v in list4:
    jenny += v
    print(jenny)
    
## POPULATING LISTS
list5 = [] #empty list


# i dont know if this worked???
for v in list1:
    v2 = v**2
    list5.extend([v2]) #extend, it has to be a single object. put it in brackets. 
    
list6 = []
for v in list1:
    list6.append(v2) #appends whatever object as is


#whatever comes out of this for loop is going to make it a list
[v**2 for v in list1]

[v+2 for v in list1]

list7 = [v < 12 for v in list1]
list7


## ITERATING USING ENUMERATE

## i can do a for loop and pair that value with its index
## enumerate only starts with index 0. 

list8 = [ [i,v/2] for i, v in enumerate(list1)]
list8

list9 = [ [i,float(v)/2] for i, v in enumerate(list1)]
list9


## ITERATE THROUGH A SERIES ------------------------------------------------
s1 = df['consump']
[v > 2 for v in s1]
[[i,v] for i, v in s1.iteritems()] #iteritems will give you the assigned index???


## ITERATE THROUGH A DATAFRAME-------------------------------------------------
[v for v in df] #
[df[v] for v in df] #?
[i, v] for i, v in df.iteritems()] #?

