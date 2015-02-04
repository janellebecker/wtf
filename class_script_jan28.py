from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os

main_dir = "C:/Users/J/Desktop/data590" 
git_dir = "C:/Users/J/Desktop/DUKE1/590bigdata/GitHub/JMB"
csvfile = "sample_data_clean.csv"
badcsv = "/sample_data_clean.csv"

# OS MODULE---------------------------------------------------

df = pd.read_csv(os.path.join(main_dir, csvfile))
df = pd.read_csv(os.path.join(main_dir, badcsv))

#PYTHON DATA TYPES----------------------------------------

##strings - surrounded in quotes

str1 = "hello, computer" 
str2 = 'hello, human' 
str3 = u'eep'
##unicode string. universal text all cpu understand. readible across platforms.

type(str1) #type str
type(str2)
type(str3) #type unicode

## numeric types
int1 = 10 #integer
float1 = 20.56 #decimal
long1 = 99999999999999999999999 #long number

## logical
bool1 = True
notbool1 = 0
bool2 = bool(notbool1) #exchange bool to integers 

bool2

##CREATING LISTS AND TUPLES-----------------------------------------
## in brief, lists can be changed, tuples CANNOT. 
## we will almost exclusively use lists
list1 = [] #empty list
list1

list2 = [7, 8, 'a']
list2[2] #first element is 0. if i wanted second value, its 2 so thats why answer is 3
list2[2] = 5
list2[2]

##TUPLES ---------------------------
tup1 = (8, 3, 19)
tup1[2] #outputs is 19
tup1[2] = 5 #CANNOT CHANGE ONCE YOUVE RUN THEM 

## CONVERTING
list2 = list(tup1)
tup2 = tuple(list1)
list2

##LISTS CAN BE APPENDED AND EXTENDED
## russian dolls of lists
list2 = [8,3,90]
list2.append([3,90])
len(list2) #will be 4

##i want to add terms, only one list. not a list in a list. 
list2 = [8,3,90]
list2.extend([6, 88])
list2

##LENGTH FUNCTION - 
list2
len(list2)

##CONVERTING LISTS TO SERIES AND DATAFRAME
##create a list of consecutive integers
list4 = range(5,10) #range(n, m) - gives a list from n to m-1

list5 = range(5) #a list starting from 0, consecutive integers

##CREATE SERIES FROM A LIST

list6 = ['q', 'r', 's', 't', 'u']
s1 = Series(list4)
s2 = Series(list6)

s1
s2

##CREATE DataFrame FROM LISTS OR SERIES
##zip function makes pairs or tuples
zip(list4, list6)

list7 = range(60, 65)
zip(list6, list7)
zip2 = zip(list4, list6, list7)
list7[2] #output 62

df1=DataFrame(zip2)

df1

df1[1]

df2 = DataFrame(zip2, columns = ['two', 'apple', ':)'])
df2 = DataFrame(zip2, columns = ['2', 'apple', ':)'])
df2

df2['apple']
df2['2']

##how to display the third and fourth row
df2[3:5]

#third and fourth of the smiley face column
df2[[':)']] [3:5] 


##MAKE A DATAFRAME USING DICT NOTATION
##jenny describes a dictionary like opening a door to see whats behind the door.

#have to reference lists of the same length

df4 = DataFrame({':(' : list4, 9 : list6})
dict1 = {':(' : list4, 9 : list6}
dict1[':(']

dictjenny = {'a' : list7}

dictjenny['a'][3] #i made a dictionary, made a list behind door a. asked for the third thing behind door a.


