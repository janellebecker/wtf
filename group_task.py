from __future__ import division 
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import time
import csv
import xlrd

print(time.ctime())
main_dir = "C:/Users/J/Desktop/data"
git_dir = "C:/Users/J/Desktop/DUKE1/bigdata590/GitHub"
root = main_dir + "/group_task_04"
assignfile = "SME and Residential allocations.xlsx"


df = pd.read_csv(os.path.join(root + "/" + "File1.txt"))
df.head()

# What type of file is File1? How is it delimited?
N = 10
with open(root + "/" + "File1.txt") as myfile:
    head = [next(myfile) for x in xrange(N)]
print head

# Bring in the data. 
pathlist = [root + v for v in os.listdir(root) if v.startswith("/File")]
list_of_dfs = [pd.read_table(v, names = ['panid', 'time', 'kwh'], sep = " ", header = None, 
na_values = ["-", "NA"]) for v in pathlist]

# Get rid of duplicates in each df

for i in list_of_dfs:
    i.drop_duplicates(['panid', 'time'], take_last = True)
    i.dropna(axis = 0, how = 'any')









