from pandas import Series, DataFrame
import pandas as pd
import numpy as np

# IMPORTING DATA-------------------------------------------

##assigning file paths 
main_dir = "C:\Users\J\Desktop\DUKE1\590bigdata\GitHub\590_JMB"
csvfile = "\sample_data_clean.csv"
textfile = "\sample_data_clean.txt"

main_dir + csvfile ##full system path for this file 

##tab completion  - type m, tab, and it'll show you  stuff. only one? itll take it

##import the two files

#read_csv and read_table  read_table is more general. read_table defaults to tab delimiter
