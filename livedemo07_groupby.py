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

#GROUPBY aka split apply and combine. 
##see more at http://pandas.pydata.org/pandas-docs/stable/groupby.html

##ignore time differences, pool the data together in order to compare all control
##to all treatment data, regardless of time. We'll group by C/T.

grp1 = df.groupby(['assignments'])

#looking at grp1 isn't helpful here when we look at it. but its an object thats efficient
#without having to deal with lists. we can do quick analysis with it based on groups.

#mean of all treatments? of all controls?
grp1.mean()
grp1.groups #  CAUTION! don't do this with super big data. it will crash!
keys = grp1.groups.keys()  #if i wanted to see the groups, once ive said group by [whatever]

gd1 = grp1.groups #let me call this group dictionary1 to be quicker
type(gd1)

# gd1. tab  to look what it can do

##peak inside gd1 (dictionary)  #how have my groups been grouped? (should be C/T)
gd1.keys()       #tells me how its been grouped, C or T
gd1.values()     #no key, just value, in order.
gd1['C']         #gd1 is a dict, so must use keys to get the data. 
gd1.values()[0]  # see the first column i think?
gd1.viewvalues()  # see all possible values 

keys = grp1.groups.keys()  #use this one. #if i wanted to see the groups, once ive said group by [whatever]

#----------------------------beginning of video 2--------------------------

##looking at iteration properties of a dictionary - use for "for loops"
gd1.itervalues() 

[v for v in gd1.itervalues()]   #ends up being the same as gd1.values(), 
[k for k in gd1.iterkeys()]     #ends up being the same as gd1.keys()
[(k, v) for k,v in gd1.iteritems()]    #i'm not clear what this returns. it 
##looks like the gd1 dictionary but it isnt a useful object like a dictionary


##SPLIT AND APPLY , SPLITTING AND APPLYING information (pooled data)
#i want to pool stuff and group based on some assignment, 
grp1.mean()  #like collapsing the data. i'll get the mean 
grp1['kwh'].mean()  #panid average was stupid, we can call up whatever we want

#but we don't pool data over time, yo. 

##split and apply (panel data/time series) 

#this will put all people's 
#readings in the control group from jan 1 together, all the jan 1 T's together, etc. 
#now I can compare the T and C on the day. 
grp2 = df.groupby(['assignments', 'date']) 
grp2.groups.keys()
gd2 = grp2.groups

df[38:39]    #seeing that the Jan 9 output we see from above is what it spit out
df[68:69]    #from looking at gd2

gd2 #look at the dictionary (key, value) pairs
grp2.mean()              #prettier format but includes panid average, which is dumn
grp2['kwh'].mean()

gd2 #shows me Jan 9 Treatment was row index 38 and row index 68. 
    # its not averaging these, it's averaging their kwh!
    #similarly, {('C', '1-Jan'): [0L, 90L, 120L], came out but we can see the mean is 2. 

df['kwh'][[0, 90, 120]]  # will spit out the kwh for those three row in index values
grp2.mean()  #will take values assigned in the groupby object and takes average of their values

#i want to compare jan 1 treatment vs control.
#-------------------------------video 3-------------------------
grp2['kwh'].mean()

##TESTING FOR BALANCE (OVER TIME)
from scipy.stats import ttest_ind
from scipy.special import stdtr

## EXAMPLE WITH T TESTS
a = [1, 4, 9, 2]
b = [1, 7, 8, 9]

t = ttest_ind(a, b, equal_var = False)  #this is saying they dont have equal variance
         #this will return a (t-stat, p-value, two tailed)
t, p = ttest_ind(a, b, equal_var = False)   #this assigns them separately.

# i could also extract the first [0] or second [1] values of that tuple:
t, p = ttest_ind(a, b, equal_var = False)[0] #for the t value
t, p = ttest_ind(a, b, equal_var = False)[1] #for the p-value


#compare treatment and control, by date. 

#set up our data
grp = df.groupby(['assignments', 'date'])
grp.groups # DONT DO THIS WITH LARGE DATA so why does he show us :( 

#get a set of all treatments by date
trt = {k[1]: df.kwh[v].values for k, v in grp.groups.iteritems() if k[0] == 'T'}

#what is that saying?  ^^^^^^^^^
grp.groups.keys()[0] #of the groupedby df, which gives me tuples (T, date): (row, row), give me the first row 
grp.groups.keys()[0] [0] #of that first tuple of the list, give me the first element, ("C")

#grp.groups.iteritems() gives a key (C, 1-jan) and the associated list (0, 90, 120)
#we only want to keep them if the first value has a key "T"
#i want all the T group, but I want it by date. the second element is index 1 (hence k[1])
#i want the values of the df for kwh in the [30, 60] row index values (hence the .values)

#break it down for me
v = [30,60]
df.kwh[v]   #but this is a series. we want a list. dictionary has to be key, list pairs
df.kwh[v].values

grp.groups.keys()[0] #output  ('C', '23-Jan')

k = grp.groups.keys()[0]
k[1]

trt

# now the control group, by date
trt = {k[1]: df.kwh[v].values for k, v in grp.groups.iteritems() if k[0] == 'T'}
ctrl = {k[1]: df.kwh[v].values for k, v in grp.groups.iteritems() if k[0] == 'C'}

keys = trt.keys() # to see the  keys in control group. same as control in this case

diff = {k: (trt)[k].mean() - ctrl[k].mean() for k in keys} #this will be a dict
#give me the date as key. then give me the trt value at that key and subtract the ctrl group's value at that key

diff  #now i have the mean from jan 1 from T and subtracted mean from C on jan 1.


# create dataframes of this information
tstats = DataFrame([k, np.abs(ttest_ind(trt[k], ctrl[k], equal_var = False)[0])] for k in keys], 
    columns = ['date', 'tstat'])
pvals = DataFrame([k, np.abs(ttest_ind(trt[k], ctrl[k], equal_var = False)[1]) for k in keys], 
    columns = ['date', 'tstat'])
    
t_p = pd.merge(tstats, pvals)



## comparison! t-stats  t stats  comparisons  t stat and p values

tstats = {k: float(ttest_ind(trt[k], ctrl[k], equal_var = False)[0]) for k in keys}
pvals = {k: float(ttest_ind(trt[k], ctrl[k], equal_var = False)[1]) for k in keys}
t_p = {k: (tstats[k], pvals[k]) for k in keys}
#above, now i have a dictionary with a pair (t, p values)

#LOOKING FOR BALANCE - we'll want to look before the T was assigned and see that
# the C group looks similar in consumption patterns. 










