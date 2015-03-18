from pandas import Series, Dataframe
import pandas as pd
import numpy as np

#DUPLICATED VALUES---------------------------------------------------

##create a new dataframe

zip3 = zip(['red', 'green', 'blue', 'orange']*3, [5, 10, 20, 40]*3, 
    [':(', ':D', ':D']*4)

df3 = DataFrame(zip3, columns = ['a', 'b', 'c'])
df3

## pandas method duplicated -->will yield boolean true/false answers 
df3.duplicated() #searching from top to bottom by row by default for duplicates
df3.duplicated(take_last = True) #searches bottom to top 

##subsets for duplicates. don't look at the entire dataframe, just look [here]
df3.duplicated(subset = ['a', 'b']) #use keys  a and b only
df3.duplicated(['a', 'b']) # same thing as above. setting criteria for looking for dups

## i wouldn't want to see the exact same data and time occur for the same person

## How to get all values that have duplicates? (purging)
## how to tag ALL 3 REPEATED ROWS and get rid of them  

##purging

t_b = df3.duplicated()
b_t = df.duplicated(take_last = True)

#anything that is dup'd top to bottom or bottom to top will return true
# so true/false will yield false. only true/true yields true. 

# ! works for not equal to, but must use ~ tilde for "not" 

unique = ~(t_b | b_t) #give me NOT where either is true (or)
unique = ~t_b & ~b_t # give me not A and not B  #same shit

df3[unique]



# DROPPING DUPLICATES -------------------------------------------------
df3.drop_duplicates()  #will keep the first instance
df3.drop_duplicates(take_last = True) #starts from the bottom, drops duplicates

##This is the same as...
t_b = df3.duplicated()
df3[~t_b]

df3.drop_duplicates() ==df3[~t_b] #output: all true ==> there are all equivalent

##subset criteria for what we're dropping over

df3.drop_duplicates(['a', 'b']) #would have to drop all but first, since the colors repeated after the first four

#WHEN TO USE THIS STUFF------------------------------------

##if you want to keep the first (top to bottom) values of duplicated values and remove others
df3.drop_duplicates()

##if you want to keep the first duplicate and drop all else (from bottom to top)
df3.drop_duplicates(take_last = True)

## purge all values that are duplicates --> use trick

t_b = df3.duplicated()
b_t = df.duplicated(take_last = True)
unique = ~(t_b | b_t) #complement where either is true
df3[unique]

