 from __future__ import division
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import csv
import xlrd
import time
import matplotlib.pyplot as plt
import pytz
import statsmodels.api as sm #package developed to do stats analyses
from datetime import datetime, timedelta
from dateutil import parser

# SET UP
print(time.ctime())
main_dir = "C:/Users/J/Desktop/data/"
root = main_dir + "logit/"
assign = "allocation_subsamp.csv"
consump = "kwh_redux_pretrial.csv"
df_assign = pd.read_csv(root + assign, header=0)
#-------------------------------------------------------------------------------
## 1. Create a vector with the IDs for the control group, and the IDs for each treatment group
df_E = df_assign.ID[(df_assign.tariff == 'E') & (df_assign.stimulus == 'E')]
df_A1 = df_assign.ID[(df_assign.tariff == 'A') & (df_assign.stimulus == '1')]
df_A3 = df_assign.ID[(df_assign.tariff == 'A') & (df_assign.stimulus == '3')]
df_B1 = df_assign.ID[(df_assign.tariff == 'B') & (df_assign.stimulus == '1')]
df_B3 = df_assign.ID[(df_assign.tariff == 'B') & (df_assign.stimulus == '3')]

#-----------------------------------------------------------------------------
## 2. Seed the random number generator
np.random.seed(1789)

#-----------------------------------------------------------------------------
## 3. Randomly draw from each vector, drawing without replacement
df_E_300 = np.random.choice(df_E, size=300, replace=False, p=None)
df_A1_150 = np.random.choice(df_A1, size=150, replace=False, p=None)
df_A3_150 = np.random.choice(df_A3, size=150, replace=False, p=None)
df_B1_50 = np.random.choice(df_B1, size=50, replace=False, p=None)
df_B3_50 = np.random.choice(df_B3, size=50, replace=False, p=None)

#-----------------------------------------------------------------------------
## 4. Create a DataFrame with all the sampled IDs
    #Combine these series to form a dataframe
    #First, make these arrays into lists. then combine lists to df
df_E_300 = np.random.choice(df_E, size=300, replace=False, p=None).tolist()
df_A1_150 = np.random.choice(df_A1, size=150, replace=False, p=None).tolist()
df_A3_150 = np.random.choice(df_A3, size=150, replace=False, p=None).tolist()
df_B1_50 = np.random.choice(df_B1, size=50, replace=False, p=None).tolist()
df_B3_50 = np.random.choice(df_B3, size=50, replace=False, p=None).tolist()

df_SampID = pd.DataFrame(df_E_300 + df_A1_150 + df_A3_150 + df_B1_50 + df_B3_50, columns=['ID'])

#-----------------------------------------------------------------------------
## 5. Import consumption data and merge with sampled ID's - Strips away all other consumption data
df_cons = pd.read_csv(root + consump, header=0, parse_dates=[2], date_parser=np.datetime64)

#-----------------------------------------------------------------------------
## 6. Merge sampled ID's dataframe with consumption data
df = pd.merge(df_SampID, df_cons, on = ['ID'], how = 'inner')
df.reset_index(drop=True, inplace=True) #drop=true will get rid of old index. inplace will replace old dataframe

#-----------------------------------------------------------------------------
## 7 Compute aggregate monthly consumption for each panel ID.
    #create month variable 
df['month'] = df['date'].apply(lambda x: x.month)
df['year'] = df['date'].apply(lambda x: x.year)

    #aggregate consumption for each ID by month 
grp = df.groupby(['year', 'month', 'ID'])
df_agg = grp['kwh'].sum().reset_index()

#dont need this, since we're going to pivot and create month specific column names
#df_agg['kwh_month'] = df_agg['kwh']
#df_agg.drop('kwh', axis=1, inplace=True)

#-----------------------------------------------------------------------------
## 8. Pivot the data
    # prep to pivot: make kwh_[month] variables for wide dataframe
df_agg['month_str'] = ['0' + str(v) if v < 10 else str(v) for v in df_agg['month']]
df_agg['kwh_ym'] = 'kwh_' + df_agg.year.apply(str) + "_" + df_agg.month_str.apply(str)

    # pivot the data from long to wide df.pivot(i, j, value)
df_piv = df_agg.pivot('ID', 'kwh_ym', 'kwh')
df_piv.reset_index(inplace=True) #to make the ID not the index 
df_piv.columns.name = None #get reid of kwh_ym over the index 

#-----------------------------------------------------------------------------
## 9. Merge the wide dataset with the treatment assignment 
df2 = pd.merge(df_assign, df_piv, on = ['ID'])
df2.head() # df2 is now my dataframe with monthly aggregate info, wide, with T/C info. 

#-----------------------------------------------------------------------------
## 10. Run a logit model to see if consumption could predict T/C status for each of the 4 T groups 
    # get list of all the kwh cols    
kwh_cols = [v for v in df2.columns.values if v.startswith('kwh')]

    # make cols of 1/0's for each type of treatment (for y variables)
    # make each its own dataframe in order to run a separate logit regression
df2['treatment'] = df2['tariff'] + df2['stimulus']
df3 = pd.get_dummies(df2, columns=['treatment'])

df_EEA1 = df3[['ID'] + ['treatment_A1'] + [v for v in df3.columns.values if v.startswith('kwh')]]
df_EEA3 = df3[['ID'] + ['treatment_A3'] + [v for v in df3.columns.values if v.startswith('kwh')]]
df_EEB1 = df3[['ID'] + ['treatment_B1'] + [v for v in df3.columns.values if v.startswith('kwh')]]
df_EEB3 = df3[['ID'] + ['treatment_B3'] + [v for v in df3.columns.values if v.startswith('kwh')]]

    # Let's run some models!
#-----------------------------------------------------------------------------
## Set up: Logit on A1, EE
y_A1 = df_EEA1['treatment_A1']
X_A1 = df_EEA1[kwh_cols]
X_A1 = sm.add_constant(X_A1)

## Run logit model on A1, EE
logit_model_A1 = sm.Logit(y_A1, X_A1)
logit_results_A1 = logit_model_A1.fit()
print(logit_results_A1.summary())
#-------------------------------------------------------
## Set up: Logit on B1, EE
y_B1 = df_EEB1['treatment_B1']
X_B1 = df_EEB1[kwh_cols]
X_B1 = sm.add_constant(X_B1)

## Run logit model on A1, EE
logit_model_B1 = sm.Logit(y_B1, X_B1)
logit_results_B1 = logit_model_B1.fit()
print(logit_results_B1.summary())
#-------------------------------------------------------
## Set up: Logit on B1, EE
y_B1 = df_EEB1['treatment_B1']
X_B1 = df_EEB1[kwh_cols]
X_B1 = sm.add_constant(X_B1)

## Run logit model on A1, EE
logit_model_B1 = sm.Logit(y_B1, X_B1)
logit_results_B1 = logit_model_B1.fit()
print(logit_results_B1.summary())
#-------------------------------------------------------
## Set up: Logit on B3, EE
y_B3 = df_EEB3['treatment_B3']
X_B3 = df_EEB3[kwh_cols]
X_B3 = sm.add_constant(X_B3)

## Run logit model on A1, EE
logit_model_B3 = sm.Logit(y_B3, X_B3)
logit_results_B3 = logit_model_B3.fit()
print(logit_results_B3.summary())
#-------------------------------------------------------

print("Yay! Done with Part I!")

##-----------------------------------------------------------------------------
##-----------------------------------------------------------------------------

## SECTION II

#Dan's code--------------------------------------------------------------------

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# DEFINE FUNCTIONS -----------------
def ques_recode(srvy):

    DF = srvy.copy()
    import re
    q = re.compile('Question ([0-9]+):.*')
    cols = [unicode(v, errors ='ignore') for v in DF.columns.values]
    mtch = []
    for v in cols:
        mtch.extend(q.findall(v))

    df_qs = Series(mtch, name = 'q').reset_index() # get the index as a variable. basically a column index
    n = df_qs.groupby(['q'])['q'].count() # find counts of variable types
    n = n.reset_index(name = 'n') # reset the index, name counts 'n'
    df_qs = pd.merge(df_qs, n) # merge the counts to df_qs
    df_qs['index'] = df_qs['index'] + 1 # shift index forward 1 to line up with DF columns (we ommited 'ID')
    df_qs['subq'] = df_qs.groupby(['q'])['q'].cumcount() + 1
    df_qs['subq'] = df_qs['subq'].apply(str)
    df_qs.ix[df_qs.n == 1, ['subq']] = '' # make empty string
    df_qs['Ques'] = df_qs['q']
    df_qs.ix[df_qs.n != 1, ['Ques']] = df_qs['Ques'] + '.' + df_qs['subq']

    DF.columns = ['ID'] + df_qs.Ques.values.tolist()

    return df_qs, DF

def ques_list(srvy):

    df_qs, DF = ques_recode(srvy)
    Qs = DataFrame(zip(DF.columns, srvy.columns), columns = [ "recoded", "desc"])[1:]
    return Qs

# df = dataframe of survey, sel = list of question numbers you want to extract free of DVT
def dvt(srvy, sel):

    """Function to select questions then remove extra dummy column (avoids dummy variable trap DVT)"""

    df_qs, DF = ques_recode(srvy)

    sel = [str(v) for v in sel]
    nms = DF.columns

    # extract selected columns
    indx = []
    for v in sel:
         l = df_qs.ix[df_qs['Ques'] == v, ['index']].values.tolist()
         if(len(l) == 0):
            print (bcolors.FAIL + bcolors.UNDERLINE +
            "\n\nERROR: Question %s not found. Please check CER documentation"
            " and choose a different question.\n" + bcolors.ENDC) % v
         indx =  indx + [i for sublist in l for i in sublist]

    # Exclude NAs Rows
    DF = DF.dropna(axis=0, how='any', subset=[nms[indx]])

    # get IDs
    dum = DF[['ID']]
    # get dummy matrix
    for i in indx:
        # drop the first dummy to avoid dvt
        temp = pd.get_dummies(DF[nms[i]], columns = [i], prefix = 'D_' + nms[i]).iloc[:, 1:]
        dum = pd.concat([dum, temp], axis = 1)
        # print dum

        # test for multicollineary

    return dum

def rm_perf_sep(y, X):

    dep = y.copy()
    indep = X.copy()
    yx = pd.concat([dep, indep], axis = 1)
    grp = yx.groupby(dep)

    nm_y = dep.name
    nm_dum = np.array([v for v in indep.columns if v.startswith('D_')])

    DFs = [yx.ix[v,:] for k, v in grp.groups.iteritems()]
    perf_sep0 = np.ndarray((2, indep[nm_dum].shape[1]),
        buffer = np.array([np.linalg.norm(DF[nm_y].values.astype(bool) - v.values) for DF in DFs for k, v in DF[nm_dum].iteritems()]))
    perf_sep1 = np.ndarray((2, indep[nm_dum].shape[1]),
        buffer = np.array([np.linalg.norm(~DF[nm_y].values.astype(bool) - v.values) for DF in DFs for k, v in DF[nm_dum].iteritems()]))

    check = np.vstack([perf_sep0, perf_sep1])==0.
    indx = np.where(check)[1] if np.any(check) else np.array([])

    if indx.size > 0:
        keep = np.all(np.array([indep.columns.values != i for i in nm_dum[indx]]), axis=0)
        nms = [i.encode('utf-8') for i in nm_dum[indx]]
        print (bcolors.FAIL + bcolors.UNDERLINE +
        "\nPerfect Separation produced by %s. Removed.\n" + bcolors.ENDC) % nms

        # return matrix with perfect predictor colums removed and obs where true
        indep1 = indep[np.all(indep[nm_dum[indx]]!=1, axis=1)].ix[:, keep]
        dep1 = dep[np.all(indep[nm_dum[indx]]!=1, axis=1)]
        return dep1, indep1
    else:
        return dep, indep


def rm_vif(X):

    import statsmodels.stats.outliers_influence as smso
    loop=True
    indep = X.copy()
    # print indep.shape
    while loop:
        vifs = np.array([smso.variance_inflation_factor(indep.values, i) for i in xrange(indep.shape[1])])
        max_vif = vifs[1:].max()
        # print max_vif, vifs.mean()
        if max_vif > 30 and vifs.mean() > 10:
            where_vif = vifs[1:].argmax() + 1
            keep = np.arange(indep.shape[1]) != where_vif
            nms = indep.columns.values[where_vif].encode('utf-8') # only ever length 1, so convert unicode
            print (bcolors.FAIL + bcolors.UNDERLINE +
            "\n%s removed due to multicollinearity.\n" + bcolors.ENDC) % nms
            indep = indep.ix[:, keep]
        else:
            loop=False
    # print indep.shape

    return indep


def do_logit(df, tar, stim, D = None):

    DF = df.copy()
    if D is not None:
        DF = pd.merge(DF, D, on = 'ID')
        kwh_cols = [v for v in DF.columns.values if v.startswith('kwh')]
        dum_cols = [v for v in D.columns.values if v.startswith('D_')]
        cols = kwh_cols + dum_cols
    else:
        kwh_cols = [v for v in DF.columns.values if v.startswith('kwh')]
        cols = kwh_cols

    # DF.to_csv("/Users/dnoriega/Desktop/" + "test.csv", index = False)
    # set up y and X
    indx = (DF.tariff == 'E') | ((DF.tariff == tar) & (DF.stimulus == stim))
    df1 = DF.ix[indx, :].copy() # `:` denotes ALL columns; use copy to create a NEW frame
    df1['T'] = 0 + (df1['tariff'] != 'E') # stays zero unless NOT of part of control
    # print df1

    y = df1['T']
    X = df1[cols] # extend list of kwh names
    X = sm.add_constant(X)

    msg = ("\n\n\n\n\n-----------------------------------------------------------------\n"
    "LOGIT where Treatment is Tariff = %s, Stimulus = %s"
    "\n-----------------------------------------------------------------\n") % (tar, stim)
    print msg

    print (bcolors.FAIL +
        "\n\n-----------------------------------------------------------------" + bcolors.ENDC)

    y, X = rm_perf_sep(y, X) # remove perfect predictors
    X = rm_vif(X) # remove multicollinear vars

    print (bcolors.FAIL +
        "-----------------------------------------------------------------\n\n\n" + bcolors.ENDC)

    ## RUN LOGIT
    logit_model = sm.Logit(y, X) # linearly prob model
    logit_results = logit_model.fit(maxiter=10000, method='newton') # get the fitted values
    print logit_results.summary() # print pretty results (no results given lack of obs)


#####################################################################
#                           SECTION 2                               #
#####################################################################

main_dir = "C:/Users/J/Desktop/data/"
root = main_dir + "logit/"

nas = ['', ' ', 'NA'] # set NA values so that we dont end up with numbers and text
srvy = pd.read_csv(root + 'Smart meters Residential pre-trial survey data.csv', na_values = nas)
df4 = pd.read_csv(root + 'data_section2.csv')

# list of questions
qs = ques_list(srvy) 

# get dummies
sel = [200, 310, 405]
dummies = dvt(srvy, sel)

# run logit, optional dummies
tariffs = [v for v in pd.unique(df4['tariff']) if v != 'E']
stimuli = [v for v in pd.unique(df4['stimulus']) if v != 'E']
tariffs.sort() # make sure the order correct with .sort()
stimuli.sort()

for i in tariffs:
    for j in stimuli:
        do_logit(df4, i, j, D = dummies)



