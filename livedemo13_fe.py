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
from dateutil import parser #use this to ensure dates are parsed correctly 

## copy and past from class_script_mar18_logit (#12) 

main_dir = "C:/Users/J/Desktop/data/"
root = main_dir + "data08/"
paths = [root + v for v in os.listdir(root) if v.startswith("08_")]

## logit will be a tool to check for balance - we don't want your demographics to predict your T/C status 

# IMPORT AND ADD/DROP VARIABLES -----------------------
df = pd.read_csv(paths[1], header=0, parse_dates=[1], date_parser=np.datetime64)  #this date parser is much faster than before.
df.head()
df_assign = pd.read_csv(paths[0], header=0)

## renaming it to match Alcott's paper
df_assign.rename(columns={'assignment':'T'}, inplace=True)


"""NOte: using notation from Alcott 2010"""

# ADD/DROP VARIBALES -----------------------------------------------
ym = pd.DatetimeIndex(df['date']).to_period('M') # m =month, d=day; converting it to time period by months
df['ym'] = ym.values
##df[ym] is string of integer values like 540. these values have particular internal value when you 
##convert them back into a date.  convert them using "astype" 
#df['ym'].values.astype('datetime64[M]')

# MONTHLY AGGREGATION -----------------------------------------------
grp = df.groupby(['ym', 'panid'])
df = grp['kwh'].sum().reset_index()

# MERGE STATIC VARIABLES --------------------------------------------
df = pd.merge(df, df_assign)
df.reset_index(drop=True, inplace=True)

df.head()

# FE MODEL (DEMEANING) FIXED EFFECTS 

"""demean function"""

def demean(df, cols, panid):
    """
    inputs: df (pandas dataframe), cols (list of str of column names from df),
                    panid (str of panel ids)
    output: dataframe with values in df[cols] demeaned
    """

    from pandas import DataFrame
    import pandas as pd
    import numpy as np

    cols = [cols] if not isinstance(cols, list) else cols
    panid = [panid] if not isinstance(panid, list) else panid
    avg = df[panid + cols].groupby(panid).aggregate(np.mean).reset_index()
    cols_dm = [v + '_dm' for v in cols]
    avg.columns = panid + cols_dm
    df_dm = pd.merge(df[panid + cols], avg)
    df_dm = DataFrame(df[cols].values - df_dm[cols_dm].values, columns=cols)
    return df_dm

## set up variables for demeaning------------------------------------------
df['log_kwh'] = df['kwh'].apply(np.log)
#he told us 540, 541 were pre treatment, 542 onward were treatment periods 
df['P'] = 0 + (df['ym'] > 541)
df['TP'] = df['T']*df['P']

#avoid dummy variable trap. then, avoid perfect multicollinearity with ym.  remove first and last columns 

# DEMEAN VARIABLES ---------------------------------------------
cols = ['log_kwh', 'TP', 'P']
panid = 'panid'
df_dm = demean(df, cols, 'panid')

df_dm.head()

## set up regression variables -----------------------------
y = df_dm['log_kwh']
X = df_dm[['TP', 'P']]
X = sm.add_constant(X)
mu = pd.get_dummies(df['ym'], prefix = 'ym').iloc[:, 1:-1] 

## run model ------------------------
fe_model = sm.OLS(y, pd.concat([X, mu], axis=1))
fe_results = fe_model.fit()
print(fe_results.summary())















