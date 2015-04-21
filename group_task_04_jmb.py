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

main_dir = "C:/Users/J/Desktop/data/"
root = main_dir + "grouptask4/"

## look in data, grouptask4 