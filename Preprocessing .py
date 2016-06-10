
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import datetime as dtc

def remove_missing_data(df):
    df=df.convert_objects(convert_numeric=True)
    df=df.fillna(method="backfill")

