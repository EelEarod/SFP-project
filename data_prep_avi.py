#!/usr/bin/env python
# coding: utf-8

# ## Data pre_processing script

# In[1]:

##import libraries (basic packages)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels 

##import libraries (ML packages)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# In[2]:

##Importing dataset. Will need to discuss with Sheena how to link this with Apocrita environment
data = 'C:/datasets/adni_data.csv' #to change
df = pd.read_csv(data)

# In[3]:

#We will need code here for extracting the data from our clinical variables from the screening visit (VISCODE2='sc'). 

#Justification

#PHYSICAL Data: 3746 screening cases vs. 289 baseline cases (some of these missing additional data)
#NEUROEXM Data: 3748 screening cases vs. 289 baseline cases (some of these missing additional data)
#VITALS Data: 2806 screening cases vs. 2722 baseline cases (many missing height data)























# In[3]:

##Preview dataset
df.head()        # First 5 rows
df.tail()        # Last 5 rows
df.sample(5)     # Random 5 rows

# In[4]:

##find missing data

df.isnull().sum()        # Total missing per column
df.isnull().mean()*100   # % of missing per column
df[df.isnull().any(axis=1)]  # Rows with missing values
