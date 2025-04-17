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
#VITALS Data: 2806 screening cases vs. 2722 baseline cases (many missing height data - coded as -4)

#Identifier variable: Across all datasets
#RID (Participant roster ID)

#Screening clinical exam date variable: PHYSICAL, NEUROEXM, VITALS
#EXAMDATE (Clinical Screening Examination Date)

#Clinical variables: PHYSICAL
#PXGENAPP (1. General Appearance; 1=Normal, 2=Abnormal)
#PXHEADEY (2. Head, Eyes, Ears, Nose and Throat; 1=Normal, 2=Abnormal)
#PXNECK (3. Neck; 1=Normal, 2=Abnormal)
#PXCHEST (4. Chest; 1=Normal, 2=Abnormal)
#PXHEART (5. Heart; 1=Normal, 2=Abnormal)
#PXABDOM (6. Abdomen; 1=Normal, 2=Abnormal)
#PXEXTREM (7. Extremities; 1=Normal, 2=Abnormal)
#PXEDEMA (8. Edema; 1=Normal, 2=Abnormal)
#PXPERIPH (9. Peripheral Vascular; 1=Normal, 2=Abnormal)
#PXSKIN (10. Skin and Appendages; 1=Normal, 2=Abnormal)
#PXMUSCUL (11. Musculoskeletal; 1=Normal, 2=Abnormal)
#PXBACK (12. Back; 1=Normal, 2=Abnormal)
#PXOTHER (13. Other; 1=Normal, 2=Abnormal)

#Clinical variables: NEUROEXM
#NEUROEXM (1. General Appearance; 1=Normal, 2=Abnormal)






#Covariates
#Age at screening visit


























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
