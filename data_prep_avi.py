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

#Original variables

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
#NXVISUAL (1. Significant Visual Impairment; 1=Absent; 2=Present)
#NXAUDITO (2. Significant Auditory Impairment; 1=Absent; 2=Present)
#NXTREMOR (3. Tremor; 1=Absent; 2=Present)
#NXCONSCI (4. Level of Consciousness; 1=Absent; 2=Present)
#NXNERVE (5. Cranial Nerves; 1=Absent; 2=Present)
#NXMOTOR (6. Motor Strength; 1=Absent; 2=Present)
#NXFINGER (7a. Cerebellar - Finger to Nose; 1=Absent; 2=Present)
#NXHEEL (7b. Cerebellar - Heel to Shin; 1=Absent; 2=Present)
#NXSENSOR (8. Sensory; 1=Absent; 2=Present)
#NXTENDON (9. Deep Tendon Reflexes; 1=Absent; 2=Present)
#NXPLANTA (10. Plantar Reflexes; 1=Absent; 2=Present)
#NXGAIT (11. Gait; 1=Absent; 2=Present)
#NXOTHER (12. Other; 1=Absent; 2=Present)

#Clinical variables: VITALS
#VSWEIGHT (1a. Weight; Numeric) #VSWTUNIT (1b. Weight Units; 1=pounds; 2=kilograms)
#VSHEIGHT (2a. Height; Numeric) #VSHTUNIT (2b. Height Units; 1=inches; 2=centimeters)
#VSBPSYS (3a. Systolic  BP- mmHg; range: 76-250)
#VSBPDIA (3b. Diastolic  BP- mmHg; range: 36-130)
#VSPULSE (4. Seated Pulse Rate (per minute); range: 40-130)
#VSRESP (5. Respirations (per minute); range: 6-40)
#VSTEMP (6a. Temperature; range: 6-40) #VSTMPSRC (6b. Temperature Source; 1=Oral, 2=Tympanic, 3=Other); VSTMPUNT (6c. Temperature Units, 1=Fahrenheit; 2=Celsius)

#Need code to seperately extaract the baseline data from the ADNIMERGE file
#Afterwards merge with the screening clinical data based only on RID 

#Demographic and diagnostic variables at baseline: ADNIMERGE
#EXAMDATE (Baseline examination date)
#SITE (ADNI exam site)
#DX_bl (Baseline diagnosis; AD=Alzheimer's disease; CN=cognitively normal; EMCI=early MCI; LMCI=late MCI; SMC=subjective memory concern)
#AGE (Baseline age)
#PTGENDER (Gender)
#PTEDUCAT (Education)
#PTRACCAT (Race)
#APOE4 (APOE e4 carrier status; 0=non-carrier; 1=heterozygous carrier; 2=homozygous carrier)



#TRANSFORMATIONS ON EXISTING VARIABLES- DORAE TO ATTEMPT AND LAUREN TO ADVISE HERE

#Convert categorical variables (numeric) into factors - this is the majority of our clinical variables and our demographic/diagnostic variables

#Ensure numeric variables are classified as numeric and not another category (e.g. string)

#Ensure missing values are coded properly (for example may be -4 for some of the height values where the data not collected properly)






#NEW VARIABLES TO CALCULATE - DORAE TO ATTEMPT AND LAUREN TO ADVISE HERE

#Rename baseline and screening exam dates (prior to the merge) - for later calculations 

#Age at screening visit; will need to calculate the difference between the screening and baseline visit dates (keep this variable for a potential covariate in sensitivity analysis)
#and subtract this from the baseline age to get your screening age

#Calculate an age squared variable for the screening visit value

#BMI. Will need to standardise height and weight to common united (e.g. kg and cm), use the formula to calculate BMI afterwards. 
VSWEIGHT=float(input())
VSHEIGHT=float(input())
bmi = VSWEIGHT/(VSHEIGHT)**2
if bmi<18.5:
    print("Underweight")
elif bmi>=18.5 and bmi<25:
    print("Normal")
elif bmi>=25 and bmi<30:
    print("Overweight")
else:
    print("Obesity")

#MAP. Use the code to calculate MAP (mean arterial pressure based on systolic and diastolic readings):
VSBPSYS=float(input))
VSBPDIA=float(input))
MAP = (VSBPSYS + 2(VSBPDIA))/3

#Temperature. Need to standardise temperature to either farenheit or celsius. 
#We can choose to create seperate variables for temperature based on oral or tympanic or other sources for sensitivity analyses if temp generally is significant.

#Total scores for neuro and physical exams (e.g. adding up number of abnormal domains across measures)

#Diagnostic categories; combine MCI groups into one and combine SMC and CN into 1 controls category


#Data exploration (overall data set)

##Preview dataset
df.head()        # First 5 rows
df.tail()        # Last 5 rows
df.sample(5)     # Random 5 rows
##check data shape and info
df.shape         # (rows, columns)
df.columns       # Column names
df.info()        # Data types & non-null counts
df.dtypes        # Just data types
##find missing data
df.isnull().sum()        # Total missing per column
df.isnull().mean()*100   # % of missing per column
df[df.isnull().any(axis=1)]  # Rows with missing values


#AS A TEAM WE CAN MEET TO DISCUSS ISSUES SURROUNDING THE MISSINGNESS OF DATA AND WHETHER FURTHER ACTION IS REQUIRED TO CORRECT
#PURSUANT CODE FOR MISSING DATA/IMPUTATION/DROPPING TO BE DETERMINED AFTER THIS EXPLORATION

#Count the number of entries per participant (RID) - should just be 1 each
dara['RID'].value_counts()

## Summarise Numeric Data
df.describe()     # count, mean, std, min, max, quartiles

## Check Unique Values & Frequencies
df['col'].value_counts()        # Count of each value
df['col'].nunique()             # Number of unique values

#THIS STEP SHOULD BE HELPFUL IN DETERMINING THE BREAKDOWN OF ABNORMAL VS NORMAL OR ABSENT VS PRESENT VALUES ON THE NEURO AND PHYSICAL EXAM
#THIS STEP WILL ALSO BE HELPFUL IN A PRELIMINARY IDENTIFICATION OF CODED MISSING VALUES (E.G -4) AND EXTREME OUTLIERS FOR YOUR NUMERIC VARIABLES
#THIS STEP SHOULD ALSO GIVE YOU MOST OF THE INFORMATION THAT YOU NEED ON YOUR BASELINE DEMOGRAPHIC/DIAGNOSITC BREAKDOWN OF THE SAMPLE (USUALLY TABLE 1 IN THE PAPER)

#FINALLY WE NEED HISTOGRAMS FOR THE DISTRIBUTION OF THE NUMERIC (NON-CATEGORICAL DATA) VARIABLES - CLINICAL AND COVARIATES
#WE ALSO NEED A METHOD TO DETECT OUTLIERS - DORAE PLEASE ATTEMPT TO WRITE SOME CODE FOR BOXPLOTS TO DO SO AND LAUREN TO SUPPORT










