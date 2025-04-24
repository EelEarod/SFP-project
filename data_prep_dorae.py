
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
##Importing dataset
##WE WILL NEED TO DISCUSS WITH SHEENA HOW TO SET YOUR DIRECTORY TO PNU WOLFSON WITHIN THE PYTHON ENVIRONMENT
data = 'C:/datasets/adni_data.csv'
df = pd.read_csv(data)

# In[3]:
#Extract the data for clinical variables from screening visit (VISCODE2 = 'sc')
#Include only adni screening visits 
#step 1: filter entries with VISCODE sc
adni = adni[adni['VISCODE']== 'sc' ]

#clinical variables: physical examination (1=normal, 2= abnormal)
#PXGENAPP (1. General Appearance)
#PXHEADEY (2. Head, Eyes, Ears, Nose and Throat
#PXNECK (3. Neck)
#PXCHEST (4. Chest)
#PXHEART (5. Heart)
#PXABDOM (6. Abdomen)
#PXEXTREM (7. Extremities)
#PXEDEMA (8. Edema)
#PXPERIPH (9. Peripheral Vascular)
#PXSKIN (10. Skin and Appendages)
#PXMUSCUL (11. Musculoskeletal)
#PXBACK (12. Back)
#PXOTHER (13. Other)

#clinical variables: neurological examination (1=Absent, 2= present)
#NXVISUAL (1. Significant Visual Impairment)
#NXAUDITO (2. Significant Auditory Impairment)
#NXTREMOR (3. Tremor)
#NXCONSCI (4. Level of Consciousness)
#NXNERVE (5. Cranial Nerves)
#NXMOTOR (6. Motor Strength)
#NXFINGER (7a. Cerebellar - Finger to Nose)
#NXHEEL (7b. Cerebellar - Heel to Shin)
#NXSENSOR (8. Sensory)
#NXTENDON (9. Deep Tendon Reflexes)
#NXPLANTA (10. Plantar Reflexes)
#NXGAIT (11. Gait)
#NXOTHER (12. Other)

#Clinical variables: Vital signs 
#VSWEIGHT (1a. Weight; Numeric) #VSWTUNIT (1b. Weight Units; 1=pounds; 2=kilograms)
#VSHEIGHT (2a. Height; Numeric) #VSHTUNIT (2b. Height Units; 1=inches; 2=centimeters)
#VSBPSYS (3a. Systolic  BP- mmHg; range: 76-250)
#VSBPDIA (3b. Diastolic  BP- mmHg; range: 36-130)
#VSPULSE (4. Seated Pulse Rate (per minute); range: 40-130)
#VSRESP (5. Respirations (per minute); range: 6-40)
#VSTEMP (6a. Temperature; range: 6-40) #VSTMPSRC (6b. Temperature Source; 1=Oral, 2=Tympanic, 3=Other); VSTMPUNT (6c. Temperature Units, 1=Fahrenheit; 2=Celsius)

#Identifier variable: Across all datasets
#RID (Participant roster ID)

#Screening clinical exam date variable: PHYSICAL, NEUROEXM, VITALS
#EXAMDATE (Clinical Screening Examination Date)

#Demographic and diagnostic variables at baseline characteristics: ADNIMERGE
#EXAMDATE (Baseline examination date)
#SITE (ADNI exam site)
#DX_bl (Baseline diagnosis; AD=Alzheimer's disease; CN=cognitively normal; EMCI=early MCI; LMCI=late MCI; SMC=subjective memory concern)
#AGE (Baseline age)
#PTGENDER (Gender)
#PTEDUCAT (Education)
#PTRACCAT (Race)
#APOE4 (APOE e4 carrier status; 0=non-carrier; 1=heterozygous carrier; 2=homozygous carrier)



# In[4] #TRANSFORMATIONS ON EXISTING VARIABLES- DORAE TO ATTEMPT AND LAUREN TO ADVISE HERE
#Convert categorical variables (numeric) into factors - this is the majority of our clinical variables and our demographic/diagnostic variables
#turning categorical variables into factors/numerical using import pandas as pd
PTGENDER, PTEDUCAT, PTETHCAT, PTRACCAT, PTMARRY, APOE4

?info not available for ADNIMERGE
PTGENDER = 1=Male; 2=Female
PTEDUCAT = 0..20
PTETHCAT = 1=Hispanic or Latino; 2=Not Hispanic or Latino; 3=Unknown
PTRACCAT = 1=American Indian or Alaskan Native; 2=Asian; 3=Native Hawaiian or Other Pacific Islander; 4=Black or African American; 5=White; 6=More than one race; 7=Unknown
PTMARRY = 1=Married; 2=Widowed; 3=Divorced; 4=Never married; 5=Unknown
APOE4

# Sample DataFrame
df = pd.DataFrame({
    'PTGENDER': ['Male', 'Female', 'Female', 'Male', 'Other']
})
# Convert to category codes
df['Gender_encoded'] = df['PTGENDER].astype('category').cat.codes
print(df)


#Ensure numeric variables are classified as numeric and not another category (e.g. string)
#Ensure missing values are coded properly (for example may be -4 for some of the height values where the data not collected properly)





#NEW VARIABLES TO CALCULATE - DORAE TO ATTEMPT AND LAUREN TO ADVISE HERE
e.g. BMI, MAP, 
total NE score, total PE score 
#Total scores for neuro and physical exams (e.g. adding up number of abnormal domains across measures)

##physical examination domains (1= normal, 2=abnormal) --> ?change to 0=normal, 1=abnormal 
#Abdomen #Back #Chest #Oedema #Extremeties #General appearance #Head, Eyes, ENT #Heart #MSK #Neck #Other #Peripheral vascular #Skin and Appendages 

df = pd.read_csv("adni_physical_examination_domains") 
# Calculate total sum of scores
total_score = df['normal/abnormal'].sum()
print("Total Score:", total_score)

##Total number of abnormal neurological examination domains 
# Load the CSV
df = pd.read_csv("adni_neurological_examination_domains") #change file name 
# Calculate total sum of scores
total_score = df['normal/abnormal'].sum()
print("Total Score:", total_score)

# In[10]:
#Making new variables based on existing ones
## BMI
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

## MAP
VSBPSYS=float(input))
VSBPDIA=float(input))
MAP = (VSBPSYS + 2(VSBPDIA))/3

#Temperature
#standardise temperature - celsius. 

#create seperate variables for temperature based on oral or tympanic or other sources for sensitivity analyses if temp generally is significant.
 

# In[5]
##check data shape and info
df.shape         # (rows, columns)
df.columns       # Column names
df.info()        # Data types & non-null counts
df.dtypes        # Just data types
##Preview dataset
df.head()        # First 5 rows
df.tail()        # Last 5 rows
df.sample(5)     # Random 5 rows


