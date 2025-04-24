#To be done within terminal:
#Set directory to Wolfson PNU - Dorae's folder SHEENA to advise on this step and to provide exact location of the data
#Now to create and activate environment to set the versions of python/packages.
#I am 99% sure that this needs to be done via the terminal. SHEENA/SARA to further advise on this.

#Create the environment from the myenv.yml file (decide whether yml file should be saved in Dorae's home or Wolfson PNU directory - a draft version is in the github). SHEENA/SARA to further advise on this.
conda env create -f myenv.yml

#Activate the new environment
conda activate myenv

#Verify that the new environment was installed correctly
conda env list

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
##Importing dataset. Obtain exact location of data from SHEENA. Or one data uploaded change your directory in the Apocrita interface. 
data = 'C:/datasets/adni_data.csv' #this is incorrect. Fix with correct path
adni = pd.read_csv(adni)

# In[3]:
#To begin creating our clinical dataset (vars of interest) we will filter our main dataset by subjects with only screening visits
adni_sc = adni[adni['VISCODE2']== 'sc']

#Ensure there are no duplicate IDs
#Select duplicate rows except first occurance based on RID (ADNI ID variable)
duplicate = adni_sc[adni_sc.duplicated(subset=['RID'])]
print("Duplicate Rows Based on RID :")
#Print the resultant dataframe
duplicate
#If duplicates found, consult collaboratory team to advise on removal of datapoints

#VIS DATE DATAFRAME
  #Extract the RID, and the different VisDates. 
  #-	Here we want a vis date screening variable
  #Combine all vis dates into a table linked by the P ID. 
    # If 3 = then create a df -  
    #	If not – create df. 
    # This would also tell us who would be dropped. 
     # Then get a masters screening vis. Undecided on how at the moment. Determine missing data. 

#from screening dataset, select all relevant ID, vis_dates and clinical variables. Need to consult with SHEENA on new variable names. 
#VISDATE can be any from the screening visit (Neuro, Phys, or Vitals) - ideally these should be on the same date/close enough in time to count as one timepoint. Check the exam dates to see if they differ?
adni_sc_clin = adni_sc[["RID", "VISCODE2", "VISDATE", 
                        "PXGENAPP", "PXHEADEY", "PXNECK", "PXCHEST", "PXHEART", "PXABDOM", "PXEXTREM", "PXEDEMA", "PXPERIPH", "PXSKIN", "PXMUSCUL", "PXBACK", "PXOTHER",
                        "NXVISUAL", "NXAUDITO", "NXTREMOR", "NXCONSCI", "NXNERVE", "NXMOTOR", "NXFINGER", "NXHEEL", "NXSENSOR", "NXTENDON", "NXPLANTA", "NXGAIT", "NXOTHER",
                        "VSWEIGHT", "VSWTUNIT", "VSHEIGHT", "VSHTUNIT", "VSBPSYS", "VSBPDIA", "VSPULSE", "VSRESP", "VSTEMP", "VSTMPSRC", "VSTMPUNT"]]

# Fix values coded incorrectly as -4 (missing data)
adni_sc_clin[col] = adni_sc_clin[col].replace(-4,-1, np.nan) #this was based on manual inspection of the data values in excel. May need to justify in R later.

#Convert categorical variables into factors
categorical_vars = [
      "PXGENAPP", "PXHEADEY", "PXNECK", "PXCHEST", "PXHEART", "PXABDOM", "PXEXTREM", "PXEDEMA", "PXPERIPH", "PXSKIN", "PXMUSCUL", "PXBACK", "PXOTHER",
                        "NXVISUAL", "NXAUDITO", "NXTREMOR", "NXCONSCI", "NXNERVE", "NXMOTOR", "NXFINGER", "NXHEEL", "NXSENSOR", "NXTENDON", "NXPLANTA", "NXGAIT", "NXOTHER",
                        "VSWTUNIT", "VSHTUNIT", "VSTMPSRC", "VSTMPUNT"]]

for col in categorical_vars:
      adni_sc_clin_v2[col] = adni_sc_clin_v2[col].astype('category')

# Make sure numeric values are treated as numbers
  numeric_vars = ["VSWEIGHT", "VSHEIGHT", "VSBPSYS", "VSBPDIA", "VSPULSE", "VSRESP", "VSTEMP"]
  for col in numeric_vars:
      adni_sc_clin_v2[col] = pd.to_numeric(adni_sc_clin_v2[col], errors='coerce')

# Descriptive stats
adni_sc_clin_v2.describe()

#calculating BMI
#Variables to use:
#VSWEIGHT (1a. Weight; Numeric) #VSWTUNIT (1b. Weight Units; 1=pounds; 2=kilograms)
#VSHEIGHT (2a. Height; Numeric) #VSHTUNIT (2b. Height Units; 1=inches; 2=centimeters)

#the first step is to convert pounds to kilograms 
if VSWTUNIT 


#create a VSWEIGHT variable with just kg

































# In[4] #TRANSFORMATIONS ON EXISTING VARIABLES- DORAE TO ATTEMPT AND LAUREN TO ADVISE HERE
#Convert categorical variables (numeric) into factors - this is the majority of our clinical variables and our demographic/diagnostic variables


#Turning categorical variables into factors/numerical using import pandas as pd
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


