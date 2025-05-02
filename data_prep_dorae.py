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
  #Extract the RID, and the different VisDates. Need to consult SHEENA about whether there are seperate VISDATES in the master dataset for Neuro, Phys, and Vitals. 
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

##change variables to match Sheena's data dictionary 
adni_sc_clin = adni_sc[["RID", "VISCODE2", "VISDATE", 
                        "PXGENAPP_PHYSICAL", "PXHEADEY_PHYSICAL", "PXNECK_PHYSICAL", "PXCHEST_PHYSICAL", "PXHEART_PHYSICAL", "PXABDOM_PHYSICAL", "PXEXTREM_PHYSICAL", "PXEDEMA_PHYSICAL", "PXPERIPH_PHYSICAL", "PXSKIN_PHYSICAL", "PXMUSCUL_PHYSICAL", "PXBACK_PHYSICAL", "PXOTHER_PHYSICAL",
                        "NXVISUAL_NEUROEXM", "NXAUDITO_NEUROEXM", "NXTREMOR_NEUROEXM", "NXCONSCI_NEUROEXM", "NXNERVE_NEUROEXM", "NXMOTOR_NEUROEXM", "NXFINGER_NEUROEXM", "NXHEEL_NEUROEXM", "NXSENSOR_NEUROEXM", "NXTENDON_NEUROEXM", "NXPLANTA_NEUROEXM", "NXGAIT_NEUROEXM", "NXOTHER_NEUROEXM",
                        "VSWEIGHT_VITALS", "VSWTUNIT_VITALS", "VSHEIGHT_VITALS", "VSHTUNIT_VITALS", "VSBPSYS_VITALS", "VSBPDIA_VITALS", "VSPULSE_VITALS", "VSRESP_VITALS", "VSTEMP_VITALS", "VSTMPSRC_VITALS", "VSTMPUNT_VITALS"]]


# Fix values coded incorrectly as -4 oe -1 (missing data)
adni_sc_clin[col] = adni_sc_clin[col].replace(-4,-1, np.nan) #this was based on manual inspection of the data values in excel. May need to justify in Python/Jupyter later.

#Convert categorical variables into factors
categorical_vars = [
      "PXGENAPP", "PXHEADEY", "PXNECK", "PXCHEST", "PXHEART", "PXABDOM", "PXEXTREM", "PXEDEMA", "PXPERIPH", "PXSKIN", "PXMUSCUL", "PXBACK", "PXOTHER",
                        "NXVISUAL", "NXAUDITO", "NXTREMOR", "NXCONSCI", "NXNERVE", "NXMOTOR", "NXFINGER", "NXHEEL", "NXSENSOR", "NXTENDON", "NXPLANTA", "NXGAIT", "NXOTHER",
                        "VSWTUNIT", "VSHTUNIT", "VSTMPSRC", "VSTMPUNT"]]

##change variables to match Sheena's data dictionary 
categorical_vars = [
      "PXGENAPP_PHYSICAL", "PXHEADEY_PHYSICAL", "PXNECK_PHYSICAL", "PXCHEST_PHYSICAL", "PXHEART_PHYSICAL", "PXABDOM_PHYSICAL", "PXEXTREM_PHYSICAL", "PXEDEMA_PHYSICAL", "PXPERIPH_PHYSICAL", "PXSKIN_PHYSICAL", "PXMUSCUL_PHYSICAL", "PXBACK_PHYSICAL", "PXOTHER_PHYSICAL",
                        "NXVISUAL_NEUROEXM", "NXAUDITO_NEUROEXM", "NXTREMOR_NEUROEXM", "NXCONSCI_NEUROEXM", "NXNERVE_NEUROEXM", "NXMOTOR_NEUROEXM", "NXFINGER_NEUROEXM", "NXHEEL_NEUROEXM", "NXSENSOR_NEUROEXM", "NXTENDON_NEUROEXM", "NXPLANTA_NEUROEXM", "NXGAIT_NEUROEXM", "NXOTHER_NEUROEXM",
                        "VSWEIGHT_VITALS", "VSWTUNIT_VITALS", "VSHEIGHT_VITALS", "VSHTUNIT_VITALS", "VSBPSYS_VITALS", "VSBPDIA_VITALS", "VSPULSE_VITALS", "VSRESP_VITALS", "VSTEMP_VITALS", "VSTMPSRC_VITALS", "VSTMPUNT_VITALS"]]


for col in categorical_vars:
      adni_sc_clin_v2[col] = adni_sc_clin_v2[col].astype('category')


# Make sure numeric values are treated as numbers
  numeric_vars = ["VSWEIGHT", "VSHEIGHT", "VSBPSYS", "VSBPDIA", "VSPULSE", "VSRESP", "VSTEMP"]
  for col in numeric_vars:
      adni_sc_clin_v2[col] = pd.to_numeric(adni_sc_clin_v2[col], errors='coerce')

##change variables to match Sheena's data dictionary 
numeric_vars = ["VSWEIGHT_VITALS", "VSHEIGHT_VITALS", "VSBPSYS_VITALS", "VSBPDIA_VITALS", "VSPULSE_VITALS", "VSRESP_VITALS", "VSTEMP_VITALS"]
  for col in numeric_vars:
      adni_sc_clin_v2[col] = pd.to_numeric(adni_sc_clin_v2[col], errors='coerce')


# Descriptive stats
adni_sc_clin_v2.describe()

#calculating BMI
#Variables to use:
#VSWEIGHT (1a. Weight; Numeric) #VSWTUNIT (1b. Weight Units; 1=pounds; 2=kilograms)

#the first step is to convert pounds to kilograms 

#VSHEIGHT (2a. Height; Numeric) #VSHTUNIT (2b. Height Units; 1=inches; 2=centimeters)

# BMI conversion functions
def pounds_to_kg(pounds):
    return pounds * 0.45359237

def inches_to_cm(inches):
    return inches * 2.54

# Apply conversions conditionally
adni_sc_clin_v2['VSWEIGHT'] = adni_sc_clin_v2.apply(lambda row: pounds_to_kg(row['VSWEIGHT']) if row['VSWTUNIT'] == 1 else row['VSWEIGHT'], axis=1)
adni_sc_clin_v2['VSHEIGHT'] = adni_sc_clin_v2.apply(lambda row: inches_to_cm(row['VSHEIGHT']) if row['VSHTUNIT'] == 1 else row['VSHEIGHT'], axis=1)

##change variables to match Sheena's data dictionary 
adni_sc_clin_v2['VSWEIGHT_VITALS'] = adni_sc_clin_v2.apply(lambda row: pounds_to_kg(row['VSWEIGHT_VITALS']) if row['VSWTUNIT_VITALS'] == 1 else row['VSWEIGHT_VITALS'], axis=1)
adni_sc_clin_v2['VSHEIGHT_VITALS'] = adni_sc_clin_v2.apply(lambda row: inches_to_cm(row['VSHEIGHT_VITALS']) if row['VSHTUNIT_VITALS'] == 1 else row['VSHEIGHT_VITALS'], axis=1)



#calculate BMI
BMI = adni_sc_clin_v2['VSWEIGHT'] / (adni_sc_clin_v2['VSHEIGHT']/100)**2 

##change variables to match Sheena's data dictionary 
BMI = adni_sc_clin_v2['VSWEIGHT_VITALS'] / (adni_sc_clin_v2['VSHEIGHT_VITALS']/100)**2 


#Create BMI category and name. LAUREN TO DOUBLECHECK THIS CODE.
adni_sc_clin_v2['BMI_CATEGORY']= if bmi<18.5:
                             print("Underweight")
                            elif bmi>=18.5 and bmi<25:
                             print("Normal")
                            elif bmi>=25 and bmi<30:
                            print("Overweight")
                            else:
                            print("Obesity")

adni_sc_clin_v2['BMI_CATEGORY'] = adni_sc_clin_v2['BMI_CATEGORY'].astype('category')

#Calculate MAP. LAUREN TO DOUBLECHECK THIS CODE.
MAP = (adni_sc_clin_v2['VSBPSYS'] + 2(adni_sc_clin_v2['VSBPDIA']))/3

##change variables to match Sheena's data dictionary 
MAP = (adni_sc_clin_v2['VSBPSYS_VITALS'] + 2(adni_sc_clin_v2['VSBPDIA_VITALS']))/3

#Convert all temperature unit from farenheit to celsius. LAUREN TO DOUBLECHECK THIS CODE. 

def fahrenheit_to_celsius(fahrenheit):
    celsius = (fahrenheit - 32) * 5 / 9
    return celsius

adni_sc_clin_v2['VSTEMP'] = adni_sc_clin_v2.apply(lambda row: fahrenheit_to_celsius(row['VSTEMP']) if row['VSTMPUNT'] == 1 else row['VSTEMP'], axis=1)

##change variables to match Sheena's data dictionary 
adni_sc_clin_v2['VSTEMP_VITALS'] = adni_sc_clin_v2.apply(lambda row: fahrenheit_to_celsius(row['VSTEMP_VITALS']) if row['VSTMPUNT_VITALS'] == 1 else row['VSTEMP_VITALS'], axis=1)



##DORAE TO WRITE CODE FOR TOTAL NE AND PE SCORES***************************************************************************************************************************************************
##PREVIOUS CODE COPY AND PASTED BELOW

Physical examination variables (1= normal, 2=abnormal)
#Abdomen #Back #Chest #Oedema #Extremeties #General appearance #Head, Eyes, ENT #Heart #MSK #Neck #Other #Peripheral vascular #Skin and Appendages 

Neurological examination variables (1= normal, 2=abnormal)
"NXVISUAL", "NXAUDITO", "NXTREMOR", "NXCONSCI", "NXNERVE",
"NXMOTOR", "NXFINGER", "NXHEEL", "NXSENSOR", "NXTENDON", "NXPLANTA", "NXGAIT", "NXOTHER"

##change variables to match Sheena's data dictionary 
"NXVISUAL_NEUROEXM","NXAUDITO_NEUROEXM","NXTREMOR_NEUROEXM","NXCONSCI_NEUROEXM","NXNERVE_NEUROEXM","NXMOTOR_NEUROEXM",
"NXFINGER_NEUROEXM","NXHEEL_NEUROEXM","NXSENSOR_NEUROEXM","NXTENDON_NEUROEXM","NXPLANTA_NEUROEXM","NXGAIT_NEUROEXM","NXOTHER_NEUROEXM"

#file name adni_sc
Neurological examination 
# Variables to sum
neurological_vars = [
    "NXVISUAL", "NXAUDITO", "NXTREMOR", "NXCONSCI", "NXNERVE",
    "NXMOTOR", "NXFINGER", "NXHEEL", "NXSENSOR", "NXTENDON",
    "NXPLANTA", "NXGAIT", "NXOTHER"
]

#change variables to match Sheena's data dictionary
neurological_vars = [
    "NXVISUAL_NEUROEXM","NXAUDITO_NEUROEXM","NXTREMOR_NEUROEXM","NXCONSCI_NEUROEXM","NXNERVE_NEUROEXM","NXMOTOR_NEUROEXM",
    "NXFINGER_NEUROEXM","NXHEEL_NEUROEXM","NXSENSOR_NEUROEXM","NXTENDON_NEUROEXM","NXPLANTA_NEUROEXM","NXGAIT_NEUROEXM","NXOTHER_NEUROEXM"
]

# Open the file
with open('adni_sc.csv', 'r') as file:
    reader = csv.DictReader(file)

# Loop through each individual's row
for row in reader:
        participant_id = row.get("RID") # Adjust if there's a unique ID column

# Sum only neurological variables, handling missing/empty values safely
total_score = sum(
            int(row[var]) if row[var].isdigit() else 0 for var in neurological_vars
        )

print(f"Participant {participant_id} - Total Neurological Score: {total_score}")


Physical examination 
##physical examination domains (1= normal, 2=abnormal)
#Abdomen #Back #Chest #Oedema #Extremeties #General appearance #Head, Eyes, ENT #Heart #MSK #Neck #Other #Peripheral vascular #Skin and Appendages 

"PXGENAPP", "PXHEADEY", "PXNECK", "PXCHEST", "PXHEART", "PXABDOM", "PXEXTREM",
"PXEDEMA", "PXPERIPH", "PXSKIN", "PXMUSCUL", "PXBACK", "PXOTHER",

PXGENAPP_PHYSICAL
PXHEADEY_PHYSICAL
PXNECK_PHYSICAL
PXCHEST_PHYSICAL
PXHEART_PHYSICAL
PXABDOM_PHYSICAL
PXEXTREM_PHYSICAL
PXEDEMA_PHYSICAL
PXPERIPH_PHYSICAL
PXSKIN_PHYSICAL
PXMUSCUL_PHYSICAL
PXBACK_PHYSICAL
PXOTHER_PHYSICAL

#file name adni_sc
Physical examination 
# Variables to sum
physical_vars = [
    "PXGENAPP", "PXHEADEY", "PXNECK", "PXCHEST", 
    "PXHEART", "PXABDOM", "PXEXTREM", "PXEDEMA", 
    "PXPERIPH", "PXSKIN", "PXMUSCUL", "PXBACK", "PXOTHER"
]

##change variables to match Sheena's data dictionary
physical_vars = [
    "PXGENAPP_PHYSICAL","PXHEADEY_PHYSICAL","PXNECK_PHYSICAL","PXCHEST_PHYSICAL",
    "PXHEART_PHYSICAL","PXABDOM_PHYSICAL","PXEXTREM_PHYSICAL","PXEDEMA_PHYSICAL",
    "PXPERIPH_PHYSICAL","PXSKIN_PHYSICAL","PXMUSCUL_PHYSICAL","PXBACK_PHYSICAL",
    "PXOTHER_PHYSICALPXGENAPP"
]

# Open the file
with open('adni_sc.csv', 'r') as file:
    reader = csv.DictReader(file)

# Loop through each individual's row
for row in reader:
        participant_id = row.get("RID") # Adjust if there's a unique ID column

# Sum only neurological variables, handling missing/empty values safely
total_score = sum(
            int(row[var]) if row[var].isdigit() else 0 for var in physical_vars
        )

print(f"Participant {participant_id} - Total Physical Score: {total_score}")


///////////

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

///////////

##DORAE TO CREATE HISTOGRAMS FOR ALL NUMERIC CLINICAL VARIABLES***************************************************************************************************************************************************

Numeric Clinical Variables
"VSWEIGHT", "VSWTUNIT", "VSHEIGHT", "VSHTUNIT", "VSBPSYS", "VSBPDIA", "VSPULSE", "VSRESP", "VSTEMP", "VSTMPSRC", "VSTMPUNT"

VSWEIGHT_VITALS
VSWTUNIT_VITALS
VSHEIGHT_VITALS
VSHTUNIT_VITALS
VSBPSYS_VITALS
VSBPDIA_VITALS
VSPULSE_VITALS
VSRESP_VITALS
VSTEMP_VITALS
VSTMPSRC_VITALS
VSTMPUNT_VITALS

##EXAMPLE PROVIDED BY LAUREN BELOW. PLEASE REPEAT FOR ALL NUMERIC CLINICAL VARIABLES

  #Histograms show us if most values are grouped together, if there are weird spikes, or if anything looks strange.
  #Example: Look at distribution of systolic blood pressure
      plt.figure(figsize=(8, 5))
      sns.histplot(combined_df['VSBPSYS'].dropna(), bins=30, kde=True)
      plt.title('Systolic Blood Pressure Distribution')
      plt.xlabel('Systolic BP (mmHg)')
      plt.ylabel('Count')
      plt.show()
    #Repeat that for any variable you're interested in (VSHEIGHT, VSWEIGHT, etc.).

  #Look at distribution of diastolic blood pressure
      plt.figure(figsize=(8, 5))
      sns.histplot(combined_df['VSBPDIA'].dropna(), bins=30, kde=True)
      plt.title('Diastolic Blood Pressure Distribution')
      plt.xlabel('Diastolic BP (mmHg)')
      plt.ylabel('Count')
      plt.show()

  #Look at distribution of Height
      plt.figure(figsize=(8, 5))
      sns.histplot(combined_df['VSHEIGHT'].dropna(), bins=30, kde=True)
      plt.title('Height Distribution')
      plt.xlabel('Height (cm)')
      plt.ylabel('Count')
      plt.show()

  #Look at distribution of weight
      plt.figure(figsize=(8, 5))
      sns.histplot(combined_df['VSWEIGHT'].dropna(), bins=30, kde=True)
      plt.title('Weight Distribution')
      plt.xlabel('Weight (kg)')
      plt.ylabel('Count')
      plt.show()

  #Look at distribution of Pulse rate (beats per minute)
      plt.figure(figsize=(8, 5))
      sns.histplot(combined_df['VSPULSE'].dropna(), bins=30, kde=True)
      plt.title('Pulse Rate Distribution')
      plt.xlabel('Pulse rate (bpm)')
      plt.ylabel('Count')
      plt.show()

#Look at distribution of respiratory rate (breaths per minute)
      plt.figure(figsize=(8, 5))
      sns.histplot(combined_df['VSRESP'].dropna(), bins=30, kde=True)
      plt.title('Resrpiratory Rate Distribution')
      plt.xlabel('Respiratory Rate (bpm)')
      plt.ylabel('Count')
      plt.show()


#Look at distribution of Temperature 
      plt.figure(figsize=(8, 5))
      sns.histplot(combined_df['VSTEMP'].dropna(), bins=30, kde=True)
      plt.title('Pulse Rate Distribution')
      plt.xlabel('Pulse rate (bpm)')
      plt.ylabel('Count')
      plt.show()


#??Look at distribution of BMI
      plt.figure(figsize=(8, 5))
      sns.histplot(combined_df['BMI'].dropna(), bins=30, kde=True)
      plt.title('BMI Distribution')
      plt.xlabel('BMI (kg/m²)')
      plt.ylabel('Count')
      plt.show()




##DORAE TO DO OUTLIER DETECTION FOR ALL NUMERIC CLINICAL VARIABLES***************************************************************************************************************************************************
##EXAMPLE PROVIDED BY LAUREN BELOW. PLEASE REPEAT FOR ALL NUMERIC CLINICAL VARIABLES

#OUTLIER DETECTION METHODS FOR NUMERIC VARIABLES
  #Outliers can be data errors, or they can point to unusual cases we want to know about.
  #USE IRQ - easiest and most common.
  
  #def detect_outliers_iqr(data, column):
      q1 = data[column].quantile(0.25)
      q3 = data[column].quantile(0.75)
      iqr = q3 - q1
      lower_bound = q1 - 1.5 * iqr
      upper_bound = q3 + 1.5 * iqr
      return data[(data[column] < lower_bound) | (data[column] > upper_bound)]
  
  # Find outliers in systolic BP
  outliers_bp = detect_outliers_iqr(combined_df, 'VSBPSYS')
  print("Number of outliers in Systolic BP:", outliers_bp.shape[0])

  # Find outliers in diastolic  BP
  outliers_bp = detect_outliers_iqr(combined_df, 'VSBPDIA')
  print("Number of outliers in diastolic BP:", outliers_bp.shape[0])

  # Find outliers in BMI
  outliers_bp = detect_outliers_iqr(combined_df, 'BMI')
  print("Number of outliers in BMI:", outliers_bp.shape[0])

  # Find outliers in weight
  outliers_bp = detect_outliers_iqr(combined_df, 'VSWEIGHT')
  print("Number of outliers in weight:", outliers_bp.shape[0])

 # Find outliers in height
  outliers_bp = detect_outliers_iqr(combined_df, 'VSHEIGHT')
  print("Number of outliers in height:", outliers_bp.shape[0])

  # Find outliers in pulse rate (beats per minute)
  outliers_bp = detect_outliers_iqr(combined_df, 'VSPULSE')
  print("Number of outliers in pulse rate:", outliers_bp.shape[0])

  # Find outliers in respiratory rate (breaths per minute)
  outliers_bp = detect_outliers_iqr(combined_df, 'VSRESP')
  print("Number of outliers in pulse rate:", outliers_bp.shape[0])

  # Find outliers in temperature
  outliers_bp = detect_outliers_iqr(combined_df, 'VSTEMP')
  print("Number of outliers in pulse rate:", outliers_bp.shape[0])



# In[4]:
#To begin creating our demographic/diagnostic dataset (covariates/outcomes) we will filter our main dataset by subjects with only baseline visits
adni_bl = adni[adni['VISCODE2']== 'bl']

#Ensure there are no duplicate IDs
#Select duplicate rows except first occurance based on RID (ADNI ID variable)
duplicate = adni_bl[adni_bl.duplicated(subset=['RID'])]
print("Duplicate Rows Based on RID :")
#Print the resultant dataframe
duplicate
#If duplicates found, consult collaboratory team to advise on removal of datapoints


#from baseline dataset, select all relevant ID, exam dates and covariate/outcome variables. Need to consult with SHEENA on new variable names. 
#VISDATE can be any from the screening visit (Neuro, Phys, or Vitals) - ideally these should be on the same date/close enough in time to count as one timepoint. Check the exam dates to see if they differ?
adni_bl_demo_diag = adni_bl[["RID", "VISCODE2", "EXAMDATE", "SITE",
                       "DX_bl", "AGE", "PTGENDER", "PTEDUCAT", "PTRACCAT", "APOE4", "PTMARRY"]]

# Fix values coded incorrectly as -4 oe -1 (missing data)
adni_bl_demo_diag[col] = adni_bl_demo_diag[col].replace(-4,-1, np.nan) #this was based on manual inspection of the data values in excel. May need to justify in Python/Jupyter later.

#Convert categorical variables into factors
categorical_vars = ["DX_bl", "PTGENDER", "PTRACCAT", "APOE4", "PTMARRY"]

for col in categorical_vars:
      adni_bl_demo_diag_v2[col] = adni_bl_demo_diag_v2[col].astype('category')

# Make sure numeric values are treated as numbers
numeric_vars = ["AGE", "PTEDUCAT"]
  
for col in numeric_vars:
      adni_bl_demo_diag_v2[col] = pd.to_numeric(adni_bl_demo_diag_v2[col], errors='coerce')

# Descriptive stats
adni_bl_demo_diag_v2.describe()



# In[5]:

##DORAE TO WRITE CODE TO MERGE SCREENING AND DEMO/DIAGNOSTIC DATAFRAMES BASED ON RID***************************************************************************************************************************************************
#screening data: adni_sc_clin_v2
#baseline data: adni_bl_demo_diag_v2
#merging variable: RID
#Guide to merging in PANDAS: https://pandas.pydata.org/docs/user_guide/merging.html 
#code from Sara below:

merged_on_proteomics_inner = pd.merge(proteomics, metabolomics_data_1, on='eid', how='inner')
participants_after_proteomics_inner_join = merged_on_proteomics_inner['eid'].nunique()

###two different datasets for SC and BL, both have RID - need to merge data sets together based on RIS
###if SC and BL data on same spreadsheet then this may not be required 





##DORAE TO WRITE CODE FOR CALCULATING SCREENING AGE AND TIME FROM SC TO BL VISIT. WILL NEED TO WRITE CODE RENAMING THE 2 VISDATE/EXAMDATE VARIABLES***************************************************************************************************************************************************
##EXAMPLE CODE PROVIDED BY LAUREN BELOW

 ## Convert dates to datetime format
  combined_df['EXAMDATE_BL'] = pd.to_datetime(combined_df['EXAMDATE_BL'])
  combined_df['EXAMDATE_SC'] = pd.to_datetime(combined_df['EXAMDATE_SC'])
  
  # Calculate how much time has passed (in years)
  combined_df['AGE_DIFF_YEARS'] = (combined_df['EXAMDATE_SC'] - combined_df['EXAMDATE_BL']).dt.days / 365.25
  
  # Calculate age at screening
  combined_df['AGE_SC'] = combined_df['AGE'] + combined_df['AGE_DIFF_YEARS']
  
  # Age squared
  combined_df['AGE_SC_SQ'] = combined_df['AGE_SC'] ** 2


VISDATE NE, PE, Vitals
print dates to visualise dates 
if all the same, then report that date
if different dates - will have to decide what age to use 































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



