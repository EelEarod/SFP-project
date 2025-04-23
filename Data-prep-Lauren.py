#VARIABLE TRANSFORMATIONS

  ##For VARIABLE TRANSFORMATIONS we are fixing variable types — for example, if a variable is supposed to be a category (like ‘Normal’ vs ‘Abnormal’), we don’t want Python to treat it like a number.
  
  # Convert clinical and demographic codes to categorical
  categorical_vars = [
      'PXGENAPP', 'PXHEADEY', 'PXNECK', 'PXCHEST', 'PXHEART', 'PXABDOM',
       'APOE4', ETC....
  ]
  
  for col in categorical_vars:
      combined_df[col] = combined_df[col].astype('category')
  
  # Make sure numeric values are treated as numbers
  numeric_vars = ['VSWEIGHT', 'VSHEIGHT', 'VSBPSYS', 'VSBPDIA', 'VSPULSE', 'VSRESP', 'VSTEMP']
  for col in numeric_vars:
      combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
  
  # Fix values coded incorrectly as -4 (missing data)
  combined_df['VSHEIGHT'] = combined_df['VSHEIGHT'].replace(-4, np.nan)


#MAKING NEW VARIABLES

  #We want to know someone’s age at the screening visit, not just baseline, and we’ll also make an age-squared variable, which can be useful in modelling.
  
  ## Convert dates to datetime format
  combined_df['EXAMDATE_BL'] = pd.to_datetime(combined_df['EXAMDATE_BL'])
  combined_df['EXAMDATE_SC'] = pd.to_datetime(combined_df['EXAMDATE_SC'])
  
  # Calculate how much time has passed (in years)
  combined_df['AGE_DIFF_YEARS'] = (combined_df['EXAMDATE_SC'] - combined_df['EXAMDATE_BL']).dt.days / 365.25
  
  # Calculate age at screening
  combined_df['AGE_SC'] = combined_df['AGE'] + combined_df['AGE_DIFF_YEARS']
  
  # Age squared
  combined_df['AGE_SC_SQ'] = combined_df['AGE_SC'] ** 2


#HISTOGRAMS FOR NUMERIC VARIABLE DISTRIBUTIONS

  #Histograms show us if most values are grouped together, if there are weird spikes, or if anything looks strange.

  #Example: Look at distribution of systolic blood pressure
      plt.figure(figsize=(8, 5))
      sns.histplot(combined_df['VSBPSYS'].dropna(), bins=30, kde=True)
      plt.title('Systolic Blood Pressure Distribution')
      plt.xlabel('Systolic BP (mmHg)')
      plt.ylabel('Count')
      plt.show()
  
  #Repeat that for any variable you're interested in (VSHEIGHT, VSWEIGHT, etc.).
  
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
