# Define mapping dictionary
converter_map = {
    "MCI_Stable": "Control_MCIStable",
    "Control_Stable": "Control_MCIStable",
    "Dementia_MCIConverter": "MCIConverter_Dementia",
    "Dementia_Stable": "MCIConverter_Dementia"
}

# Apply mapping to create new column
adni_merged_clean2["stableconverter"] = adni_merged_clean2["DIAG_Conversion_Simple"].map(converter_map)

# View unique values in new column
print(adni_merged_clean2["stableconverter"].value_counts())

# Optional: Crosstab to verify mapping
print(pd.crosstab(adni_merged_clean2["DIAG_Conversion_Simple"], adni_merged_clean2["stableconverter"]))

adni_merged_clean2.head(10)
