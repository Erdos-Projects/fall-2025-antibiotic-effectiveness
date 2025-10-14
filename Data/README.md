# Folder Structure
The original csv files are too large to upload to GitHub. The individual files can be found in this website: https://datadryad.org/dataset/doi:10.5061/dryad.jq2bvq8kp
- data_sources.txt: Contains the links to website used to download the data.
- final_dataframes: Directory containing train and test split dataframes per each antibiotic.
  - ARMD_ecoli_raw_dataset.csv: Dataframe which contains all the columns from data_sources merged.
  - <antibiotic>_test_dataset.csv: testing dataset for a particular antibiotic.
  - <antibiotic>_train_dataset.csv: training dataset for a particular antibiotic.
