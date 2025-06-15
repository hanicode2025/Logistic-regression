import pandas as pd

# Load the dataset
df = pd.read_csv("C:/Users/T.Haneesh/Downloads/churn-bigml-20.csv")  # example path

# 1. Preview the first 5 rows
print(df.head())

# 2. Show column names and data types
print(df.info())

# 3. Summary statistics for numerical columns
print(df.describe())

# 4. Check for missing values
print(df.isnull().sum())
