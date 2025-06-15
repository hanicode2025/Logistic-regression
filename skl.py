# logistic_churn.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset
df = pd.read_csv('C:/Users/T.Haneesh/logistic/churn-bigml-20.csv')

# Step 2: Convert 'Churn' to 0 and 1
df['Churn'] = df['Churn'].astype(int)

# Step 3: Encode categorical features
df['International plan'] = df['International plan'].map({'Yes': 1, 'No': 0})
df['Voice mail plan'] = df['Voice mail plan'].map({'Yes': 1, 'No': 0})

# Step 4: One-hot encode 'State' column
df = pd.get_dummies(df, columns=['State'], drop_first=True)

# Step 5: Separate features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Step 6: Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Let’s verify everything went well
print("✅ Data Preprocessing Complete!")
print("Training data shape:", X_train_scaled.shape)
print("Testing data shape:", X_test_scaled.shape)
