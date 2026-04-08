import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load the dataset
df = pd.read_csv('Dataset/insurance.csv')

# 2. Basic Cleaning & Encoding
# Manual Binary Encoding
df['sex'] = df['sex'].map({'female': 1, 'male': 0})
df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})

# Label Encoding for Region
df['region'] = df['region'].astype('category').cat.codes

# 3. Feature-Target Split
X = df.drop('charges', axis=1)
y = df['charges']

# 4. Train-Test Split (Do this BEFORE scaling)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Scaling (Preventing Leakage)
num_features = ['age', 'bmi', 'children']
scaler = StandardScaler()

# Fit and transform only the training numerical features
X_train[num_features] = scaler.fit_transform(X_train[num_features])

# Transform the test numerical features using the training fit
X_test[num_features] = scaler.transform(X_test[num_features])

# 6. Save preprocessed data
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

train_data.to_csv('Dataset/train_data.csv', index=False)
test_data.to_csv('Dataset/test_data.csv', index=False)

print("Preprocessing complete. Files saved.")