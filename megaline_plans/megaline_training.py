# Libraries and programs needed
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Loading dataset
df = pd.read_csv('https://drive.google.com/uc?export=download&id=1zh5ae-9FMgf_WAJTykTo3CKlfN9bc1HT')

# Adding client column
df['client_id'] = range(len(df))
new_column_order = ['client_id', 'calls', 'minutes', 'messages', 'mb_used', 'is_ultra']
df = df[new_column_order]

# Setting features and target
X = df[['calls', 'minutes', 'messages', 'mb_used']]
y = df['is_ultra']

# Data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# Data scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training
model_train = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model_train.fit(X_train_scaled, y_train)
model_train_score = model_train.score(X_test_scaled, y_test)
y_train_pred = model_train.predict(X_test_scaled)
print(f"Model predictions: {y_train_pred}")
print(f"Model accuracy score: {round(model_train_score, 4)}")

# Testing prediction on a specific client
client_25 = df.loc[df['client_id'] == 25, ['calls', 'minutes', 'messages', 'mb_used']]
client_25_scaled = scaler.transform(client_25)
client_25_pred = model_train.predict(client_25_scaled)
print(f"Client 25 prediction: {client_25_pred}")

# Model for deployment
final_scaler = StandardScaler()
X_scaled = final_scaler.fit_transform(X)
final_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
final_model.fit(X_scaled, y)
model_score = final_model.score(X_scaled, y)
y_pred = final_model.predict(X_scaled)
print(f"Final model predictions: {y_pred}")
print(f"Final model accuracy score: {round(model_score, 4)}")

# Joblib save
try: 
    joblib.dump(final_model, 'megaline_model.joblib')
    joblib.dump(final_scaler, 'megaline_scaler.joblib')
    print("Model and scaler saved successfully!")
except Exception as e:
    print(f'Error saving files: {e}')


# Creating modified datasets for the website
# January dataset
df_january = df.copy(True)
df_january = df_january.drop('is_ultra', axis=1)

# February dataset
df_february = df.copy(True)
df_february = df_february.drop('is_ultra', axis=1)

# Calls
variation_calls_minutes_feb = np.random.normal(0, 0.1, len(df))
df_february['calls'] = df['calls'] * (1 + variation_calls_minutes_feb)
df_february['calls'] = df_february['calls'].clip(lower=0, upper=200)
df_february['calls'] = df_february['calls'].round()

# Minutes
df_february['minutes'] = df['minutes'] * (1 + variation_calls_minutes_feb * 1.2)
df_february['minutes'] = df_february['minutes'].clip(lower=0, upper=1500)

# Messages
variation_messages_feb = np.random.normal(0, 0.08, len(df))
df_february['messages'] = df['messages'] * (1 + variation_messages_feb)
df_february['messages'] = df_february['messages'].clip(lower=0, upper=300)
df_february['messages'] = df_february['messages'].round()

# MB used
variation_mb_feb = np.random.normal(0, 0.09, len(df))
df_february['mb_used'] = df['mb_used'] * (1 + variation_mb_feb)
df_february['mb_used'] = df_february['mb_used'].clip(lower=0, upper=50000)

# March dataset
df_march = df.copy(True)
df_march = df_march.drop('is_ultra', axis=1)

# Calls
variation_calls_minutes_mar = np.random.normal(0, 0.15, len(df))
df_march['calls'] = df['calls'] * (1 + variation_calls_minutes_mar)
df_march['calls'] = df_march['calls'].clip(lower=0, upper=200)
df_march['calls'] = df_march['calls'].round()

# Minutes
df_march['minutes'] = df['minutes'] * (1 + variation_calls_minutes_mar * 1.2)
df_march['minutes'] = df_march['minutes'].clip(lower=0, upper=1500)

# Messages
variation_messages_mar = np.random.normal(0, 0.12, len(df))
df_march['messages'] = df['messages'] * (1 + variation_messages_mar)
df_march['messages'] = df_march['messages'].clip(lower=0, upper=300)
df_march['messages'] = df_march['messages'].round()

# MB used
variation_mb_mar = np.random.normal(0, 0.13, len(df))
df_march['mb_used'] = df['mb_used'] * (1 + variation_mb_mar)
df_march['mb_used'] = df_march['mb_used'].clip(lower=0, upper=50000)

# Saving datasets
df_january.to_csv('megaline_january.csv', index=False)
df_february.to_csv('megaline_february.csv', index=False) 
df_march.to_csv('megaline_march.csv', index=False)
