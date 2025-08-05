import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import json

# --- 1. Load Data ---
try:
    file_path = '/kaggle/input/salaried-employee-dataset-alticred'
    df = pd.read_csv(file_path)
    print("Adaptability Model: Dataset loaded successfully.")
except FileNotFoundError:
    print(f"Error: The file was not found at {file_path}")
    exit()

# --- 2. Data Cleaning and Feature Engineering ---
feature_cols = [
    'owns_home', 'monthly_rent', 'income-expense ratio', 'emi_status_log',
    'recovery_days', 'default_label', 'user_id', 'monthly_credit_bills', 'mortgage_status'
]
df.dropna(subset=feature_cols, inplace=True)

def count_missed_emis(emi_log_str):
    try:
        return json.loads(emi_log_str.replace("'", "\"")).count(0)
    except: return 5
df['missed_emi_count'] = df['emi_status_log'].apply(count_missed_emis)
mortgage_dummies = pd.get_dummies(df['mortgage_status'], prefix='mortgage')
df = pd.concat([df, mortgage_dummies], axis=1)
df.reset_index(drop=True, inplace=True)
print("Feature engineering complete.")

# --- 3. Model Training and Selection ---
print("\n--- Training and Selecting Best Model ---")
model_features = [
    'owns_home', 'monthly_rent', 'income-expense ratio', 'missed_emi_count',
    'recovery_days', 'monthly_credit_bills'
] + list(mortgage_dummies.columns)

X = df[model_features]
y = df['default_label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Model 1: Logistic Regression
log_reg_model = LogisticRegression(random_state=42, max_iter=1000)
log_reg_model.fit(X_train, y_train)
accuracy_log_reg = accuracy_score(y_test, log_reg_model.predict(X_test))
print(f"Logistic Regression Accuracy: {accuracy_log_reg:.4f}")

# Model 2: Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
accuracy_rf = accuracy_score(y_test, rf_model.predict(X_test))
print(f"Random Forest Classifier Accuracy: {accuracy_rf:.4f}")

if accuracy_rf > accuracy_log_reg:
    best_model = rf_model
    print("\nSelected Random Forest as the final model.")
else:
    best_model = log_reg_model
    print("\nSelected Logistic Regression as the final model.")

# --- 4. Generate Final Adaptability Score ---
all_predictions_proba = best_model.predict_proba(X_scaled)
df['adaptability_score'] = all_predictions_proba[:, 0]
print("Final 'adaptability_score' generated for all users.")

# --- 5. Interactive Prediction ---
print("\n--- Interactive Adaptability Score Calculator ---")
while True:
    try:
        print("\nPlease enter the following details for the user:")
        user_raw = {}
        owns_home_input = int(input("Does the user own a home? (1 for yes, 0 for no): "))
        user_raw['owns_home'] = owns_home_input
        
        if owns_home_input == 1:
            user_raw['monthly_rent'] = 0
        else:
            user_raw['monthly_rent'] = float(input("Enter Monthly Rent: "))

        user_raw['income-expense ratio'] = float(input("Enter Income-Expense Ratio: "))
        user_raw['missed_emi_count'] = int(input("Enter Number of Missed EMIs: "))
        user_raw['recovery_days'] = int(input("Enter typical recovery days: "))
        user_raw['monthly_credit_bills'] = float(input("Enter Monthly Credit Bills: "))
        mortgage_status = input("Enter Mortgage Status (ongoing, paid, none): ").lower()

        user_df_raw = pd.DataFrame([user_raw])
        for col in mortgage_dummies.columns:
            user_df_raw[col] = 1 if col.endswith(mortgage_status) else 0
        
        user_df_ordered = user_df_raw[model_features]
        user_scaled = scaler.transform(user_df_ordered)
        
        user_prediction_proba = best_model.predict_proba(user_scaled)
        final_score = user_prediction_proba[0][0]

        print("\n--- Calculated Adaptability Score ---")
        print(f"Adaptability Score: {final_score:.4f}")
        print("-------------------------------------\n")

    except ValueError:
        print("\nInvalid input. Please enter numbers correctly.\n")
    except Exception as e:
        print(f"\nAn error occurred: {e}\n")

    if input("Calculate for another user? (yes/no): ").lower() != 'yes':
        break
