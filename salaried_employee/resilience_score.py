import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import json

# --- 1. Load Data ---
try:
    file_path = '/kaggle/input/salaried-employee-dataset-alticred'
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print(f"Error: The file was not found at {file_path}")
    exit()
except Exception as e:
    print(f"An error occurred while reading the file: {e}")
    exit()

# --- 2. Data Cleaning and Preprocessing ---
feature_cols = [
    'monthly_credit_bills', 'bnpl_utilization_rate', 'mortgage_months_left',
    'income-expense ratio', 'upi_balances', 'emi_status_log', 'user_id', 'default_label'
]
df.dropna(subset=feature_cols, inplace=True)
df = df.reset_index(drop=True)
print(f"Dataset shape after cleaning: {df.shape}")


# --- 3. Feature Engineering ---
def get_avg_upi(balance_str):
    try:
        balances = json.loads(balance_str.replace("'", "\""))
        return np.mean([float(b) for b in balances]) if balances else 0
    except (json.JSONDecodeError, TypeError): return 0
df['avg_upi_balance'] = df['upi_balances'].apply(get_avg_upi)

def count_missed_emis(emi_log_str):
    try:
        emis = json.loads(emi_log_str)
        return emis.count(0)
    except (json.JSONDecodeError, TypeError): return 5
df['missed_emi_count'] = df['emi_status_log'].apply(count_missed_emis)
print("\nEngineered features 'avg_upi_balance' and 'missed_emi_count'.")


# --- 4. Normalization ---
scaler = MinMaxScaler()
features_to_normalize = [
    'monthly_credit_bills', 'bnpl_utilization_rate', 'mortgage_months_left',
    'avg_upi_balance', 'income-expense ratio', 'missed_emi_count'
]
df_normalized = df.copy()
df_normalized[features_to_normalize] = scaler.fit_transform(df[features_to_normalize])
print("\nAll features normalized to a 0-1 scale.")


# --- 5. Resilience Score Calculation (Rule-Based) ---
INTEREST_RATES = {'credit': 0.20, 'bnpl': 0.15, 'mortgage': 0.08}

def calculate_resilience_score(row, raw_row):
    norm_credit = row['monthly_credit_bills']
    norm_bnpl = row['bnpl_utilization_rate']
    norm_mortgage = row['mortgage_months_left']
    norm_income_ratio = row['income-expense ratio']
    norm_missed_emis = row['missed_emi_count']
    norm_upi = row['avg_upi_balance']

    risk_val_credit = raw_row['monthly_credit_bills'] * INTEREST_RATES['credit']
    risk_val_bnpl = (raw_row['monthly_credit_bills'] * raw_row['bnpl_utilization_rate']) * INTEREST_RATES['bnpl']
    risk_val_mortgage = raw_row['mortgage_months_left'] * INTEREST_RATES['mortgage']
    total_risk_value = risk_val_credit + risk_val_bnpl + risk_val_mortgage

    financial_stress_score = 0
    if total_risk_value > 0:
        w_credit = risk_val_credit / total_risk_value
        w_bnpl = risk_val_bnpl / total_risk_value
        w_mortgage = risk_val_mortgage / total_risk_value
        financial_stress_score = (w_credit * norm_credit + w_bnpl * norm_bnpl + w_mortgage * norm_mortgage)

    w_income_health = 0.15 / 0.75; w_emi_health = 0.10 / 0.75; w_upi_health = 0.50 / 0.75
    score_income = norm_income_ratio; score_emi = 1 - norm_missed_emis; score_upi = norm_upi
    financial_health_score = (w_income_health * score_income + w_emi_health * score_emi + w_upi_health * score_upi)
    
    final_score = (0.5 * financial_health_score) + (0.5 * (1 - financial_stress_score))
    return final_score

df['resilience_score'] = [calculate_resilience_score(df_normalized.iloc[i], df.iloc[i]) for i in range(len(df))]
print("\nRule-based 'resilience_score' calculated for all users.")


# --- 6. Default Prediction using Ridge Regression ---
print("\n--- Training Predictive Model ---")

df_normalized['resilience_score'] = df['resilience_score']
ml_features = features_to_normalize + ['resilience_score']
X = df_normalized[ml_features]
y = df['default_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use Ridge regression model
model = Ridge(alpha=1.0, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model's performance using R-squared
y_pred = model.predict(X_test)
print(f"Model R-squared: {r2_score(y_test, y_pred):.4f}")

# --- 7. Generate Predictor Score ---
# Use the model's predictions to get a continuous score
predictions = model.predict(X)

# Scale the predictions to a 0-1 range
score_scaler = MinMaxScaler()
scaled_scores = score_scaler.fit_transform(predictions.reshape(-1, 1))
df['predictor_score'] = 1 - scaled_scores

print("\n'predictor_score' generated for all users.")


# --- 8. Display Final Results ---
print("\n--- Final Combined Scores ---")
display_cols = ['user_id', 'resilience_score', 'predictor_score', 'default_label']
print("\nTop 10 Users by Predictor Score (Least Likely to Default):")
print(df[display_cols].sort_values(by='predictor_score', ascending=False).head(10).round(4))
print("\nTop 10 Users by Resilience Score:")
print(df[display_cols].sort_values(by='resilience_score', ascending=False).head(10).round(4))

# --- 9. Interactive Prediction Section ---
print("\n--- Interactive Score Calculator ---")
while True:
    try:
        print("Please enter the following details for the user:")
        user_raw = {}
        user_raw['monthly_credit_bills'] = float(input("Enter Monthly Credit Bills: "))
        user_raw['bnpl_utilization_rate'] = float(input("Enter BNPL Utilization Rate: "))
        user_raw['mortgage_months_left'] = float(input("Enter Mortgage Months Left: "))
        user_raw['avg_upi_balance'] = float(input("Enter Average UPI Balance: "))
        user_raw['income-expense ratio'] = float(input("Enter Income-Expense Ratio: "))
        user_raw['missed_emi_count'] = int(input("Enter Number of Missed EMIs in log: "))

        user_df_raw = pd.DataFrame([user_raw])
        user_df_normalized = pd.DataFrame(scaler.transform(user_df_raw[features_to_normalize]), columns=features_to_normalize)
        user_df_normalized.clip(0, 1, inplace=True)

        user_resilience_score = calculate_resilience_score(user_df_normalized.iloc[0], user_df_raw.iloc[0])
        user_df_normalized['resilience_score'] = user_resilience_score
        
        user_prediction = model.predict(user_df_normalized[ml_features])
        user_scaled_score = score_scaler.transform(user_prediction.reshape(-1, 1))
        user_scaled_score = np.clip(user_scaled_score, 0, 1)
        user_predictor_score = 1 - user_scaled_score[0][0]

        print("\n--- Calculated Scores for User ---")
        print(f"Resilience Score: {user_resilience_score:.4f}")
        print(f"Predictor Score (Likelihood of Not Defaulting): {user_predictor_score:.4f}")
        print("----------------------------------\n")

    except ValueError:
        print("\nInvalid input. Please ensure you enter numbers only.\n")
    except Exception as e:
        print(f"\nAn error occurred: {e}\n")

    another = input("Do you want to calculate scores for another user? (yes/no): ")
    if another.lower() != 'yes':
        print("Exiting interactive calculator.")
        break
