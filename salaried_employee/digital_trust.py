import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

# --- 1. Load Data ---
try:
    # Corrected file path to point directly to the CSV file within the directory
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
# Drop rows with missing values in the target or key feature columns
key_cols = ['defaulter_neighbors', 'verified_neighbors', 'connections', 'default_label']
df.dropna(subset=key_cols, inplace=True)
df = df.reset_index(drop=True) # Reset index after dropping rows
print(f"Dataset shape after dropping NaNs: {df.shape}")

# --- 3. Feature Engineering ---
# Create 'num_connections' from the 'connections' string
def count_connections(connection_string):
    if isinstance(connection_string, str) and connection_string.strip():
        return len(connection_string.split(','))
    return 0
df['num_connections'] = df['connections'].apply(count_connections)
print("\nCreated 'num_connections' feature.")

# --- 4. Define Features (X) and Target (y) ---
features = ['defaulter_neighbors', 'verified_neighbors', 'num_connections']
target = 'default_label'
X = df[features]
y = df[target]

# --- 5. Feature Scaling ---
# Scale features to have zero mean and unit variance. This is crucial for the autoencoder.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\nFeatures scaled using StandardScaler.")

# --- 6. Autoencoder for Feature Representation Learning ---
# We build a simple autoencoder to learn a compressed representation of the data.
# This can help the classification models by providing them with more abstract features.
input_dim = X_scaled.shape[1]
encoding_dim = 2 # Compress 3 features down to 2

# Encoder
input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation='relu')(input_layer)

# Decoder
decoder = Dense(input_dim, activation='sigmoid')(encoder)

# Autoencoder model
autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')

# Train the autoencoder
autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=16, shuffle=True, verbose=0)
print("\nAutoencoder training complete.")

# Use the encoder part of the autoencoder to transform our data
encoder_model = Model(inputs=input_layer, outputs=encoder)
X_encoded = encoder_model.predict(X_scaled)
print("Original features transformed into learned representations by the encoder.")

# --- 7. Split Data for Classification Models ---
# We use the new, encoded features for training our classifiers
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)

# --- 8. Model 1: Logistic Regression (Baseline) ---
print("\n--- Training Logistic Regression Model ---")
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)
lr_y_pred = lr_model.predict(X_test)
print("Logistic Regression - Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, lr_y_pred):.4f}")
print(classification_report(y_test, lr_y_pred))

# --- 9. Model 2: XGBoost for Final Prediction ---
print("\n--- Training XGBoost Model ---")
xgb_model = xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)
xgb_y_pred = xgb_model.predict(X_test)
print("XGBoost - Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, xgb_y_pred):.4f}")
print(classification_report(y_test, xgb_y_pred))

# --- 10. Generate Final "Digital Trust Score" using XGBoost ---
# The score is the probability of NOT defaulting (class 0).
# We use the more powerful XGBoost model for the final score.
digital_trust_score = xgb_model.predict_proba(X_encoded)[:, 0]
df['digital_trust_score'] = digital_trust_score
print("\n--- Digital Trust Score Generation (Zhima-Inspired) ---")
print("Successfully generated the Digital Trust Score for each user using the XGBoost model.")

# --- 11. Interactive Prediction ---
print("\n--- Test the Model with Your Own Data ---")
while True:
    try:
        # Get user input
        def_neighbors = int(input("Enter number of defaulter neighbors: "))
        ver_neighbors = int(input("Enter number of verified neighbors: "))
        num_connections = int(input("Enter total number of connections: "))

        # Create a DataFrame from the input
        user_data = pd.DataFrame(
            [[def_neighbors, ver_neighbors, num_connections]],
            columns=features
        )

        # Apply the same transformations as the training data
        user_scaled = scaler.transform(user_data)
        user_encoded = encoder_model.predict(user_scaled)

        # Make predictions
        prediction_label = xgb_model.predict(user_encoded)[0]
        prediction_score_prob = xgb_model.predict_proba(user_encoded)[:, 0][0]

        # Display the results
        print("\n--- Prediction Result ---")
        print(f"Digital Trust Score: {prediction_score_prob:.4f}")
        if prediction_label == 0:
            print("Predicted Outcome: Not a defaulter")
        else:
            print("Predicted Outcome: Likely to be a defaulter")
        print("-------------------------\n")

    except ValueError:
        print("Invalid input. Please enter whole numbers only.")
    except Exception as e:
        print(f"An error occurred: {e}")

    # Ask to continue
    another = input("Do you want to make another prediction? (yes/no): ")
    if another.lower() != 'yes':
        print("Exiting interactive prediction.")
        break