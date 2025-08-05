import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack
import re

# --- 1. Load Data ---
try:
    # Using the new dataset provided by the user
    file_path = '/kaggle/input/salaried-employee-dataset-alticred'
    df = pd.read_csv(file_path)
    print("Language Sentiment Model: Dataset loaded successfully.")
except FileNotFoundError:
    print(f"Error: The file was not found at {file_path}")
    exit()

# --- 2. Data Cleaning and Preprocessing ---
feature_cols = ['user_posts', 'sentiment_score', 'default_label']
df.dropna(subset=feature_cols, inplace=True)
df = df.reset_index(drop=True)

# Simple text cleaning function
def clean_text(text):
    text = text.lower() # Lowercase
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    return text

df['cleaned_posts'] = df['user_posts'].apply(clean_text)
print("Text data cleaned.")

# --- 3. Feature Engineering (TF-IDF Vectorizer) ---
# This converts the raw text of user posts into a matrix of numerical features.
# It captures the importance of each word in the context of all posts.
vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
X_text = vectorizer.fit_transform(df['cleaned_posts'])
print("User posts converted to numerical features using TF-IDF.")

# --- 4. Combine Text Features with Sentiment Score ---
# We combine the TF-IDF features with the existing sentiment_score column.
# Note: sentiment_score needs to be reshaped to be combined.
X_sentiment = df['sentiment_score'].values.reshape(-1, 1)

# Horizontally stack the sparse text matrix and the dense sentiment score array
X_combined = hstack([X_text, X_sentiment])
y = df['default_label']
print("Text features and sentiment scores have been combined.")

# --- 5. Train Logistic Regression Model ---
print("\n--- Training Predictive Model ---")
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42, stratify=y)

# Logistic Regression is a strong baseline for text classification tasks
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# Evaluate the model's performance
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# --- 6. Generate Final Language Sentiment Score ---
# The score is the probability of NOT defaulting (class 0)
all_predictions_proba = model.predict_proba(X_combined)
df['language_sentiment_score'] = all_predictions_proba[:, 0]
print("\nFinal 'language_sentiment_score' generated for all users.")
print("A score of 1.0 means highest trustworthiness (least likely to default).")

# --- 7. Interactive Prediction ---
print("\n--- Interactive Language Sentiment Score Calculator ---")
while True:
    try:
        print("\nPlease enter a user's social media post to calculate their score:")
        user_post = input("Enter post text: ")
        
        if not user_post.strip():
            print("Post cannot be empty.")
            continue

        # For a single post, we can't calculate a meaningful pre-set sentiment score,
        # so we'll use a neutral value (0.0) as a placeholder.
        # A more advanced model could use a sentiment library here.
        user_sentiment_score = 0.0

        # Apply the same transformations
        cleaned_post = clean_text(user_post)
        user_text_features = vectorizer.transform([cleaned_post])
        user_sentiment_feature = np.array([user_sentiment_score]).reshape(-1, 1)
        
        user_combined_features = hstack([user_text_features, user_sentiment_feature])

        # Predict the probability of not defaulting
        user_prediction_proba = model.predict_proba(user_combined_features)
        final_score = user_prediction_proba[0][0] # Get probability for class 0

        print("\n--- Calculated Language Sentiment Score ---")
        print(f"Language Sentiment Score: {final_score:.4f}")
        print("-------------------------------------------\n")

    except ValueError:
        print("\nInvalid input. Please try again.\n")
    except Exception as e:
        print(f"\nAn error occurred: {e}\n")

    if input("Calculate for another post? (yes/no): ").lower() != 'yes':
        print("Exiting.")
        break