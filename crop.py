# Crop Recommendation System

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset 
data = pd.read_csv('data/Crop_recommendation.csv')

# Explore the dataset
print(data.head())

# Separate features and labels
X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]  # Features (replace with your relevant columns)
y = data['label']  # Target labels (crops)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

joblib.dump(rf_model, 'crop_model.pkl')

# Predict on the test data
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Function to predict the crop based on new input values
def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    input_data = [[N, P, K, temperature, humidity, ph, rainfall]]
    prediction = rf_model.predict(input_data)
    return prediction[0]

# Example of predicting a crop
predicted_crop = predict_crop(90, 42, 43, 20.87, 82.0, 6.5, 202.9)
print(f"Recommended Crop: {predicted_crop}")
