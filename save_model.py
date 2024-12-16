import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the CSV file
data = pd.read_csv('crop_fertilizers.csv')

# Encode categorical features
label_encoders = {}
for column in ['Soil_Type', 'Crop_Type']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le  # Store the encoders for later use

# Define features and target
X = data[['Temparature', 'Humidity', 'Moisture', 'Soil_Type', 'Crop_Type', 'Nitrogen', 'Potassium', 'Phosphorous']]
y = data['Fertilizer']

# Encode the target column
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)
label_encoders['Fertilizer'] = target_encoder  # Store target encoder

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the model and encoders
joblib.dump(model, 'fertilizer_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')

print("Model and encoders saved successfully.")
