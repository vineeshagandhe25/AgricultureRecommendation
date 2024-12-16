from flask import Flask, render_template, request
import joblib
import numpy as np 
app = Flask(__name__)

model = joblib.load('models/fertilizer_model.pkl')
label_encoders = joblib.load('models/label_encoders.pkl')

# Routes
@app.route('/')
def home():
    return render_template('home.html')  

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/inputs')
def inputs():
    return render_template('inputs.html')  

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        moisture = float(request.form['moisture'])
        soil_type = request.form['soil_type']
        crop_type = request.form['crop_type']
        nitrogen = int(request.form['nitrogen'])
        potassium = int(request.form['potassium'])
        phosphorous = int(request.form['phosphorous'])

        # Encode categorical variables
        encoded_soil_type = label_encoders['Soil_Type'].transform([soil_type])[0]
        encoded_crop_type = label_encoders['Crop_Type'].transform([crop_type])[0]

        # Prepare the input for the model
        input_features = np.array([[
            temperature, humidity, moisture,
            encoded_soil_type, encoded_crop_type,
            nitrogen, potassium, phosphorous
        ]])

        # Make prediction
        prediction = model.predict(input_features)

        # Decode the predicted fertilizer name
        fertilizer_name = label_encoders['Fertilizer'].inverse_transform(prediction)[0]

        # Redirect to result page with the prediction
        return render_template('result.html', result=fertilizer_name)

    except Exception as e:
        return render_template('result.html', result=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
