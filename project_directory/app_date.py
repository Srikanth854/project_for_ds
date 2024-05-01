from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = Flask(__name__)

# Load the dataset
data_url = 'https://raw.githubusercontent.com/Srikanth854/code_files/main/code%20files/updated_dataset_with_AQI.csv'
data = pd.read_csv(data_url)

# Convert 'Date & Time' to datetime
data['Date & Time'] = pd.to_datetime(data['Date & Time'])
data['Year'] = data['Date & Time'].dt.year
data['Month'] = data['Date & Time'].dt.month
data['Day'] = data['Date & Time'].dt.day
data['Hour'] = data['Date & Time'].dt.hour

# Assuming we create a time trend feature (e.g., years since 2004)
data['Time Trend'] = data['Year'] - 2004

# Prepare the data
X = data[['Year', 'Month', 'Day', 'Hour', 'Time Trend']]  # Using time-related features
y = data['AQI_Category']  # Target

# Encode categorical target variable
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Creating and training the model for predicting air quality by date
classifier_date = RandomForestClassifier(n_estimators=100, random_state=42)
classifier_date.fit(X_scaled, y_encoded)

@app.route('/')
def index_date():
    return render_template('index_date.html')

@app.route('/predict_date', methods=['POST'])
def predict_date():
    if request.method == 'POST':
        # Get input date from the form
        input_date = request.form['date']
        input_datetime = datetime.strptime(input_date, "%m/%d/%Y")
        input_features = np.array([[input_datetime.year, input_datetime.month, input_datetime.day, input_datetime.hour, input_datetime.year - 2004]])
        input_features_scaled = scaler.transform(input_features)
        
        # Predict the AQI category
        predicted_category_index = classifier_date.predict(input_features_scaled)
        predicted_category = encoder.inverse_transform(predicted_category_index)[0]

        return render_template('result_date.html', prediction=predicted_category)

if __name__ == '__main__':
    app.run(debug=True)
