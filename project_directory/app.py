import pandas as pd
import numpy as np
from flask import Flask, request, render_template
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the pre-trained model and scaler
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
scaler = StandardScaler()
encoder = LabelEncoder()

# Define the URL of the CSV file on GitHub
data_url = 'https://raw.githubusercontent.com/Srikanth854/code_files/main/code%20files/updated_dataset_with_AQI.csv'

# Load the dataset from the GitHub URL
data = pd.read_csv(data_url)

# Define specific columns to use as features
specific_columns = [
    'CO(GT)', 'PT08.S1(CO)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)',
    'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'wspd'
]

# Prepare the data
X = data[specific_columns]  # Use only specific columns as features
y = data['AQI_Category']  # Target

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Fit and transform scaler
X_scaled = scaler.fit_transform(X_imputed)

# Fit the classifier
classifier.fit(X_scaled, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = [request.form.get(feature) for feature in specific_columns]
    user_input = [float(val) if val else np.nan for val in user_input]
    user_input_array = np.array(user_input).reshape(1, -1)
    
    # Impute missing values
    user_input_array_imputed = imputer.transform(user_input_array)
    
    user_input_scaled = scaler.transform(user_input_array_imputed)
    predicted_category = classifier.predict(user_input_scaled)
    return render_template('result.html', prediction=predicted_category[0])

if __name__ == '__main__':
    app.run(debug=True)
