from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

app = Flask(__name__)

# Load the trained model and encoders once
model = joblib.load('model/Obesit_Risk_Using_Behavioral_and_Dietary_Pattern.joblib')
one_hot_encoder = joblib.load('model/one_hot_encoder.joblib')
label_encoders = joblib.load('model/label_encoders.joblib')

# Define preprocessing function
def preprocess_input(data):
    # Convert input data to a DataFrame
    df = pd.DataFrame(data, index=[0])

    # Label Encoding for categorical features
    label_encode_features = ['Gender', 'family_history_with_overweight']
    for column in label_encode_features:
        if column in df.columns:
            le = label_encoders[column]
            known_classes = set(le.classes_)
            df[column] = df[column].apply(lambda x: le.transform([x])[0] if x in known_classes else -1)  # Using -1 for unknown

    # One-Hot Encoding for categorical features
    one_hot_features = ['CALC', 'FAVC', 'SMOKE', 'CAEC']
    if all(feature in df.columns for feature in one_hot_features):
        # Use the handle_unknown='ignore' parameter to ignore unknown categories
        one_hot_encoded = one_hot_encoder.transform(df[one_hot_features])
        one_hot_encoded_df = pd.DataFrame(one_hot_encoded, columns=one_hot_encoder.get_feature_names_out(one_hot_features))

        # Drop the original one-hot encoded columns and concatenate the new ones
        df = df.drop(columns=one_hot_features)
        
        # Ensure the concatenation aligns properly
        df = pd.concat([df.reset_index(drop=True), one_hot_encoded_df.reset_index(drop=True)], axis=1)

    return df

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()

        # Check for required fields
        required_fields = ['Age', 'Gender', 'CALC', 'FAVC', 'FCVC', 'NCP', 'SMOKE', 'CH2O',
       'family_history_with_overweight', 'FAF', 'CAEC']
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return render_template('index.html', prediction=f"Missing required fields: {', '.join(missing_fields)}")

        preprocessed_data = preprocess_input(data)
        prediction = model.predict(preprocessed_data)
        prediction_result = prediction[0]

        if prediction_result == 0:
            prediction_result ="Insufficient Weight"
        elif prediction_result == 1:
            prediction_result="Normal Weight"
        elif prediction_result == 2:
            prediction_result="Obesity Type I"
        elif prediction_result == 3:
            prediction_result="Obesity Type II"
        elif prediction_result == 4:
            prediction_result="Obesity Type III"
        elif prediction_result == 5:
            prediction_result="Overweight Level I"
        elif prediction_result == 6:
            prediction_result="Overweight Level II"
        else:
            prediction_result="Unknown label"

        return render_template('index.html', prediction=f'Your are likely to be: {prediction_result}')
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
