import joblib
import pandas as pd
import os


def predict_blood_pressure(age, body_temp, heart_rate):
    """
    Predicts BP Category based on health metrics using a model stored in a subfolder.
    """
    # 1. Define the path to your model file
    # This points to blood_pressure_model/blood_pressure_model.pkl
    model_path = os.path.join('blood_pressure_model', 'blood_pressure_model.pkl')

    try:
        # 2. Load the saved model
        if not os.path.exists(model_path):
            return f"Error: Model file not found at {model_path}"

        model = joblib.load(model_path)

        # 3. Prepare the input data
        data = {
            'Age': [age],
            'BodyTemp': [body_temp],
            'HeartRate': [heart_rate]
        }
        input_df = pd.DataFrame(data)

        # 4. Perform prediction
        prediction_code = model.predict(input_df)[0]

        # 5. Map the numeric output to medical labels
        categories = {
            0: "Low",
            1: "Normal",
            2: "Elevated",
            3: "Stage 1 Hypertension",
            4: "Stage 2 Hypertension"
        }

        return categories.get(prediction_code, "Unknown Category")

    except Exception as e:
        return f"Error during prediction: {str(e)}"


if __name__ == "__main__":
    # Test with sample data
    print("Testing local model...")
    result = predict_blood_pressure(age=24, body_temp=98.78, heart_rate=74)
    print(f"Prediction: {result}")