import os
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, EfficientNetB0, MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tempfile

app = Flask(__name__)

# Define anemia threshold constant
ANEMIA_THRESHOLD = 0.7  # 70% probability threshold for anemia


# Function to create model architecture and load weights
def load_cnn_model(base_model_class, weights_path):
    base_model = base_model_class(weights=None, include_top=False, input_shape=(224, 224, 3))
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=base_model.input, outputs=output)
    try:
        # Use absolute path for model weights
        full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), weights_path)
        model.load_weights(full_path)
        return model
    except Exception as e:
        print(f"Error loading {weights_path}: {e}")
        return None


# Load CNN models properly
cnn_models = {}
for model_name, model_class, weight_file in [
    ("resnet", ResNet50, "resnet50.weights.h5"),
    ("efficientnet", EfficientNetB0, "efficientnetb0.weights.h5"),
    ("mobilenet", MobileNetV2, "mobilenetv2.weights.h5")
]:
    loaded_model = load_cnn_model(model_class, weight_file)
    if loaded_model:
        cnn_models[model_name] = loaded_model
    else:
        print(f"Warning: {model_name} model could not be loaded")

# Load ML ensemble model and scaler
ml_ensemble = None
scaler = None
try:
    # Use absolute paths for model and scaler
    model_dir = os.path.dirname(os.path.abspath(__file__))
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    model_path = os.path.join(model_dir, "ensemble_model.pkl")

    scaler = joblib.load(scaler_path)
    ml_ensemble = joblib.load(model_path)
except Exception as e:
    print(f"Error loading ML ensemble model: {e}")
    print("ML ensemble model will not be used for predictions")


def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)
    return img


def cnn_ensemble_predict(img_path):
    if not cnn_models:
        return 0.5  # Return neutral prediction if no models were loaded

    img = preprocess_image(img_path)
    preds = []

    for model in cnn_models.values():
        try:
            preds.append(model.predict(img, verbose=0)[0][0])
        except Exception as e:
            print(f"Error during CNN prediction: {e}")

    if not preds:
        return 0.5  # Return neutral prediction if all predictions failed

    return np.mean(preds)  # Soft voting


def ml_ensemble_predict(features):
    if ml_ensemble is None or scaler is None:
        return 0.5  # Return neutral prediction if model or scaler is not loaded

    try:
        features_scaled = scaler.transform([features])
        return ml_ensemble.predict_proba(features_scaled)[:, 1][0]  # Probability of class 1 (Anemic)
    except Exception as e:
        print(f"Error during ML prediction: {e}")
        return 0.5  # Return neutral prediction on error


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return render_template('index.html', error="Please provide an image file")

        img_file = request.files['image']

        # Get individual form fields
        gender = request.form.get('gender', 'Male')

        # Check if we're using the new individual fields or the old combined field
        if all(param in request.form for param in ['hemoglobin', 'mch', 'mchc', 'mcv']):
            # Using new individual fields
            try:
                hemoglobin = float(request.form['hemoglobin'])
                mch = float(request.form['mch'])
                mchc = float(request.form['mchc'])
                mcv = float(request.form['mcv'])

                # Create features list
                features = [hemoglobin, mcv, mch, mchc]
            except ValueError:
                return render_template('index.html', error="Invalid numerical values in blood parameters")
        else:
            # Fall back to using the combined tabular_data field
            if 'tabular_data' not in request.form:
                return render_template('index.html', error="Please provide blood test parameters")

            csv_data = request.form['tabular_data']

            # Validate tabular data
            try:
                features = list(map(float, csv_data.split(',')))
                if len(features) != 4:
                    return render_template('index.html',
                                           error="Please provide exactly 4 values for blood parameters")
            except ValueError:
                return render_template('index.html',
                                       error="Invalid tabular data format. Please provide valid numbers")

        if img_file.filename == '':
            return render_template('index.html', error="No image file selected")

        temp_file = None
        try:
            # Use a unique temporary file per request to avoid collisions on Windows.
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                temp_file = tmp.name

            # Save temporary image file
            img_file.save(temp_file)

            # Get predictions
            cnn_pred = cnn_ensemble_predict(temp_file)
            ml_pred = ml_ensemble_predict(features) if ml_ensemble is not None else None

            # Calculate final prediction
            if ml_pred is not None:
                final_pred = (cnn_pred + ml_pred) / 2  # Average ensemble of CNN and ML
            else:
                final_pred = cnn_pred  # Use only CNN prediction
        finally:
            # Clean up temp file
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception as e:
                    print(f"Warning: Could not remove temporary file: {e}")

        # Determine anemia status based on the new threshold
        is_anemic = final_pred > ANEMIA_THRESHOLD

        return render_template('index.html',
                               cnn_pred=cnn_pred,
                               ml_pred=ml_pred,
                               final_pred=final_pred)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return render_template('index.html', error=f"An error occurred: {str(e)}")


if __name__ == '__main__':
    app.run(debug=True)
