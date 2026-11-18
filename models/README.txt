NETWORK ATTACK DETECTION - MODEL PACKAGE

Model: XGBoost
Imbalance Handling: SMOTE
Input Features: 78
Output Classes: 5

FILES INCLUDED
--------------
1. final_classifier.joblib
   Trained XGBoost classification model.

2. label_encoder.joblib
   Converts model predictions into attack labels.

3. feature_columns.json
   Contains the expected input feature names and order.

4. class_labels.json
   Contains the supported attack categories.


USAGE
-----
Load the model and encoder using joblib:

    import joblib
    import json

    model = joblib.load("final_classifier.joblib")
    encoder = joblib.load("label_encoder.joblib")

    with open("feature_columns.json") as f:
        features = json.load(f)

    predictions = model.predict(incoming_data[features])
    labels = encoder.inverse_transform(predictions.astype(int))


SUPPORTED CLASSES
-----------------
Normal
DoS
Port Scan
Brute Force
Web Attack


IMPORTANT
---------
- Input must contain all 78 expected numerical features.
- Feature names, calculations, units and ordering must
  match the original training dataset.
- Live packet capture and feature extraction are not
  included in this package.
- Compatible Python and library versions are required.
- No model retraining is required for inference.

EVALUATION
----------
Accuracy: 99.8981%
Macro F1: 99.8579%

These scores represent evaluation on the held-out
test dataset and are not guaranteed live performance.