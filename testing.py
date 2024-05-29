import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model
loaded_model = joblib.load('random_forest_model.pkl')

# New observation user should type manually
new_observation = [[647, 40, 3, 85000.45, 2, 0, 0, 92012.45, 0, 1, 1]]

# Create the scaler object
scaler = StandardScaler()

# Scale the new observation using the loaded scaler
scaled_observation = scaler.fit_transform(new_observation)

# Predict churn for the new observation
predicted_churn = loaded_model.predict(scaled_observation)

# Print the prediction
if predicted_churn[0] == 1:
    print("The model predicts that the user will churn.")
else:
    print("The model predicts that the user will not churn.")