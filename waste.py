import pickle
import numpy as np

# Load trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Function to predict waste type
def predict_waste(features):
    features = np.array(features).reshape(1, -1)  # Reshape input for model
    prediction = model.predict(features)
    return "Organic" if prediction[0] == 0 else "Recyclable"

# Example test
if __name__ == "__main__":
    test_input = [3, 4]  # Change values as needed
    result = predict_waste(test_input)
    print(f"Prediction: {result}")