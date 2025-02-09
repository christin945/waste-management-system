from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/")
def home():
    return render_template("waste.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = np.array([[float(data["feature1"]), float(data["feature2"])]])
    prediction = model.predict(features)
    result = "Organic" if prediction[0] == 0 else "Recyclable"
    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(debug=True)