import sys
sys.stdout.reconfigure(encoding='utf-8')

print("✅ Model loaded successfully!")
import pickle
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Sample training data (Replace with actual dataset)
X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y_train = np.array([0, 1, 0, 1])  # 0 = Organic, 1 = Recyclable

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("✅ Model trained and saved as model.pkl")