
import pickle
from sklearn.ensemble import RandomForestClassifier  

# Example training data  
X_train = [[1, 2], [3, 4], [5, 6]]  
y_train = [0, 1, 0]  

# Train model  
model = RandomForestClassifier()  
model.fit(X_train, y_train)  

# Save model as .pkl  
with open('model.pkl', 'wb') as f:  
    pickle.dump(model, f)  

print("Model saved as model.pkl")