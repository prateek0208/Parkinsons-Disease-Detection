import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv('../parkinsons.data')

# Prepare features and labels exactly as in the notebook
df.drop(columns=['name'], inplace=True)
X = df.drop(columns=['status'])
y = df['status']

# Train Random Forest (identical to the notebook)
model = RandomForestClassifier(random_state=45)
model.fit(X, y)

# Save the model
with open('rf_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and exported successfully!")
