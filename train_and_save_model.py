import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Example URL data - replace with actual data
data = {
    'url': [
        'http://example.com',
        'http://phishingsite.com',
        'http://legitimatesite.com',
        'http://malicious.com'
    ],
    'label': [0, 1, 0, 1]  # 0: legitimate, 1: phishing
}
df = pd.DataFrame(data)

# Feature extraction (dummy example, replace with actual feature extraction)
def extract_features(url):
    return np.array([len(url), url.count('.'), url.count('/'), url.count('-')])  # Simple features

df['features'] = df['url'].apply(extract_features)

# Prepare dataset
X = np.vstack(df['features'])
y = df['label'].values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a simple neural network model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Save the model and scaler
model.save('phishing_detection_model.h5')
import joblib
joblib.dump(scaler, 'scaler.pkl')

print("Model and scaler trained and saved.")
