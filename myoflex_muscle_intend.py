import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'my_arm_data.csv')

# 2. Load and verify data
try:
    # Use 'on_bad_lines' to skip any remaining boot-up text lines
    df = pd.read_csv(csv_path, on_bad_lines='skip')
    print("Data loaded successfully.")
    print(df.head()) # Verify the first 5 rows
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit()

# 3. Define Features and Labels
# Features: First 4 columns (EMG and 3 Flex sensors)
X = df[['emg', 'flex1', 'flex2', 'flex3']]
# Target: The 5th column (Label: 0, 1, 2, or 3)
y = df['label']

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 5. Train Random Forest Model
# This is fast and reliable for 5-DOF control
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 6. Save the Model
model_path = os.path.join(script_dir, 'myoflex_model.pkl')
joblib.dump(model, model_path)
print(f"Model saved to: {model_path}")
