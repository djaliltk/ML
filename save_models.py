import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

# Load and prepare the data
df = pd.read_csv('creditcard.csv')

# Define the feature columns in the correct order
feature_columns = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                  'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                  'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

# Prepare features and target
X = df[feature_columns].copy()
y = df.Class

# Create and fit the scaler
scaler = StandardScaler()
X['Amount'] = scaler.fit_transform(X['Amount'].values.reshape(-1,1))

# Train models
rf_model = RandomForestClassifier(n_estimators=100, random_state=123)
dt_model = DecisionTreeClassifier(random_state=123)
ab_model = AdaBoostClassifier(n_estimators=100, random_state=123)
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=123)

# Fit models
rf_model.fit(X, y)
dt_model.fit(X, y)
ab_model.fit(X, y)
gb_model.fit(X, y)

# Save models
joblib.dump(rf_model, 'random_forest_model.joblib')
joblib.dump(dt_model, 'decision_tree_model.joblib')
joblib.dump(ab_model, 'ada_boost_model.joblib')
joblib.dump(gb_model, 'gradient_boosting_model.joblib')
joblib.dump(scaler, 'scaler.joblib')

print("Models saved successfully!") 