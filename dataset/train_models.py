import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Load dataset
df = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")
df["LoanAmount"].fillna(df["LoanAmount"].median(), inplace=True)
df["Credit_History"].fillna(df["Credit_History"].mode()[0], inplace=True)
df["Self_Employed"].fillna("No", inplace=True)

# Encode target
le = LabelEncoder()
df["Loan_Status"] = le.fit_transform(df["Loan_Status"])

# Encode features
df["Employment"] = df["Self_Employed"].map({"No": 1, "Yes": 0})
df["Property_Area_Enc"] = df["Property_Area"].map({"Rural": 0, "Semiurban": 1, "Urban": 2})

X = df[["ApplicantIncome", "CoapplicantIncome", "LoanAmount", 
        "Credit_History", "Employment", "Property_Area_Enc"]]
y = df["Loan_Status"]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train SVM models
model_linear = SVC(kernel="linear", probability=True).fit(X_scaled, y)
model_poly   = SVC(kernel="poly", degree=3, probability=True).fit(X_scaled, y)
model_rbf    = SVC(kernel="rbf", probability=True).fit(X_scaled, y)

# Save models and scaler
joblib.dump(model_linear, "model_linear.pkl")
joblib.dump(model_poly, "model_poly.pkl")
joblib.dump(model_rbf, "model_rbf.pkl")
joblib.dump(scaler, "scaler.pkl")
