import joblib

scaler = joblib.load('scaler')
print(scaler.__dict__)