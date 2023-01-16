import joblib

# Predicts the outcome

def startup(pregnancy, glucose, bp, skin, insulin, bmi, dpf, age):
    mod1=joblib.load('diabetesmodel.joblib')
    outcome = mod1.predict([[pregnancy, glucose, bp, skin, insulin, bmi, dpf, age]])

    return outcome[0]