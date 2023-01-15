from joblib import Parallel, delayed
import joblib
import pandas as pd

def predictHeartDisease(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):

    model = joblib.load('model.pkl')

    input = {'age':age, 'sex':sex, 'cp':cp, 'trestbps':trestbps, 'chol':chol, 'fbs':fbs, 'restecg':restecg, 'thalach':thalach, 'exang':exang, 'oldpeak':oldpeak, 'slope':slope, 'ca':ca, 'thal':thal}
    df = pd.DataFrame(input, index=[0])

    return model.predict(df)