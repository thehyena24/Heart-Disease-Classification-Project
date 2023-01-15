from joblib import Parallel, delayed
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("heart-disease.csv")

'''
Split data into X and Y
X - The entire data set excluding the target variable. It is to be processed by the ML model to try and figure out a way to get the target variable.
Y - Only consists of the target variable. The ML model should find a way to arrive at Y using X.
'''

X = df.drop("target", axis = 1)
Y = df.target

# Now, we further split these into training and test sets.

np.random.seed(42)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,               
                                                    test_size = 0.2)    # This means that 20% of the data will be used as the test set
                                                    
                                                    
np.random.seed(42)           #To make the results reproducable
model = LogisticRegression()
model.fit(X_train, Y_train)

print(model.score(X_test, Y_test))

joblib.dump(model, 'model.pkl')