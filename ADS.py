import pandas as pd
import warnings
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")


# takes a DataFrame containing all of the data
# returns the DataFrame transformed using the ADS' process
def preprocess(data):
    le = LabelEncoder()
    le.fit(data.sex.drop_duplicates())
    data.sex = le.transform(data.sex)

    le.fit(data.smoker.drop_duplicates())
    data.smoker = le.transform(data.smoker)

    le.fit(data.region.drop_duplicates())
    data.region = le.transform(data.region)

    return data

# takes a DataFrame containing the training data
# returns a fitted sklearn LinearRegression object
def simpleLR(data):
    X = data.drop(["charges"], axis=1)
    y = data.charges

    lr = LinearRegression().fit(X, y)
    return lr

def polynomialLR(data):
    X = data.drop(["charges", "region"], axis=1)
    y = data.charges

    quad = PolynomialFeatures(degree=2)
    X_quad = quad.fit_transform(X)

    plr = LinearRegression().fit(X_quad, y)
    return plr

def forest(data):
    X = data.drop(["charges"], axis=1)
    y = data.charges

    forest = RandomForestRegressor(n_estimators=100, criterion="mse", random_state=1, n_jobs=1)
    forest.fit(X, y)
    return forest
