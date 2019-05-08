import pandas as pd
import warnings
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

# See Accuracy_Analysis.ipynb for examples how to use all these!


# takes a DataFrame containing all of the data
# returns the DataFrame transformed using the ADS' label encoding process
def encode(data):
    le = LabelEncoder()
    le.fit(data.sex.drop_duplicates())
    data.sex = le.transform(data.sex)

    le.fit(data.smoker.drop_duplicates())
    data.smoker = le.transform(data.smoker)

    le.fit(data.region.drop_duplicates())
    data.region = le.transform(data.region)

    return data


# takes a DataFrame containing either training or test data
# returns the DataFrame's features and labels prepared to input directly into a fit()/predict()/score()/etc. function
def prepareLR(data):
    X = data.drop(["charges"], axis=1)
    y = data.charges
    return X, y

# takes a DataFrame containing the ENCODED training data
# returns a fitted sklearn LinearRegression object
def simpleLR(data):
    X, y = prepareLR(data)

    lr = LinearRegression().fit(X, y)
    return lr


def preparePoly(data):
    X = data.drop(["charges", "region"], axis=1)
    y = data.charges
    quad = PolynomialFeatures(degree=2)
    X = quad.fit_transform(X)
    return X, y

def polynomialLR(data):
    X, y = preparePoly(data)

    plr = LinearRegression().fit(X, y)
    return plr


def prepareForest(data):
    X = data.drop(["charges"], axis=1)
    y = data.charges
    return X, y

def forest(data):
    X, y = prepareForest(data)

    forest = RandomForestRegressor(n_estimators=100, criterion="mse", random_state=1, n_jobs=1)
    forest.fit(X, y)
    return forest
