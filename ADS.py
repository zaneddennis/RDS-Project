import pandas as pd
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

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
    x = data.drop(["charges"], axis=1)
    y = data.charges

    lr = LinearRegression().fit(x, y)
    return lr
