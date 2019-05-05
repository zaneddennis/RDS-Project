import pandas as pd
import warnings
from sklearn.preprocessing import LabelEncoder
#from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

# takes a DataFrame
# returns a regressor
"""def simpleLR(data):
    le = LabelEncoder()
    le.fit(data.sex.drop_duplicates())
    data.sex = le.transform(data.sex)

    le.fit(data.smoker.drop_duplicates())
    data.smoker = le.transform(data.smoker)

    le.fit(data.region.drop_duplicates())
    data.region = le.transform(data.region)

    x = data.drop(["charges"], axis=1)
    y = data.charges

    #x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
    lr = LinearRegression().fit(x, y)

    return lr"""

# takes a DataFrame
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

def simpleLR(data):
    x = data.drop(["charges"], axis=1)
    y = data.charges

    lr = LinearRegression().fit(x, y)
    return lr
