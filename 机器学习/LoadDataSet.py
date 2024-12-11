from enum import Enum
from sklearn import datasets
import seaborn as sns
from sklearn.datasets import fetch_openml

class DataName(Enum):
    Wine = 0
    Titanic = 1
    BreastCancer = 2
    CriditApproval = 3
    Diabetes = 4



def LoadData(dataName : DataName):
    if dataName == DataName.Wine:
        return datasets.load_wine()
    elif dataName == DataName.Titanic:
        return sns.load_dataset('titanic')
    elif dataName == DataName.BreastCancer:
        return datasets.load_breast_cancer()
    elif dataName == DataName.CriditApproval:
        return fetch_openml(name='credit-approval')
    elif dataName == DataName.Diabetes:
        return datasets.load_diabetes()
    else:
        print("DataName not found")
        return None

if __name__ == "__main__":
    data = LoadData(DataName.Wine)