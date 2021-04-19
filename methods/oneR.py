from typing import List

import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_iris
from sklearn import preprocessing
from sklearn.utils.multiclass import unique_labels

class oneRClassifier():
    def __init__(self):
        self.best_feature = None
        self.prob_table = None
        self.discretizer = None


    def predict(self, data):
        result = []
        best_feature_array = pd.DataFrame(data=self.discretizer.transform(data), columns=data.columns)[self.best_feature]
        target = [*self.prob_table.columns]
        for point in best_feature_array:
            prediction = np.random.choice(target, p=self.prob_table.loc[point].values)
            result.append(prediction)
        return result


    def fit(self, data: sklearn.utils.Bunch):
        discretizer = preprocessing.KBinsDiscretizer(n_bins=2*len(unique_labels(data.target)), encode='ordinal', strategy='kmeans')
        discretizer.fit(data.data)
        self.discretizer = discretizer
        df_data = (pd.DataFrame(data=discretizer.transform(data.data), columns=data.feature_names))
        df_data['classe'] = data.target
        contingency_df = {}
        best_score = float('-inf')
        best_feature = None
        for col in list(df_data.iloc[:, :-1]):
            contingency_df[col] = pd.crosstab(df_data['classe'], df_data[col])
            score_feature = contingency_df[col].agg('max').sum()
            if(score_feature > best_score):
                best_feature = col
        self.best_feature = best_feature
        self.prob_table = contingency_df[col].apply(lambda x: x/sum(x)).T


if __name__ == '__main__':
    iris = load_iris()
    oneR = oneRClassifier()
    oneR.fit(iris)
    df_data = (pd.DataFrame(data=iris.data, columns=iris.feature_names))
    oneR.predict(df_data)
