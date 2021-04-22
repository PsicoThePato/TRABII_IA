from typing import List

import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_iris
from sklearn import preprocessing
from sklearn.utils.multiclass import unique_labels
import sklearn.model_selection
import sklearn.preprocessing
import sklearn.pipeline
from scipy import stats


class oneRClassifier():
    def __init__(self):
        self.best_feature = None
        self.prob_table = None
        self.discretizer = None


    def predict(self, data):
        result = []
        best_feature_array = pd.DataFrame(data=self.discretizer.transform(data))[self.best_feature]
        target = [*self.prob_table.columns]
        for point in best_feature_array:
            prediction = np.random.choice(target, p=self.prob_table.loc[point].values)
            result.append(prediction)
        return result


    def fit(self, x_train, y_train):
        discretizer = preprocessing.KBinsDiscretizer(n_bins=2*len(unique_labels(y_train)), encode='ordinal', strategy='kmeans')
        discretizer.fit(x_train)
        self.discretizer = discretizer
        df_data = (pd.DataFrame(data=discretizer.transform(x_train)))
        df_data['classe'] = y_train
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
    rkf = sklearn.model_selection.RepeatedStratifiedKFold(n_splits=10, n_repeats=3)
    scalar = sklearn.preprocessing.StandardScaler()
    pipeline = sklearn.pipeline.Pipeline([('transformer', scalar), ('estimator', oneR)])
    scores = sklearn.model_selection.cross_val_score(pipeline, iris.data, iris.target, scoring='accuracy', cv = rkf)
    print(scores)
    mean = scores.mean()
    std = scores.std()
    inf, sup = stats.norm.interval(0.95, loc=mean, 
                                scale=std/np.sqrt(len(scores)))

    print("\nMean Accuracy: %0.2f Standard Deviation: %0.2f" % (mean, std))
    print ("Accuracy Confidence Interval (95%%): (%0.2f, %0.2f)\n" % 
        (inf, sup)) 