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
from sklearn.base import BaseEstimator
from scipy import stats
import sklearn.cluster
import numpy.linalg
from sklearn.model_selection import cross_val_score, GridSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class kCentroidsClassifier(BaseEstimator):
    def __init__(self, k=None, groupingMethod=None):
        super().__init__()
        self.k = k
        self.groupingMethod = groupingMethod
        self.classes_map = None
        self.clusters = None


    def fit(self, x_train, y_train):
        if len(x_train) != len(y_train):
            raise ValueError("Quantidade de classes diferente de quantidade de dados a serem classificados")
        df_data = pd.DataFrame(data=x_train)
        df_data['classe'] = y_train
        data_by_classes = df_data.groupby('classe')
        classes_map = []
        clusters_list = []
        for classe, dado_classe in data_by_classes:
            dado_classe.drop(columns=['classe'], inplace=True)
            kminhos = self.groupingMethod(n_clusters=self.k, random_state=0).fit(dado_classe)
            clusters_list.append(kminhos.cluster_centers_)
            for _ in kminhos.cluster_centers_:
                classes_map.append(classe)
        clusters_list = np.array(clusters_list)
        clusters_list = clusters_list.reshape(len(data_by_classes) * self.k, x_train.shape[1])
        self.clusters = clusters_list
        self.classes_map = classes_map


    def predict(self, x_test):
        y_test = []
        for row in x_test:
            dist_array = np.linalg.norm(self.clusters - row, axis=1)
            predicted = np.argmin(dist_array)
            y_test.append(self.classes_map[predicted])
        return np.array(y_test)

if __name__ == '__main__':
    iris = load_iris()
    method = sklearn.cluster.KMeans
    kClassifier = kCentroidsClassifier()
    
    scalar = StandardScaler()
    pipeline = Pipeline([('transformer', scalar), ('estimator', kClassifier)])

    grade={'estimator__k': [2,3,4], 'estimator__groupingMethod': [sklearn.cluster.KMeans]}

    gs = GridSearchCV(estimator=pipeline, param_grid = grade, 
                    scoring='accuracy', cv = 5)

    rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=6)

    scores = cross_val_score(gs, iris.data, iris.target, scoring='accuracy', cv = rkf)

    print(scores)

    mean = scores.mean()
    std = scores.std()
    inf, sup = stats.norm.interval(0.95, loc=mean, 
                                scale=std/np.sqrt(len(scores)))