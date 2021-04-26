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

from grouping import geneticGrouping

class kCentroidsClassifier(BaseEstimator):
    
    
    def __init__(self, k=None, groupingMethodString=None):
        super().__init__()
        self.k = k
        self.groupingMethodString = groupingMethodString  #piece of shit sklearn demands it
        self.availableGroupingMethods = {'placeholder': None, 'genetic': self.genetic_grouping, 'kmeans': self.kminhos_grouping}
        self.groupingMethod = self.getGroupingMethod(self.groupingMethodString)
        self.classes_map = None
        self.clusters = None


    def getGroupingMethod(self, groupingMethodString):
        if not groupingMethodString or groupingMethodString not in self.availableGroupingMethods.keys():
            raise ValueError("Método inválido!")
        return self.availableGroupingMethods[groupingMethodString]


    def kminhos_grouping(self, dado_classe):
        kminhos = sklearn.cluster.KMeans(n_clusters=self.k, random_state=0).fit(dado_classe)
        return kminhos.cluster_centers_

    
    def genetic_grouping(self, dado_classe):
        groups = geneticGrouping.run_ag(dado_classe, self.k, 10, 0.5, 0.1)
        groups_df_list = groups.groupby('C-grupo')
        
        group_centroid_list = []
        for group_label, group_df in groups_df_list:
            group_df.drop(columns=['C-grupo'], inplace=True)
            group_centroid = np.array(group_df.sum()/len(group_df))
            group_centroid_list.append(group_centroid)
        
        return np.array(group_centroid_list)


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
            cluster_centers = self.groupingMethod(dado_classe)
            for center in cluster_centers:
                clusters_list.append(center)
            classes_map.extend([classe]*len(cluster_centers))
        clusters_list = np.array(clusters_list)
        self.clusters = clusters_list
        self.classes_map = classes_map
        print("fitei")

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
    kClassifier = kCentroidsClassifier(groupingMethodString='genetic')

    scalar = StandardScaler()
    pipeline = Pipeline([('transformer', scalar), ('estimator', kClassifier)])

    grade={'estimator__k': [2,3,4]}

    gs = GridSearchCV(estimator=pipeline, param_grid = grade, 
                    scoring='accuracy', cv = 5)

    rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=6)

    scores = cross_val_score(gs, iris.data, iris.target, scoring='accuracy', cv = rkf)

    print(scores)

    mean = scores.mean()    
    std = scores.std()
    inf, sup = stats.norm.interval(0.95, loc=mean, 
                                scale=std/np.sqrt(len(scores)))