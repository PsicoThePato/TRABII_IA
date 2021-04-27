import numpy as np
import pandas as pd
import sklearn
from scipy import stats
from sklearn.datasets import load_iris, load_digits, load_wine, load_breast_cancer
import sklearn.model_selection
import sklearn.preprocessing
import sklearn.pipeline

from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import seaborn as sns
import matplotlib.pyplot as plt

import methods.kCentroids


def dict_initializer():
    iris = load_iris()
    wine = load_wine()
    cancer = load_breast_cancer()
    digits = load_digits()

    dataset_dict = {
        'iris': iris, 
        'wine': wine, 
        'breast_cancer': cancer,
        'digits': digits
        }

    knn = KNeighborsClassifier()
    knnDistance = KNeighborsClassifier(weights="distance")
    decisionTree = DecisionTreeClassifier()
    randomForest = RandomForestClassifier()
    kGA = methods.kCentroids.kCentroidsClassifier(
    groupingMethodString="genetic"
    )
    kMeansCentroids = methods.kCentroids.kCentroidsClassifier(
        groupingMethodString="kmeans"
    )

    classifiers_dict = {
        "decisionTree": decisionTree,
        "randomForest": randomForest,
        "knnDistance": knnDistance,
        "knn": knn,
        "kMeansCentroids": kMeansCentroids,
        "kGA": kGA,
    }

    hiperParams_dict = {
        "decisionTree": {"estimator__max_depth": [None, 3, 5, 10]},
        "randomForest": {"estimator__n_estimators": [10, 20, 50, 100]},
        "knnDistance": {"estimator__n_neighbors": [1, 3, 5, 7]},
        "knn": {"estimator__n_neighbors": [1, 3, 5, 7]},
        "kMeansCentroids": {"estimator__k": [1, 3, 5, 7]},
        "kGA": {"estimator__k": [1, 3, 5, 7]},
    }

    return dataset_dict, classifiers_dict, hiperParams_dict

if __name__ == "__main__":
    rkf = sklearn.model_selection.RepeatedStratifiedKFold(
        n_splits=10, n_repeats=3
    )

    results_table = pd.DataFrame(
        columns=["Média", "Desvio Padrão", "Limite inferior", "Limite Superior"]
    )

    dataset_dict, classifiers_dict, hiperParams_dict = dict_initializer()
    
    for df_name, dataset in dataset_dict.items():
        scores_df = pd.DataFrame()
        for name, classifier in classifiers_dict.items():
            scalar = sklearn.preprocessing.StandardScaler()
            pipeline = sklearn.pipeline.Pipeline(
                [("transformer", scalar), ("estimator", classifier)]
            )
            gs = sklearn.model_selection.GridSearchCV(
                estimator=pipeline,
                param_grid=hiperParams_dict[name],
                scoring="accuracy",
                cv=4,
            )

            scores = sklearn.model_selection.cross_val_score(
                gs, dataset.data, dataset.target, scoring="accuracy", cv=rkf
            )
            scores_df[name] = scores
            mean = scores.mean()
            std = scores.std()
            inf, sup = stats.norm.interval(
                0.95, loc=mean, scale=std / np.sqrt(len(scores))
            )

            results_table = results_table.append(
                pd.Series(
                    {
                        "Média": mean,
                        "Desvio Padrão": std,
                        "Limite inferior": inf,
                        "Limite Superior": sup,
                    },
                    name=name,
                )
            )
            print(results_table)
        path = f"output/tables/ParametrizedMethods{df_name}Table.csv"
        results_table.to_csv(path)
        sns.boxplot(data=scores_df)
        plt.savefig(f"output/fig/{df_name}_boxplot_parametrized")
        plt.close()
        scores_df.to_csv(f'output/scores/ParametrizedMethods{df_name}_scores.csv')
