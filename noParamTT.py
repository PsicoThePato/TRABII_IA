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

import seaborn as sns
import matplotlib.pyplot as plt

import methods.oneR

if __name__ == "__main__":
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

    rkf = sklearn.model_selection.RepeatedStratifiedKFold(
        n_splits=10, n_repeats=3
    )

    results_table = pd.DataFrame(
        columns=["Média", "Desvio Padrão", "Limite inferior", "Limite Superior"]
    )
    
    oneR = methods.oneR.oneRClassifier()
    gNB = GaussianNB()
    zeroR = DummyClassifier()
    random = DummyClassifier(strategy="uniform")
    stratifiedRandom = DummyClassifier(strategy="stratified")
    classifiers_dict = {
        "zeroR": zeroR,
        "oneR": oneR,
        "naiveBayes": gNB,
        "random": random,
        "stratifiedRandom": stratifiedRandom,
    }

    for df_name, dataset in dataset_dict.items():

        scores_df = pd.DataFrame()
        for name, classifier in classifiers_dict.items():
            scalar = sklearn.preprocessing.StandardScaler()
            pipeline = sklearn.pipeline.Pipeline(
                [("transformer", scalar), ("estimator", classifier)]
            )
            scores = sklearn.model_selection.cross_val_score(
                pipeline, dataset.data, dataset.target, scoring="accuracy", cv=rkf
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
        path=f'output/tables/noParamMethods{df_name}Table.csv'
        results_table.to_csv(path)
        sns.boxplot(data=scores_df)
        plt.savefig(f"output/fig/{df_name}_boxplot")
        plt.close()
        scores_df.to_csv(f'output/scores/noParamMethods{df_name}_scores.csv')