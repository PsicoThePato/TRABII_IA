import pandas as pd
import sklearn
from scipy import stats
import numpy as np


def get_pipeline(classifier):
    scalar = sklearn.preprocessing.StandardScaler()
    pipeline = sklearn.pipeline.Pipeline(
        [("transformer", scalar), ("estimator", classifier)]
    )
    return pipeline


def metrics_series(scores, name):
    mean = scores.mean()
    std = scores.std()
    inf, sup = stats.norm.interval(
        0.95, loc=mean, scale=std / np.sqrt(len(scores))
    )
    score_series = pd.Series(
        {
            "Média": mean,
            "Desvio Padrão": std,
            "Limite inferior": inf,
            "Limite Superior": sup,
        },
        name=name,
    )
    return score_series
