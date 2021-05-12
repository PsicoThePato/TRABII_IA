from typing import List
from time import time

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

from . import utility


def sexo(estados, populacao, cross_ratio, sse_teste):
    choice = np.random.choice(range(populacao), 2, p=sse_teste, replace=False)
    pai = estados[choice[0]]
    mae = estados[choice[1]]
    novo_estado_paiD = pai.copy()
    novo_estado_maeD = mae.copy()
    novo_estado_paiD[int(cross_ratio * len(novo_estado_paiD)):] = \
        mae[int(cross_ratio * len(novo_estado_paiD)):]
    novo_estado_maeD[int(cross_ratio * len(novo_estado_maeD)):] = \
        pai[int(cross_ratio * len(novo_estado_maeD)):]
    return (novo_estado_paiD, novo_estado_maeD)


def pega_nova_geracao(
    populacao: int,
    sse_total: float,
    sse_list: np.array,
    cross_ratio: float,
    k: int,
    estados: List,
    mutation_ratio: float,
):
    #sse_list = (1 - sse_list/sse_total)/(populacao - 1)
    sse_teste = sse_list.copy()
    nova_geracao = []
    sse_list_return = np.array([])
    sse_total = 0
    for _ in range(int(populacao/2)):
        novo_estado_paiD, novo_estado_maeD = sexo(estados, populacao, cross_ratio, sse_teste)
        if(np.random.rand() < mutation_ratio):
            mutated_linha = np.random.choice(len(novo_estado_paiD))
            novo_estado_paiD.iloc[[mutated_linha], [-1]] = np.random.choice(k)
        if(np.random.rand() < mutation_ratio):
            mutated_linha = np.random.choice(len(novo_estado_maeD))
            novo_estado_maeD.iloc[[mutated_linha], [-1]] = np.random.choice(k)
          
        nova_geracao.append(novo_estado_paiD.copy())
        nova_geracao.append(novo_estado_maeD.copy())

        sse_total, sse_list_return = utility.append_sse(
            sse_total, novo_estado_paiD, sse_list_return
            )

        sse_total, sse_list_return = utility.append_sse(
            sse_total, novo_estado_maeD, sse_list_return
            )

    return nova_geracao, sse_total, sse_list_return


def run_ag(iris_df, k, populacao, cross_ratio, m):
    estados = []
    sse_list = np.array([])
    sse_total = 0
    for _ in range(populacao):
        #
        state = utility.random_state(iris_df, k)
        estados.append(state.copy())
        sse_total, sse_list = utility.append_sse(
            sse_total, state, sse_list
            )

    sse_list = (1 - sse_list/sse_total)/(populacao - 1)

    nova_gen = estados.copy()
    inicio = time()
    for _ in range(20):
        if (time() - inicio >= 1):
            return nova_gen[sse_list.argmin()].copy()
        nova_gen, sse_total, sse_list = pega_nova_geracao(
            populacao, sse_total, sse_list,
            cross_ratio, k, nova_gen, m
            )

        sse_list = (1 - sse_list/sse_total)/(populacao - 1)

    return nova_gen[sse_list.argmin()].copy()


if __name__ == "__main__":
    populacao = 10
    k = 10
    cross_ratio = 0.5
    m = 0.1
    iris = load_iris()
    iris_df = (pd.DataFrame(data=iris.data, columns=iris.feature_names))
    melhor = run_ag(iris_df, k, populacao, cross_ratio, m)
    print(melhor)
