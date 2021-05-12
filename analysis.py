import pandas as pd
import os
import re
from functools import reduce
from scipy import stats
import numpy as np

def load_tables(path):
    tables_path = os.getcwd() + path
    tables_list = os.listdir(tables_path)
    dataset_list = [pd.read_csv(tables_path + table) for table in tables_list]
    return dataset_list, tables_list


def join_tables(name_list, tables_path, output_path, merge=False):
    join_function = lambda x,y: x.iloc[:, 1:].append(y.iloc[:, 1:])
    if merge:
        join_function = lambda x,y: x.iloc[:, 1:].merge(y.iloc[:, 1:], left_index=True, right_index=True)

    tables_path = os.getcwd() + tables_path
    tables_list = os.listdir(tables_path)
    for name in name_list:
        p = re.compile('.*'+name+'.*')
        matches = [string for string in tables_list if p.match(string)]
        dataset_tables = [pd.read_csv(tables_path + match) for match in matches]
        joined_table = reduce(join_function, dataset_tables)
        joined_table.to_csv(os.getcwd() + output_path + name, index=False)
        

def tWilcoxon(score1, score2):
    t_test = stats.ttest_rel(score1, score2)[1]
    wilcoxon = stats.wilcoxon(score1, score2)[1]
    return t_test, wilcoxon


def generate_latex_pairTable(path):
    dataset_list, tables_list = load_tables('/output/joined_scores/')
    for dataset, dataset_name in zip(dataset_list, tables_list):
        len_row = len(dataset.columns)
        tabela_pareada = pd.DataFrame(np.zeros(shape=(len_row, len_row)))
        for i in range(len(dataset.columns) - 1):
            name = dataset.columns[i]
            t_test_list = []
            wilcoxon_list = []
            for j in range(i+1, len(dataset.columns)):
                t_test, wilcoxon = tWilcoxon(dataset.iloc[:, i], dataset.iloc[:, j])
                t_test_list.append(t_test)
                wilcoxon_list.append(wilcoxon)
            linha = [name] + t_test_list
            coluna = [name] + wilcoxon_list
            tabela_pareada.iloc[i, i:] = linha
            tabela_pareada.iloc[i:, i] = coluna
        tabela_pareada.iloc[-1, -1] = dataset.columns[-1]
        tabela_pareada.to_latex(path+dataset_name)


if __name__ == '__main__':
    name_list = ['cancer', 'digits', 'iris', 'wine']
    join_tables(name_list, '/output/scores/', '/output/joined_scores/', merge=True)
    path = 'output/latex/'
    generate_latex_pairTable(path)

