from numpy import argmax
import random
import seaborn as sbn
from numpy.random import rand, randn
from scipy.linalg import qr
from numpy import ones
from scipy import stats
import scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import linalg as LA
import time
import numpy as np




# This function is reused from the official replication package of SETS:
# https://github.com/GIST-NJU/SETS
def GD(IDs, features):
    selected_features = features[list(IDs)]
    dot_p = np.dot(selected_features, selected_features.T)
    sign, Log_det = np.linalg.slogdet(dot_p)
    return Log_det

# This function is reused from the official replication package of SETS:
# https://github.com/GIST-NJU/SETS
def STD(IDs, features):
  x_sample = features[list(IDs)]
  std_f=np.std(x_sample, axis=0)
  L2norm=LA.norm(std_f, 2)
  L1norm=LA.norm(std_f, 1)
  end_time = time.perf_counter()
  return L1norm

# This function is reused from the official replication package of SETS:
# https://github.com/GIST-NJU/SETS
def gini_score(Output_probability):
  gini_scores=[]
  for i in range(len(Output_probability)):
    sum=0
    for j in range(len(Output_probability[0])):
      sum= sum + Output_probability[i][j]**2
    gini_scores.append(1-sum)
  return gini_scores

# This function is reused from the official replication package of SETS:
# https://github.com/GIST-NJU/SETS
def maxp_score(Output_probability):
    return [1 - max(prob) for prob in Output_probability]



# This function is reused from the official replication package of SETS:
# https://github.com/GIST-NJU/SETS

# size: budget
# index: indexes of all the test inputs
# features: features of all the test inputs
# output_probability: the output probabilities of all the test inputs
# uncertainty:  str, Maxp or DeepGini
# diversity: str, GD or STD
# a: the reduction coefficient
def sets(size, index, features, output_probability, uncertainty, diversity, a):
    start_time = time.time()

    if uncertainty == "gini":
        un_scores = gini_score(output_probability)
    elif uncertainty == "maxp":
        un_scores = maxp_score(output_probability)

    sorted_indices = sorted(index, key=lambda i: un_scores[i], reverse=True)

    top_percent_count = max(1, int(a * size)) # reduction
    if a * size > len(index):
        top_percent_count = len(index)
    filtered_indices = sorted_indices[:top_percent_count]

    chunks = [filtered_indices[i::size] for i in range(size)]

    S = []
    current_gd = 0
    for chunk in chunks:
        max_gd_delta = -float('inf')
        best_index = -1
        gd_deltas = []
        gd_datas = []

        if len(chunk) == 0:
            continue

        for i in chunk:

            if diversity == "gd":
                new_gd = GD(S + [i], features)
            elif diversity == "std":
                new_gd = STD(S + [i], features)
            gd_datas.append(new_gd)
            gd_delta = new_gd - current_gd  #the diversity contribution
            gd_deltas.append(gd_delta)

        min_gd = min(gd_deltas) #min_max normalization
        max_gd = max(gd_deltas)
        if max_gd - min_gd > 0:
            normalized_gd_deltas = [(gd - min_gd) / (max_gd - min_gd + 0.5) for gd in gd_deltas]
        else:
            normalized_gd_deltas = [0] * len(gd_deltas)

        for idx, i in enumerate(chunk):
            current_un = un_scores[i]
            objective_value = current_un * normalized_gd_deltas[idx]

            if objective_value > max_gd_delta:
                max_gd_delta = objective_value
                best_index = i


        if best_index != -1:
            S.append(best_index)
            ind = chunk.index(best_index)
            current_gd = gd_datas[ind]


    end_time = time.time()
    execution_time = end_time - start_time


    return S, execution_time