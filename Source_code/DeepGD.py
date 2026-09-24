import tensorflow as tf
import copy
from scipy.spatial.distance import cdist
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.core.mutation import Mutation
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.crossover import Crossover
from pymoo.core.sampling import Sampling
import time
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
import random
import numpy as np

# All functions and classes below are directly reused from the official SETS replication package:
# https://github.com/GIST-NJU/SETS


def find_knee_point(pareto_front):
    num_points = pareto_front.shape[0]

    # Calculate distances to the nearest neighbors for each point
    distances = cdist(pareto_front, pareto_front)
    min_distances = np.min(distances + np.eye(num_points) * np.inf, axis=1)

    # Find the point with the maximum minimum distance (the knee point)
    knee_point_index = np.argmax(min_distances)

    return pareto_front[knee_point_index], knee_point_index


def gini_score(Output_probability):
  gini_scores=[]
  for i in range(len(Output_probability)):
    sum=0
    for j in range(len(Output_probability[0])):
      sum= sum + Output_probability[i][j]**2
    gini_scores.append(1-sum)
  return gini_scores



def GD(IDs, features):

    selected_features = features[list(IDs)]
    dot_p = np.dot(selected_features, selected_features.T)
    sign, Log_det = np.linalg.slogdet(dot_p)
    return Log_det



ID = 300
size_ind = 100
n_gen = 30
p_size = 700
n_off = 500
m_rate = 0.7


class MyProblem(ElementwiseProblem):

    def __init__(self,Gini_scores,features_test,n_var=size_ind):
        super().__init__(n_var=n_var, n_obj=2, n_constr=0, xl=0, xu=len(Gini_scores) - 1, type_var=list)
        self.Gini_scores = Gini_scores
        self.features_test = features_test

    def _evaluate(self, x, out, *args, **kwargs):
        global count
        # Opt version: Calculate the average Gini score
        Ave_gini = np.mean([self.Gini_scores[c] for c in x])

        # print(type(x))
        div_score = GD(x, self.features_test)
        fit1 = Ave_gini
        fit2 = abs(div_score)

        out["F"] = np.array([-fit1, -fit2], dtype=float)


class MySampling(Sampling):
    def __init__(self, index_withoutnoisy):
        super().__init__()
        self.index_withoutnoisy = index_withoutnoisy

    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, problem.n_var), None, dtype=object)
        #print(len(index_withoutnoisy))

        for i in range(n_samples):
            X[i] = np.asarray(random.sample(list(self.index_withoutnoisy), problem.n_var))
        return X


# original
class MyCrossover(Crossover):
    def __init__(self,Gini_scores,index_withoutnoisy):
        # define the crossover: number of parents and number of offsprings
        super().__init__(2, 2)
        self.Gini_scores = Gini_scores
        self.index_withoutnoisy = index_withoutnoisy

    def _do(self, problem, X, **kwargs):
        # The input of has the following shape (n_parents, n_matings, n_var)
        _, n_matings, n_var = X.shape

        def pri(xx):
            gini_X = [self.Gini_scores[i] for i in xx]
            zipped_lists = zip(gini_X, xx)
            # sort from the highest gini score to the lowest
            sorted_zipped_lists = sorted(zipped_lists, reverse=True)
            sorted_list1 = [element for _, element in sorted_zipped_lists]
            return sorted_list1

        # The output with the shape (n_offsprings, n_matings, n_var)
        Y = np.full_like(X, None, dtype=object)
        # # for each mating provided
        for k in range(n_matings):
            a = pri(X[0, k])
            b = pri(X[1, k])
            crossoverpoint = random.randint(1, (n_var - 1))
            off1 = list(a[:crossoverpoint]) + list(b[:len(b) - crossoverpoint])
            off2 = list(b[len(b) - crossoverpoint:]) + list(a[crossoverpoint:])
            # # prepare the offsprings

            off1_final = off1
            off2_final = off2

            #print(len(index_withoutnoisy))

            # join the character list and set the output
            if len(set(off1)) < n_var:
                Rsize = n_var - len(set(off1))
                off1_replace = random.sample(list(set(self.index_withoutnoisy) - set(off1)), Rsize)
                off1 = list(set(off1))
                off1.extend(set(off1_replace))
            if len(set(off2)) < n_var:
                Rsize = n_var - len(set(off2))
                off2_replace = random.sample(list(set(self.index_withoutnoisy) - set(off2)), Rsize)
                off2 = list(set(off2))
                off2.extend(set(off2_replace))

            Y[0, k], Y[1, k] = np.array(off1), np.array(off2)
        return Y


class MyMutation(Mutation):
    def __init__(self,Gini_scores,features_test,index_withoutnoisy):
        super().__init__()
        self.Gini_scores = Gini_scores
        self.features_test =features_test
        self.index_withoutnoisy = index_withoutnoisy

    def _do(self, problem, X, **kwargs):

        def pri(xx):
            gini_X = [self.Gini_scores[i] for i in xx]
            zipped_lists = zip(gini_X, xx)
            # sort from the highest gini score to the lowest
            sorted_zipped_lists = sorted(zipped_lists, reverse=True)
            sorted_list1 = [element for _, element in sorted_zipped_lists]
            return sorted_list1

        mutation_rate = m_rate
        mut = np.full_like(X, None, dtype=object)
        for i in range(len(X)):
            random_value = random.random()
            N_least = int(size_ind / 100)
            ff = len(X[i]) - N_least
            gd_scores_mut = []
            xx = pri(X[i])
            if random_value <= mutation_rate:
                while ff < len(X[i]):
                    b = xx[:ff] + xx[ff + 1:]
                    ss = GD(b, self.features_test)
                    gd_scores_mut.append(ss)
                    ff = ff + 1
                xx[len(X[i]) - N_least + gd_scores_mut.index(max(gd_scores_mut))] = random.choice(
                    list(set(self.index_withoutnoisy) - set(X[i])))
                mut[i] = np.array(xx)
            else:
                mut[i] = X[i]
        ###########################################

        return mut


class MyDuplicateElimination(ElementwiseDuplicateElimination):
    def is_equal(self, a, b):
        return np.array_equal(a.X, b.X)


def deepgd(size,index,gini_scores,features):
    problem = MyProblem(n_var=size,Gini_scores=gini_scores,features_test=features)

    algorithm = NSGA2(pop_size=p_size,
                    n_offsprings=n_off,
                    sampling=MySampling(index),
                    crossover=MyCrossover(gini_scores,index),
                    mutation=MyMutation(gini_scores,features,index),
                    eliminate_duplicates=MyDuplicateElimination())

    start_time = time.time()
    res = minimize(problem,
                   algorithm,
                   ('n_gen', n_gen),
                   verbose=False)
    F = res.F
    X = res.X
    knee_point, indd = find_knee_point(F)
    end_time = time.time()
    execution_time = end_time-start_time

    return X[indd], execution_time