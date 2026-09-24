import os
import numpy as np
import copy
import random
from SETS import gini_score, sets
import pickle
from DeepGD import deepgd
from RS import rs
from UwARS import uwars, load_test_data, preprocessing
import sys

# This function is reused from the official replication package of SETS:
# https://github.com/GIST-NJU/SETS
def faults(sample, mis_ind_test, Clustering_labels):
    pos = 0
    neg = 0
    cluster_lab = []
    nn = -1
    for l in sample:
        if l in list(mis_ind_test):
            neg = neg + 1
            ind = list(mis_ind_test).index(l)
            if Clustering_labels[ind] > -1:
                cluster_lab.append(Clustering_labels[ind])
            if Clustering_labels[ind] == -1:
                cluster_lab.append(nn)
                nn = nn - 1
        else:
            pos = pos + 1
    faults_n = len(list(set(cluster_lab)))
    cluster_1noisy = copy.deepcopy(cluster_lab)
    for i in range(len(cluster_1noisy)):
        if cluster_1noisy[i] <= -1:
            cluster_1noisy[i] = -1
    faults_1noisy = len(list(set(cluster_1noisy)))
    return faults_n, faults_1noisy, neg


data_model_pairs = [
    ("mnist", "LeNet1"),
    ("mnist", "LeNet5"),
    ("cifar10", "12Conv"),
    ("cifar10", "ResNet20"),
    ("Fashion_mnist", "LeNet4"),
    ("SVHN", "LeNet5"),
    ("Fruit360", "ResNet50"),
    ("TinyImageNet", "ResNet101")
]

DATASET_CLASSES = {
    "mnist": 10,
    "Fashion_mnist": 10,
    "cifar10": 10,
    "SVHN": 10,
    "Fruit360": 141,
    "TinyImageNet": 200,
}

SEEDS = [0, 1, 2, 3, 4]
SIZE = 500

if len(sys.argv) < 4:
    print("Usage: python exp_4.py <n> <data_path> <output_path>")
    sys.exit(1)

data_path = sys.argv[2]
output_path = sys.argv[3]

log_file = os.path.join(output_path, "selection_fdr_log.txt")

for data_name, model_name in data_model_pairs:
    print(f"Dataset: {data_name}, Model: {model_name}")

    if data_name == "TinyImageNet":
        with open(f'{data_path}/Fault_clusters/{data_name}_{model_name}/cluster_results.pkl', 'rb') as f:
            Clustering_labels = pickle.load(f)
        with open(f'{data_path}/Fault_clusters/{data_name}_{model_name}/mis_index_test.pkl', 'rb') as f:
            val_mis_data = pickle.load(f)
        mis_ids_test = [item[2] for item in val_mis_data]
    else:
        Clustering_labels = np.load(f'{data_path}/Fault_clusters/{data_name}_{model_name}/cluster_results.npy')
        mis_ids_test = np.load(f'{data_path}/Fault_clusters/{data_name}_{model_name}/mis_index_test.npy')

    output_probability = np.load(f'{data_path}/Fault_clusters/{data_name}_{model_name}/output_probability.npy')
    features_test = np.load(f'{data_path}/Fault_clusters/{data_name}_{model_name}/features_test.npy')

    if data_name == "Fruit360":
        mis_ids_test = mis_ids_test[0]

    total_faults = len(set(Clustering_labels)) - 1

    with open(f'{data_path}/Retrain/{data_name}_{model_name}.pkl', 'rb') as f:
        loaded_indices = pickle.load(f)
    noisy_index = []
    for i in range(len(mis_ids_test)):
        if Clustering_labels[i] == -1:
            noisy_index.append(mis_ids_test[i])
    intersection = sorted(
        set(loaded_indices) & (set(range(len(output_probability))) - set(noisy_index)))

    gini_scores = gini_score(output_probability)

    def log_fdr(method, seed, subset):
        _, found, _ = faults(subset, mis_ids_test, Clustering_labels)
        fdr = found / min(SIZE, total_faults)
        with open(log_file, 'a') as lf:
            lf.write(f"{method};{data_name}_{model_name};seed={seed};FDR={fdr:.4f}\n")

    subset, _ = sets(SIZE, list(intersection), features_test,
                     output_probability, "maxp", "gd", a=3)
    os.makedirs(f"{output_path}/SETS", exist_ok=True)
    with open(f"{output_path}/SETS/{data_name}_{model_name}_{SIZE}.pkl", 'wb') as f:
        pickle.dump(subset, f)
    log_fdr("SETS", "NA", subset)
    print(f"Saved SETS subset for {data_name}_{model_name}")

    os.makedirs(f"{output_path}/DeepGD", exist_ok=True)
    for seed in SEEDS:
        random.seed(seed)
        np.random.seed(seed)
        subset, _ = deepgd(SIZE, list(intersection), gini_scores, features_test)
        subset = [int(x) for x in subset]
        with open(f"{output_path}/DeepGD/{data_name}_{model_name}_{SIZE}_seed{seed}.pkl", 'wb') as f:
            pickle.dump(subset, f)
        log_fdr("DeepGD", seed, subset)
    print(f"Saved 5 DeepGD subsets for {data_name}_{model_name}")

    os.makedirs(f"{output_path}/RS", exist_ok=True)
    for seed in SEEDS:
        random.seed(seed)
        np.random.seed(seed)
        subset, _ = rs(SIZE, list(intersection))
        with open(f"{output_path}/RS/{data_name}_{model_name}_{SIZE}_seed{seed}.pkl", 'wb') as f:
            pickle.dump(subset, f)
        log_fdr("RS", seed, subset)
    print(f"Saved 5 RS subsets for {data_name}_{model_name}")

    k = DATASET_CLASSES.get(data_name, 10)
    x_test = load_test_data(data_name, data_path)
    phash_list, cluster_list = preprocessing(x_test, k)

    os.makedirs(f"{output_path}/UwARS", exist_ok=True)
    for seed in SEEDS:
        random.seed(seed)
        np.random.seed(seed)
        subset, _ = uwars(SIZE, list(intersection), x_test,
                            output_probability, "maxp", 2, k=k,
                            precomputed=(phash_list, cluster_list))
        with open(f"{output_path}/UwARS/{data_name}_{model_name}_{SIZE}_seed{seed}.pkl", 'wb') as f:
            pickle.dump(subset, f)
        log_fdr("UwARS", seed, subset)
    print(f"Saved 5 UwARS subsets for {data_name}_{model_name}")