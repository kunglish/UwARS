import numpy as np
import copy
from SETS import GD,STD,gini_score,maxp_score,sets
from DeepGD import deepgd
from RS import rs
import pickle
import sys

# This function is reused from the official replication package of SETS:
# https://github.com/GIST-NJU/SETS
def faults(sample,mis_ind_test,Clustering_labels):
  i=0
  pos=0
  neg=0
  i=0
  cluster_lab=[]
  nn=-1
  for l in sample:
    if l in list(mis_ind_test):
      neg=neg+1
      # print("index mis",l)
      ind=list(mis_ind_test).index(l)
      if (Clustering_labels[ind]>-1):
        cluster_lab.append(Clustering_labels[ind])
      if (Clustering_labels[ind]==-1):
        cluster_lab.append(nn)
        nn=nn-1
    else:
      pos=pos+1

    # i=i+1
  faults_n=len(list(set(cluster_lab)))

  cluster_1noisy=copy.deepcopy(cluster_lab)
  for i in range(len(cluster_1noisy)):
   if cluster_1noisy[i] <=-1:
     cluster_1noisy[i]=-1
  faults_1noisy=len(list(set(cluster_1noisy)))
  return faults_n,faults_1noisy, neg


data_model_pairs = [
    ("mnist", "LeNet1"),
    ("mnist", "LeNet5"),
    ("cifar10", "12Conv"),
    ("cifar10", "ResNet20"),
    ("Fashion_mnist", "LeNet4"),
    ("SVHN", "LeNet5"),
    ("Fruit360","ResNet50"),
    ("TinyImageNet", "ResNet101")
]




if len(sys.argv) < 5:
    print("Usage: python exp_2_3.py <approach> <n> <data_path> <output_path>")
    sys.exit(1)

approach = sys.argv[1] # the approach name
n = sys.argv[2] # repeat times
data_path = sys.argv[3] # the path you locate the Fault_clusters folder
output_path = sys.argv[4] # the path you want to save the result file


for data_name, model_name in data_model_pairs:
    print(f"Dataset: {data_name}, Model: {model_name}")
    if data_name == "TinyImageNet":
        with open(f'{data_path}/{data_name}_{model_name}/cluster_results.pkl', 'rb') as f:
            Clustering_labels = pickle.load(f)
        with open(f'{data_path}/{data_name}_{model_name}/mis_index_test.pkl', 'rb') as f:
            val_mis_data = pickle.load(f)
        mis_ids_test = [item[2] for item in val_mis_data]
    else:
        Clustering_labels = np.load(f'{data_path}/{data_name}_{model_name}/cluster_results.npy')
        mis_ids_test = np.load(f'{data_path}/{data_name}_{model_name}/mis_index_test.npy')

    output_probability = np.load(f'{data_path}/{data_name}_{model_name}/output_probability.npy')
    features_test = np.load(f'{data_path}/{data_name}_{model_name}/features_test.npy')

    if data_name == "Fruit360":
        mis_ids_test = mis_ids_test[0]

    total_faults = len(set(Clustering_labels)) - 1

    noisy_index = []
    for i in range(len(mis_ids_test)):
        if Clustering_labels[i] == -1:
            noisy_index.append(mis_ids_test[i])
    sett = list(range(0, len(output_probability)))
    index_withoutnoisy = set(sett) - set(noisy_index)

    if approach == "sets":
        print("Approach:", approach)
        # for size in [100, 300, 500]:
        for size in [100, 300, 500, 700, 900, 1100, 1300, 1500]:
            # save the selection results
            fault_list = []
            time_list = []
            fdr_list = []
            subset_list = []

            for i in range(int(n)):
                selected_subset, exe_time = sets(size, index_withoutnoisy, features_test, output_probability, "maxp", "gd", a=3)
                _, find_faults, _ = faults(selected_subset, mis_ids_test, Clustering_labels)
                fdr = find_faults / min(size, total_faults)

                fault_list.append(find_faults)
                time_list.append(exe_time)
                fdr_list.append(fdr)
                subset_list.append(selected_subset)

                print("FDR:", fdr)
                print("selection time:", exe_time)

            filename = f"{output_path}/{approach}_{data_name}_{model_name}_{size}.txt"
            with open(filename, 'w') as file:
               file.write("Fault List:\n")
               file.write("\n".join(map(str, fault_list)) + "\n")  # Write fault_list
               file.write("FDR List:\n")
               file.write("\n".join(map(str, fdr_list)) + "\n")  # Write fdr_list
               file.write("Time List:\n")
               file.write("\n".join(map(str, time_list)) + "\n")  # Write time_list
               file.write("Subset List:\n")
               file.write("\n".join(map(str, subset_list)) + "\n")  # Write subset_list
            print(f"Saved file: {filename}")

    elif approach == "deepgd":
        print("Approach:", approach)
        # for size in [100, 300, 500]:
        for size in [100, 300, 500, 700, 900, 1100, 1300, 1500]:
            # save the selection results
            fault_list = []
            time_list = []
            fdr_list = []
            subset_list = []

            gini_scores = gini_score(output_probability)
            for i in range(int(n)):
                selected_subset, exe_time = deepgd(size,index_withoutnoisy,gini_scores,features_test)
                _, find_faults, _ = faults(selected_subset, mis_ids_test, Clustering_labels)
                fdr = find_faults / min(size, total_faults)

                fault_list.append(find_faults)
                time_list.append(exe_time)
                fdr_list.append(fdr)
                subset_list.append(selected_subset)

                print("FDR:", fdr)
                print("selection time:", exe_time)

            filename = f"{output_path}/{approach}_{data_name}_{model_name}_{size}.txt"
            with open(filename, 'w') as file:
                file.write("Fault List:\n")
                file.write("\n".join(map(str, fault_list)) + "\n")  # Write fault_list
                file.write("FDR List:\n")
                file.write("\n".join(map(str, fdr_list)) + "\n")  # Write fdr_list
                file.write("Time List:\n")
                file.write("\n".join(map(str, time_list)) + "\n")  # Write time_list
                file.write("Subset List:\n")
                file.write("\n".join(map(str, subset_list)) + "\n")  # Write subset_list
            print(f"Saved file: {filename}")

    elif approach == "rs":
        print("Approach:", approach)
        # for size in [100, 300, 500]:
        for size in [100, 300, 500, 700, 900, 1100, 1300, 1500]:
            # save the selection results
            fault_list = []
            time_list = []
            fdr_list = []

            for i in range(int(n)):
                selected_subset, exe_time = rs(size,index_withoutnoisy)
                _, find_faults, _ = faults(selected_subset, mis_ids_test, Clustering_labels)
                fdr = find_faults / min(size, total_faults)

                fault_list.append(find_faults)
                time_list.append(exe_time)
                fdr_list.append(fdr)

                print("FDR:", fdr)
                print("selection time:", exe_time)

            filename = f"{output_path}/{approach}_{data_name}_{model_name}_{size}.txt"
            with open(filename, 'w') as file:
                file.write("Fault List:\n")
                file.write("\n".join(map(str, fault_list)) + "\n")  # Write fault_list
                file.write("FDR List:\n")
                file.write("\n".join(map(str, fdr_list)) + "\n")  # Write fdr_list
                file.write("Time List:\n")
                file.write("\n".join(map(str, time_list)) + "\n")  # Write time_list
            print(f"Saved file: {filename}")





