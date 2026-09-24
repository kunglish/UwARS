import numpy as np
import random
import time
import copy
import pickle
import os
import cv2
from PIL import Image
from sklearn.cluster import KMeans

from keras.datasets import mnist, cifar10, fashion_mnist
from scipy.io import loadmat


def pHash_array(img_array):

    img_array = img_array.astype(np.uint8)

    if len(img_array.shape) == 2:
        gray = img_array
    elif len(img_array.shape) == 3 and img_array.shape[2] == 1:
        gray = img_array[:, :, 0]
    elif len(img_array.shape) == 3 and img_array.shape[2] == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError(f"Unsupported image shape: {img_array.shape}")

    gray = cv2.resize(gray, (32, 32))
    dct = cv2.dct(np.float32(gray))
    dct_roi = dct[0:10, 0:10]
    avreage = np.mean(dct_roi)
    hash = []
    for i in range(dct_roi.shape[0]):
        for j in range(dct_roi.shape[1]):
            if dct_roi[i, j] > avreage:
                hash.append(1)
            else:
                hash.append(0)
    return hash


def load_test_mnist():
    (_, _), (x_test, _) = mnist.load_data()
    return x_test  # (10000, 28, 28), uint8, [0, 255]


def load_test_fashion_mnist():
    (_, _), (x_test, _) = fashion_mnist.load_data()
    return x_test  # (10000, 28, 28), uint8, [0, 255]


def load_test_cifar10():
    (_, _), (x_test, _) = cifar10.load_data()
    return x_test  # (10000, 32, 32, 3), uint8, [0, 255]


def load_test_svhn(data_path):
    candidates = [
        os.path.join(data_path, "svhn", "test_32x32.mat"),
        os.path.join(data_path, "SVHN", "test_32x32.mat"),
        os.path.join(data_path, "Data", "test_32x32.mat"),
    ]
    for mat_path in candidates:
        if os.path.exists(mat_path):
            test_raw = loadmat(mat_path)
            x_test = np.array(test_raw['X'])
            x_test = np.moveaxis(x_test, -1, 0)
            return x_test  # (26032, 32, 32, 3), uint8, [0, 255]
    raise FileNotFoundError(f"Could not find the SVHN test set directory!")


def load_test_fruit360(data_path):
    candidates = [
        os.path.join(data_path, "fruits-360", "Test"),
        os.path.join(data_path, "fruits-360", "test"),
        os.path.join(os.path.dirname(data_path), "dataset", "fruits-360", "Test"),
        os.path.join(os.path.dirname(data_path), "dataset", "fruits-360", "test"),
    ]
    test_dir = None
    for c in candidates:
        if os.path.isdir(c):
            test_dir = c
            break
    if test_dir is None:
        raise FileNotFoundError(f"Could not find the Fruit-360 test set directory!")

    print(f"    Fruit-360: {test_dir}", flush=True)

    class_names = sorted([d for d in os.listdir(test_dir)
                          if os.path.isdir(os.path.join(test_dir, d))])

    images = []
    count = 0
    for class_name in class_names:
        class_path = os.path.join(test_dir, class_name)
        img_files = sorted([f for f in os.listdir(class_path)
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        for img_file in img_files:
            img_path = os.path.join(class_path, img_file)
            img = Image.open(img_path).convert('RGB')
            img = img.resize((100, 100))
            images.append(np.array(img, dtype=np.uint8))
            count += 1
            if count % 2000 == 0:
                print(f"      Fruit-360: {count}/23619", flush=True)

    return np.array(images)


def load_test_tinyimagenet(data_path):
    candidate_paths = [
        os.path.join(data_path, "Retrain", "TinyImageNet_ResNet101", "tiny_x_test.npy"),
        os.path.join(os.path.dirname(data_path), "Input_data", "Retrain",
                     "TinyImageNet_ResNet101", "tiny_x_test.npy"),
        os.path.join(data_path, "tiny_x_test.npy"),
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            return np.load(p)  # (10000, 64, 64, 3), uint8, [0, 255]
    raise FileNotFoundError(f"Error!")


def load_test_data(data_name, data_path):
    data_name = data_name.lower().strip()

    if data_name == "mnist":
        return load_test_mnist()
    elif data_name == "fashion_mnist":
        return load_test_fashion_mnist()
    elif data_name == "cifar10":
        return load_test_cifar10()
    elif data_name == "svhn":
        return load_test_svhn(data_path)
    elif data_name == "fruit360":
        return load_test_fruit360(data_path)
    elif data_name == "tinyimagenet":
        return load_test_tinyimagenet(data_path)
    else:
        raise ValueError(f"Unsupported datasets: {data_name}")



def verify_tiny_alignment(output_probability, mis_ids_test, author_data_path):
    y_path = os.path.join(author_data_path, "Retrain",
                          "TinyImageNet_ResNet101", "tiny_y_test.npy")
    if not os.path.exists(y_path):
        print(f"Error!")
        return None
    y = np.load(y_path)
    pred = np.argmax(output_probability, axis=1)
    recomputed = set(np.where(pred != y)[0])
    author = set(mis_ids_test)
    ratio = len(recomputed & author) / max(len(author), 1)
    if ratio < 0.99:
        print("Error!")
    return ratio


def preprocessing(x_test, k):
    phash_list = []
    for i in range(len(x_test)):
        phash_list.append(pHash_array(x_test[i]))

    feature_array = np.array(phash_list, dtype=np.float32)
    kmeans = KMeans(n_clusters=k, n_init=10)
    cluster_list = kmeans.fit_predict(feature_array).tolist()

    return phash_list, cluster_list


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


def compare_hash(hash1, hash2):
    max_possible = len(hash1)
    hash1 = np.array(hash1)
    hash2 = np.array(hash2)
    manhattan_dist = np.sum(np.abs(hash1 - hash2))
    normalized_dist = manhattan_dist / max_possible
    return normalized_dist

# This function is reused from the official replication package of SETS:
# https://github.com/GIST-NJU/SETS
def gini_score(Output_probability):
    gini_scores = []
    for i in range(len(Output_probability)):
        sum = 0
        for j in range(len(Output_probability[0])):
            sum += Output_probability[i][j] ** 2
        gini_scores.append(1 - sum)
    return gini_scores

# This function is reused from the official replication package of SETS:
# https://github.com/GIST-NJU/SETS
def maxp_score(Output_probability):
    return [1 - max(prob) for prob in Output_probability]


# ========================== UwARS ==========================

def uwars(size, index, x_test, output_probability, uncertainty, a, k,
            precomputed=None):

    if precomputed is not None:
        phash_list, cluster_list = precomputed
    else:
        phash_list, cluster_list = preprocessing(x_test, k)

    start_time = time.time()
    selected_list = []
    selected_cluster_list = []

    if uncertainty == "gini":
        un_scores = gini_score(output_probability)
    elif uncertainty == "maxp":
        un_scores = maxp_score(output_probability)
    else:
        raise ValueError(f"Error!: {uncertainty}")

    sorted_indices = sorted(index, key=lambda i: un_scores[i], reverse=True)

    top_percent_count = max(1, int(a * size))  # reduction
    if a * size > len(index):
        top_percent_count = len(index)
    filtered_indices = sorted_indices[:top_percent_count]

    selected_first = filtered_indices[0]
    selected_list.append(selected_first)
    selected_cluster_list.append(cluster_list[selected_first])
    filtered_indices.remove(selected_first)

    while len(selected_list) < size:
        best_fitness = -1.0
        best_condidate = -1
        condidate_list = random.sample(filtered_indices, min(10, len(filtered_indices)))
        new_selected = False

        for condidates in condidate_list:
            if cluster_list[condidates] not in selected_cluster_list:
                selected_cluster_list.append(cluster_list[condidates])
                selected_list.append(condidates)
                filtered_indices.remove(condidates)
                new_selected = True
                break
            else:
                temp_selected_list = []
                cluster_label = cluster_list[condidates]
                for selected in selected_list:
                    if cluster_list[selected] == cluster_label:
                        temp_selected_list.append(selected)

                distance = 0
                for temp_selected in temp_selected_list:
                    distance += compare_hash(phash_list[condidates], phash_list[temp_selected])

                mean_distance = distance / len(temp_selected_list)

                fitness = mean_distance * un_scores[condidates]

                if fitness >= best_fitness:
                    best_fitness = fitness
                    best_condidate = condidates

        if not new_selected and best_condidate != -1:
            selected_list.append(best_condidate)
            filtered_indices.remove(best_condidate)

    return selected_list, time.time() - start_time


# ========================== Main ==========================

if __name__ == "__main__":

    a = 2
    uncertainty = "maxp"
    n = 30

    data_model_pairs = [
        ("mnist", "LeNet1"),
        ("mnist", "LeNet5"),
        ("cifar10", "12Conv"),
        ("cifar10", "ResNet20"),
        ("Fashion_mnist", "LeNet4"),
        ("SVHN", "LeNet5"),
        ("Fruit360", "ResNet50"),
        ("TinyImageNet", "ResNet101"),
    ]

    DATASET_CLASSES = {
        "mnist": 10,
        "Fashion_mnist": 10,
        "cifar10": 10,
        "SVHN": 10,
        "Fruit360": 141,
        "TinyImageNet": 200,
    }

    # ==================== Path Configuration ====================
    data_path = r"D:\Python_project\SETS-main\SETS-main\dataset"
    author_data_path = r"D:\Python_project\SETS-main\SETS-main\Input_data"
    fault_clusters_path = os.path.join(author_data_path, "Fault_clusters")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    output_path = os.path.join(root_dir, "results")
    os.makedirs(output_path, exist_ok=True)

    for data_name, model_name in data_model_pairs:
        print(f"\n{'=' * 50}")
        print(f"Dataset: {data_name}, Model: {model_name}")
        print(f"{'=' * 50}")

        x_test = load_test_data(data_name, data_path)

        subdir = f"{data_name}_{model_name}"
        output_probability = np.load(
            os.path.join(fault_clusters_path, subdir, "output_probability.npy"))

        if data_name == "TinyImageNet":
            with open(os.path.join(fault_clusters_path, subdir,
                                   "cluster_results.pkl"), 'rb') as f:
                Clustering_labels = pickle.load(f)
            with open(os.path.join(fault_clusters_path, subdir,
                                   "mis_index_test.pkl"), 'rb') as f:
                val_mis_data = pickle.load(f)
            mis_ids_test = [item[2] for item in val_mis_data]
            # verify_tiny_alignment(output_probability, mis_ids_test, author_data_path)
        else:
            Clustering_labels = np.load(
                os.path.join(fault_clusters_path, subdir, "cluster_results.npy"))
            mis_ids_test = np.load(
                os.path.join(fault_clusters_path, subdir, "mis_index_test.npy"))

        if data_name == "Fruit360":
            if len(mis_ids_test.shape) > 1:
                mis_ids_test = mis_ids_test[0]

        total_faults = len(set(Clustering_labels)) - 1

        noisy_index = []
        for i in range(len(mis_ids_test)):
            if Clustering_labels[i] == -1:
                noisy_index.append(mis_ids_test[i])
        index_withoutnoisy = sorted(
            set(range(0, len(output_probability))) - set(noisy_index))

        num_classes = DATASET_CLASSES.get(data_name, 10)
        k = int(num_classes)
        prep_start = time.time()
        phash_list, cluster_list = preprocessing(x_test, k)
        prep_time = time.time() - prep_start
        print(f"  Preprocessing time: {prep_time}")

        for size in [100, 300, 500, 700, 900, 1100, 1300, 1500]:
            fault_list = []
            time_list = []
            fdr_list = []

            # max a
            # test_size = len(x_test)
            # k = int(num_classes)
            # a = test_size / k

            for i in range(int(n)):
                selected_subset, exe_time = uwars(
                    size, list(index_withoutnoisy), x_test,
                    output_probability, uncertainty, a, k,
                    precomputed=(phash_list, cluster_list))

                _, find_faults, _ = faults(selected_subset, mis_ids_test,
                                           Clustering_labels)
                fdr = find_faults / min(size, total_faults)

                fault_list.append(find_faults)
                time_list.append(exe_time)
                fdr_list.append(fdr)

            avg_fdr = sum(fdr_list) / len(fdr_list)
            avg_time = sum(time_list) / len(time_list)

            print(f"{data_name}_{model_name};{size};{k};{a};"
                  f"{avg_fdr};{avg_time};prep_time={prep_time}")

            filename = os.path.join(
                output_path, f"UwARS_{data_name}_{model_name}_{size}.txt")
            with open(filename, 'w') as file:
                file.write("Fault List:\n")
                file.write("\n".join(map(str, fault_list)) + "\n")
                file.write("FDR List:\n")
                file.write("\n".join(map(str, fdr_list)) + "\n")
                file.write("Time List:\n")
                file.write("\n".join(map(str, time_list)) + "\n")
                file.write(f"Preprocessing Time: {prep_time:.2f}\n")
            print(f"Saved file: {filename}")