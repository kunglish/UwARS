import pickle
import os
from tensorflow.keras.utils import to_categorical
import copy
from keras.models import load_model, Model
from scipy.io import loadmat
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from keras.datasets import mnist, cifar10, fashion_mnist, cifar100
import sys
sys.path.append('..')
import random
import numpy as np
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.utils import shuffle

# This function is reused from the official replication package of SETS:
# https://github.com/GIST-NJU/SETS
def scale(X, x_min, x_max):
    nom = (X - X.min(axis=0)) * (x_max - x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom == 0] = 1
    return x_min + nom / denom


def dataset(arg, model_name):
    CLIP_MIN = -0.5
    CLIP_MAX = 0.5
    flag = True

    if arg == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        x_train = x_train.astype("float32")
        x_test = x_test.astype("float32")
        # ##Model
        if model_name == "LeNet1":
            model = load_model(str(data_path) + "/Pretrained_model/model_" + str(arg) + "_" + str(model_name) + ".h5")
        if model_name == "LeNet5":
            model = load_model(str(data_path) + "/Pretrained_model/model_" + str(arg) + "_" + str(model_name) + ".h5")

        y_test = to_categorical(y_test, 10)
        y_test = np.argmax(y_test, axis=1)
        y_train = to_categorical(y_train, 10)

    if arg == "Fashion_mnist":
        # load dataset
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        x_train = x_train.astype("float32")
        x_test = x_test.astype("float32")
        if model_name == "LeNet4":
            model = load_model(str(data_path) + "/Pretrained_model/model_" + str(arg) + "_" + str(model_name) + ".h5")
            y_test = to_categorical(y_test, 10)
            y_test = np.argmax(y_test, axis=1)
            y_train = to_categorical(y_train, 10)

    if arg == "SVHN":
        train_raw = loadmat(str(data_path) + '/Data/train_32x32.mat')
        test_raw = loadmat(str(data_path) + '/Data/test_32x32.mat')
        x_train = np.array(train_raw['X'])
        x_test = np.array(test_raw['X'])
        y_train = train_raw['y']
        y_test = test_raw['y']
        x_train = np.moveaxis(x_train, -1, 0)
        x_test = np.moveaxis(x_test, -1, 0)
        x_test = x_test.reshape(-1, 32, 32, 3)
        x_train = x_train.reshape(-1, 32, 32, 3)
        x_train = x_train.astype("float32")
        x_test = x_test.astype("float32")
        if model_name == "LeNet5":
            model = load_model(str(data_path) + "/Pretrained_model/model_" + str(arg) + "_" + str(model_name) + ".h5")
            lb = LabelBinarizer()
            y_train = lb.fit_transform(y_train)
            y_test = lb.fit_transform(y_test)
            y_test = np.argmax(y_test, axis=1)

    if arg == "cifar10":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype("float32")
        x_test = x_test.astype("float32")
        if model_name == "12Conv":
            model = load_model(str(data_path) + "/Pretrained_model/model_" + str(arg) + "_" + str(model_name) + ".h5")
        if model_name == "ResNet20":
            model = load_model(str(data_path) + "/Pretrained_model/model_" + str(arg) + "_" + str(model_name) + ".h5")

        y_test = to_categorical(y_test, 10)
        y_test = np.argmax(y_test, axis=1)
        y_train = to_categorical(y_train, 10)

    if arg == "fruit360":
        x_train = np.load(str(data_path) + "/Fruit360_ResNet50/fruit_x_train_origin.npy")
        y_train = np.load(str(data_path) + "/Fruit360_ResNet50/fruit_y_train.npy")
        x_test = np.load(str(data_path) + "/Fruit360_ResNet50/fruit_x_test_origin.npy")
        y_test = np.load(str(data_path) + "/Fruit360_ResNet50/fruit_y_test.npy")
        model = load_model(str(data_path) + "/Fruit360_ResNet50/fruit_resnet2.h5")
        flag = False

    if flag:
        x_train = (x_train / 255.0) - (1.0 - CLIP_MAX)
        x_test = (x_test / 255.0) - (1.0 - CLIP_MAX)

    return x_train, y_train, x_test, y_test, model


data_model_pairs = [
    ("mnist", "LeNet1"),
    ("mnist", "LeNet5"),
    ("cifar10", "12Conv"),
    ("cifar10", "ResNet20"),
    ("Fashion_mnist", "LeNet4"),
    ("SVHN", "LeNet5"),
    # ("fruit360","resnet50"),
    # ("tinyimagenet","resnet101")
]

# model retrain parameters all the same
opt = tf.keras.optimizers.legacy.Adadelta(learning_rate=0.005)
# opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.00001)
Epi = 0
final = []
ep = 30
bat = 50
optim = opt
Ac = []
indx = 0

classes = 10

if __name__ == '__main__':
    import sys

    output_path = sys.argv[1]
    data_path = sys.argv[2]
    selcetion_method = sys.argv[3]

    for data_name, model_name in data_model_pairs:
        print(f"Dataset: {data_name}, Model: {model_name}, Method: {selcetion_method}")
        x_train, y_train, x_test, y_test, model = dataset(data_name, model_name)

        # print(y_test)
        t_file = f"{data_path}/Retrain/{data_name}_{model_name}.pkl"
        with open(t_file, 'rb') as f:
            t_indices = pickle.load(f)

        # v_set = list(set(range(10000)) - set(t_indices))
        v_set = list(set(range(len(x_test))) - set(t_indices))
        v_test = x_test[v_set]
        vy_test = y_test[v_set]


        if data_name == "fruit360":
            classes = 141
        if data_name == "tinyimagenet":
            classes = 200

        # y_pred = model(x_test) #first 6 subject
        y_pred = model(v_test)
        # y_pred = model.predict(x_test)
        y_pred = np.argmax(y_pred, axis=1)

        # f1_score(y_test, y_pred, average="weighted")

        acc_ori = accuracy_score(vy_test, y_pred)
        print(f"Original Model Accuracy: {acc_ori:.4f}")
        y_test1 = to_categorical(y_test, classes)
        # y_train1 = to_categorical(y_train,classes) #6 subject no need

        x_tr1 = copy.deepcopy(x_train)
        y_tr1 = copy.deepcopy(y_train)
        # y_tr1 = copy.deepcopy(y_train1)

        print("Start Retraining!")

        for size in [500]:
            acc_re_ets_list = []
            acc_imp_ets_list = []
            for i in range(5):
                if selcetion_method == "SETS":
                    file_name = str(data_path) + f"/Retrain/SETS/{data_name}_{model_name}_{size}.pkl"
                else:
                    file_name = str(
                        data_path) + f"/Retrain/{selcetion_method}/{data_name}_{model_name}_{size}_seed{i}.pkl"
                with open(file_name, 'rb') as f:
                    sub_ets = np.array(pickle.load(f), dtype=np.int32)

                x_newtr_ets = np.concatenate((x_tr1, x_test[sub_ets]), axis=0)
                # y_newtr_ets = np.concatenate((y_tr1, y_test1[subset]), axis=0)
                y_newtr_ets = np.concatenate((y_tr1, y_test1[sub_ets]), axis=0)

                model = load_model(
                    str(data_path) + "/Pretrained_model/model_" + str(data_name) + "_" + str(model_name) + ".h5")
                model.compile(optimizer=optim, loss='categorical_crossentropy')
                # model.compile(optimizer=optim, loss='sparse_categorical_crossentropy')

                x, y = shuffle(x_newtr_ets, y_newtr_ets)
                v_id = random.sample(range(len(x_test)), 2500)

                model.fit(x, y, epochs=ep, batch_size=bat, validation_data=(x_test[v_id], y_test1[v_id]), verbose=0)
                # model.fit(x, y, epochs=ep, batch_size=bat, validation_data=(x_test[v_id], y_test[v_id]), verbose=0)
                # model.fit(x, y, epochs=ep, batch_size=bat, validation_data=(x_test, y_test1), verbose=0)
                ypre1 = model(v_test)
                # ypre1 = model.predict(x_test)
                ypre1 = np.argmax(ypre1, axis=1)
                acc_re_ets = accuracy_score(vy_test, ypre1)
                acc_re_ets_list.append(acc_re_ets)
                print(f"Retrained Model Accuracy {selcetion_method}: {acc_re_ets:.4f}")
                acc_imp_ets = acc_re_ets - acc_ori
                acc_imp_ets_list.append(acc_imp_ets)

            print(f"average accuracy imp {selcetion_method} {size}: {(sum(acc_imp_ets_list) / len(acc_imp_ets_list)):.4f}")

            file_path = str(output_path) + f"/new_retrain_result/{data_name}_{model_name}_{selcetion_method}_{size}.txt"

            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(file_path, "w") as f:
                f.write(f"Original Model Accuracy: {acc_ori:.4f}\n")


                f.write(f"{selcetion_method} Retraining\n")
                f.write("Acc Re List:\n")
                f.write("\n".join([f"{x:.4f}" for x in acc_re_ets_list]) + "\n")

                f.write("Acc Imp List:\n")
                f.write("\n".join([f"{x:.4f}" for x in acc_imp_ets_list]) + "\n")

            print("save successfully!")



