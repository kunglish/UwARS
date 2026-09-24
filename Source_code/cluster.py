# the clustering algorithm is reused from the official implementation provided by the authors of SETS:
# https://github.com/GIST-NJU/SETS


import time
import copy
import pandas as pd
import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt
import seaborn as sns
import hdbscan
import umap.umap_ as umap

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import layers, callbacks
from tensorflow.keras.applications.vgg16 import VGG16
from keras.models import load_model, Model
import random


def convert_to_numpy(dataset):
    images = []
    labels = []
    for batch in dataset:
        x_batch, y_batch = batch
        images.append(x_batch.numpy())
        labels.append(y_batch.numpy())

    images = np.concatenate(images, axis=0)
    labels = np.concatenate(labels, axis=0)

    return images, labels


def resize_and_normalize(images, target_size=(48, 48)):
    images_resized = tf.image.resize(images, target_size)
    images_normalized = images_resized / 255.0
    return images_normalized


# Load dataset
train_dir = 'fruits-360_dataset_100x100/fruits-360/Training'
test_dir = 'fruits-360_dataset_100x100/fruits-360/Test'

CLIP_MIN = -0.5
CLIP_MAX = 0.5

model = load_model('fruit_resnet2.h5')


train_ds = tf.keras.preprocessing.image_dataset_from_directory(train_dir, image_size=(100, 100), batch_size=32,
                                                               shuffle=False)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(test_dir, image_size=(100, 100), batch_size=32,
                                                              shuffle=False)

x_train_origin, y_train = convert_to_numpy(train_ds)
x_test_origin, y_test = convert_to_numpy(test_ds)

normalization_layer = tf.keras.layers.Rescaling(1. / 255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

x_train, _ = convert_to_numpy(train_ds)
x_test, _ = convert_to_numpy(test_ds)

# Resize to (48, 48) and normalize
x_train = resize_and_normalize(x_train, target_size=(48, 48))
x_test = resize_and_normalize(x_test, target_size=(48, 48))

# Feature extraction using VGG16
input_layer = layers.Input(shape=(48, 48, 3))
model_vgg = VGG16(weights='imagenet', input_tensor=input_layer, include_top=False)
name_layer = 'block5_conv3'
intermediate_layer_model = Model(inputs=model_vgg.input, outputs=model_vgg.get_layer(name_layer).output)

FF_test = intermediate_layer_model.predict(x_test)
features_test = FF_test.reshape((len(x_test), 9 * 512))

FF_train = intermediate_layer_model.predict(x_train)
features_train = FF_train.reshape((len(x_train), 9 * 512))

# features_test = np.vstack(features_test)
# features_train = np.vstack(features_train)

from tensorflow.keras.applications.resnet import preprocess_input
def scale_one(X):
    return (X - X.min()) / (X.max() - X.min())

batch_size = 64

def predict_in_batches(model, x_data, batch_size):
    n_samples = x_data.shape[0]
    Y_pred = []
    for i in range(0, n_samples, batch_size):
        end_index = min(i + batch_size, n_samples)
        x_batch = x_data[i:end_index]
        Y_pred_batch = model.predict(x_batch)
        Y_pred.append(Y_pred_batch)
    Y_pred = np.concatenate(Y_pred, axis=0)
    return Y_pred


#x_test_origin = preprocess_input(x_test_origin)
Y_pred_test_batches = model.predict(x_test_origin)
#Y_pred_test_batches = predict_in_batches(model, x_test_origin, batch_size)

Y_pred_test = np.argmax(Y_pred_test_batches, axis=1)
print(Y_pred_test)
YP_Scaled_test = scale_one(Y_pred_test)
YT_Scaled_test = scale_one(y_test)
mis_ids_test = np.where(YP_Scaled_test != YT_Scaled_test)

# mis_feature_test
mis_f_test = features_test[list(mis_ids_test)]
YP_mis_test = YP_Scaled_test[list(mis_ids_test)]
YT_mis_test = YT_Scaled_test[list(mis_ids_test)]


#x_train_origin = preprocess_input(x_train_origin)
#Y_pred_train_batches = predict_in_batches(model, x_train_origin, batch_size)
Y_pred_train_batches = model.predict(x_train_origin)

Y_pred_train = np.argmax(Y_pred_train_batches, axis=1)
print(Y_pred_train)
#y_train = np.argmax(y_train, axis=1)
YP_Scaled_train = scale_one(Y_pred_train)
YT_Scaled_train = scale_one(y_train)
mis_ids_train = np.where(YP_Scaled_train != YT_Scaled_train)

# mis_feature_train
mis_f_train = features_train[list(mis_ids_train)]
YP_mis_train = YP_Scaled_train[list(mis_ids_train)]
YT_mis_train = YT_Scaled_train[list(mis_ids_train)]

mis_f_test = np.squeeze(mis_f_test, axis=0)
mis_f_train = np.squeeze(mis_f_train, axis=0)

total_features = np.vstack([mis_f_test, mis_f_train])
total_YT = np.concatenate((YT_mis_test, YT_mis_train), axis=1)
total_YP = np.concatenate((YP_mis_test, YP_mis_train), axis=1)

bb, trace, hdbscan_in_umap, clustering_results = [], [], [], []
Sumn = 0
# The following values are hyperparameters that you can adjust to find the best clustering results.
# Since UMAP and HDBSCAN incorporate randomness in their algorithms, ensure that you save the final settings for your reproducibility. We have saved our clustering results in this repository.
total_YT = total_YT.T
total_YP = total_YP.T
for i, j in zip([500, 400, 300, 250], [450, 350, 250, 200]): #[500, 400, 300, 250], [450, 350, 250, 200]
    for k, o in zip([5, 10, 15, 20, 25], [3, 5, 10, 15, 20]): #[5, 10, 15, 20, 25], [3, 5, 10, 15, 20]
        for n_n in [0.03, 0.1, 0.25, 0.5]:
            fit = umap.UMAP(min_dist=n_n, n_components=i, n_neighbors=k)
            u1 = fit.fit_transform(total_features)
            fit = umap.UMAP(min_dist=0.1, n_components=j, n_neighbors=o)
            u = fit.fit_transform(u1)
            u = np.c_[u, total_YT, total_YP]
            print("UMAP output shape:", u.shape)

            clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
            labels = clusterer.fit_predict(u)
            silhouette_umap = sklearn.metrics.silhouette_score(u, labels)
            silhouette_features = sklearn.metrics.silhouette_score(total_features, labels)

            print("Silhouette Score UMAP:", silhouette_umap)
            print("Silhouette Score Features:", silhouette_features)

            if (silhouette_umap >= 0.1 or silhouette_features >= 0.1) and labels.max() + 2 >= 200:
                bb.append(labels)
                config = [i, j, k, o]
                trace.append([i, j, k, o, silhouette_umap, labels.max() + 2, list(labels).count(-1)])
                hdbscan_in_umap.append(u)
                Sumn += 1

                clustering_results.append({
                    "Number of Clusters": labels.max() + 1,
                    "Silhouette Score": silhouette_umap,
                    "Number of Noisy Inputs": list(labels).count(-1),
                    "Config": config
                })

                print(f"Iteration {Sumn}: Noisy labels count: {list(labels).count(-1)}")

# Save the results example:
# np.save("/content/drive/MyDrive/RQ_Con_factor/clustering/Cifar10_12Conv/Test_cluster_4068.npy", bb)
# np.save("/content/drive/MyDrive/RQ_Con_factor/clustering/Cifar10_12Conv/all_trace_4068.npy", trace)
# np.save("/content/drive/MyDrive/RQ_Con_factor/clustering/Cifar10_12Conv/umap_output_config7_4068.npy", np.array(hdbscan_in_umap[7]))

# Display clustering results in a table and select the one config clustering that has best Silhouette score
clustering_df = pd.DataFrame(clustering_results)
print(clustering_df)
