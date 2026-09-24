from tensorflow.keras.utils import to_categorical
from keras.models import load_model, Model
from scipy.io import loadmat
from sklearn.preprocessing import LabelBinarizer
from keras import layers
from tensorflow.keras.applications.vgg16 import VGG16
from keras.datasets import mnist, cifar10 , fashion_mnist, cifar100
import sys
sys.path.append('..')
import time
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img




# A small demo is provided to illustrate how to call the function.
# You need to adapt the code based on how the model and dataset are loaded
# You can also refer to the original source of these functions in the official reproduction package of DeepGD.

# This function is reused from the official replication package of SETS:
# https://github.com/GIST-NJU/SETS


def vgg16_features_GD(arg):

  CLIP_MIN = -0.5
  CLIP_MAX = 0.5

  if (arg=="cifar10" or arg=="cifar100" or arg=="SVHN"):
    if(arg=="cifar10"):
      (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    if(arg=="cifar100"):
      (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    if (arg=="SVHN"):
      train_raw = loadmat(str(data_path)+'/Images/SVHN/train_32x32.mat')
      test_raw = loadmat(str(data_path)+'/Images/SVHN/test_32x32.mat')
      x_train = np.array(train_raw['X'])
      x_test = np.array(test_raw['X'])
      y_train = train_raw['y']
      y_test = test_raw['y']
      x_train = np.moveaxis(x_train, -1, 0)
      x_test = np.moveaxis(x_test, -1, 0)
      # lb = LabelBinarizer()
      # train_labels = lb.fit_transform(train_labels)
      # test_labels = lb.fit_transform(test_labels)

    x_test1= x_test.reshape (-1,32,32,3)

  if (arg =="mnist" or arg=="Fashion_mnist"):
    if (arg=="mnist"):
      (x_train, y_train), (x_test, y_test) = mnist.load_data()
    if(arg=="Fashion_mnist"):
      (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_test1=np.dstack([x_test]*3)
    x_test1= x_test1.reshape(-1,28,28,3)
    #Resize the images 48*48 as required by VGG16

  x_test1 = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in x_test1])

  x_test1 = x_test1.astype("float32")
  x_test1 = (x_test1 / 255.0) - (1.0 - CLIP_MAX)
  input_layer=layers.Input(shape=(48,48,3))
  model_vgg16=VGG16(weights='imagenet',input_tensor=input_layer,include_top=False)
  model_vgg16.summary()
  base_model = model_vgg16
  name_layer = 'block5_conv3'
  intermediate_layer_model = Model(inputs=base_model.input, outputs=base_model.get_layer(name_layer).output)
  FF = intermediate_layer_model.predict(x_test1)
  features= FF.reshape((len(x_test1),9*512))
  nom = (features-features.min(axis=0))*(1-0)
  denom = features.max(axis=0) - features.min(axis=0)
  denom[denom==0] = 1
  X_scf = nom/denom
  #print(X_scf)
  #print("rank of feature matrix", np.linalg.matrix_rank(X_scf))
  return features, X_scf

# Your_google_drive_path = 'Input_data'
Your_google_drive_path = '../Input_data'
data_path = Your_google_drive_path


# This function is reused from the official replication package of SETS:
# https://github.com/GIST-NJU/SETS
def dataset(arg, model_name):
  CLIP_MIN = -0.5
  CLIP_MAX = 0.5

  if arg=="mnist":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    # ##Model
    if model_name=="LeNet1":
        model =  load_model(str(Your_google_drive_path)+"/Pretrained_model/model_"+str(arg)+"_"+str(model_name)+".h5")
    if model_name=="LeNet5":
        model =  load_model(str(Your_google_drive_path)+"/Pretrained_model/model_"+str(arg)+"_"+str(model_name)+".h5")
    ##VGG feature extaction (4068)
    ##rank of features (2476)
    #You can use the stored verion or call vgg function to extract features(to do so comment out the second line)
    # MNIST_VGG=np.load(str(Your_google_drive_path)+"/Extracted Features/MNIST/block5_conv3_3_3_512.npy")
    _,MNIST_VGG=vgg16_features_GD("mnist")
    features_vgg=MNIST_VGG
    y_test = to_categorical(y_test, 10)
    y_test=np.argmax(y_test, axis=1)
    y_train = to_categorical(y_train, 10)

  if arg=="Fashion_mnist":
    # load dataset
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    _,features_vgg=vgg16_features_GD("Fashion_mnist")
    if model_name=="LeNet4":
      model =  load_model(str(Your_google_drive_path)+"/Pretrained_model/model_"+str(arg)+"_"+str(model_name)+".h5")
      y_test = to_categorical(y_test, 10)
      y_test=np.argmax(y_test, axis=1)
      y_train = to_categorical(y_train, 10)


  if arg=="SVHN":
    train_raw = loadmat(str(Your_google_drive_path)+'/Images/SVHN/train_32x32.mat')
    test_raw = loadmat(str(Your_google_drive_path)+'/Images/SVHN/test_32x32.mat')
    x_train = np.array(train_raw['X'])
    x_test = np.array(test_raw['X'])
    y_train = train_raw['y']
    y_test = test_raw['y']
    x_train = np.moveaxis(x_train, -1, 0)
    x_test = np.moveaxis(x_test, -1, 0)
    x_test= x_test.reshape (-1,32,32,3)
    x_train= x_train.reshape (-1,32,32,3)
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    _,features_vgg=vgg16_features_GD("SVHN")
    if model_name=="LeNet5":
      model =  load_model(str(Your_google_drive_path)+"/Pretrained_model/model_"+str(arg)+"_"+str(model_name)+".h5")
      lb = LabelBinarizer()
      y_train = lb.fit_transform(y_train)
      y_test = lb.fit_transform(y_test)
      y_test=np.argmax(y_test, axis=1)

  if arg=="cifar10":
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    if model_name=="12Conv":
      model =  load_model(str(Your_google_drive_path)+"/Pretrained_model/model_"+str(arg)+"_"+str(model_name)+".h5")
    if model_name=="ResNet20":
      model =  load_model(str(Your_google_drive_path)+"/Pretrained_model/model_"+str(arg)+"_"+str(model_name)+".h5")

    #VGG feature extaction (4068)
    #rank of features (3845)
    #You can use the stored verion or call vgg function to extract features (comment out the second line)
    # Cifar_VGG=np.load(str(Your_google_drive_path)+"/Extracted Features/Cifar10/x_cifar_inputshape48_block5_conv3.npy")
    _,Cifar_VGG=vgg16_features_GD("cifar10")
    features_vgg=Cifar_VGG

    y_test = to_categorical(y_test, 10)
    y_test=np.argmax(y_test, axis=1)
    y_train = to_categorical(y_train, 10)

  x_train = (x_train / 255.0) - (1.0 - CLIP_MAX)
  x_test = (x_test / 255.0) - (1.0 - CLIP_MAX)

  return x_train, y_train, x_test, y_test,features_vgg , model





# a demo
# data_name = "mnist"
# model_name = "LeNet1"
#
#
# x_train, y_train, x_test, y_test, features, model= dataset(data_name, model_name)
#
# s_time = time.time()
# Output_probability=model.predict(x_test) # you can get the probability
# e_time = time.time()
# prediction_time = e_time - s_time
