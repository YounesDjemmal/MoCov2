# The dataset structure should be like this:
# cifar10/train/
#  L airplane/
#    L 10008_airplane.png
#    L ...
#  L automobile/
#  L bird/
#  L cat/
#  L deer/
#  L dog/
#  L frog/
#  L horse/
#  L ship/
#  L truck/
# path_to_train = "/datasets/cifar10/train/"
# path_to_test = "/datasets/cifar10/test/"

import os
import tarfile
import urllib.request
import pickle
from PIL import Image
import numpy as np

# Download the dataset
url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
filename = "cifar-10-python.tar.gz"
urllib.request.urlretrieve(url, filename)

# Extract the dataset
with tarfile.open(filename, 'r:gz') as tar:
    tar.extractall()

# Load each batch file
def load_batch(filename):
    with open(filename, 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')
    return data_dict

# Define the labels
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Create the required directory structure
os.makedirs("./datasets/cifar10/train", exist_ok=True)
os.makedirs("./datasets/cifar10/test", exist_ok=True)
for label in labels:
    os.makedirs(f"./datasets/cifar10/train/{label}", exist_ok=True)
    os.makedirs(f"./datasets/cifar10/test/{label}", exist_ok=True)

# Process the training set
for i in range(1, 6):
    data_dict = load_batch(f"cifar-10-batches-py/data_batch_{i}")
    data = data_dict[b'data']
    filenames = data_dict[b'filenames']
    batch_labels = data_dict[b'labels']
    for j in range(len(data)):
        img = Image.fromarray(np.transpose(np.reshape(data[j], (3, 32, 32)), (1, 2, 0)))
        img.save(f"./datasets/cifar10/train/{labels[batch_labels[j]]}/{filenames[j].decode()}")

# Process the test set
data_dict = load_batch("cifar-10-batches-py/test_batch")
data = data_dict[b'data']
filenames = data_dict[b'filenames']
batch_labels = data_dict[b'labels']
for j in range(len(data)):
    img = Image.fromarray(np.transpose(np.reshape(data[j], (3, 32, 32)), (1, 2, 0)))
    img.save(f"./datasets/cifar10/test/{labels[batch_labels[j]]}/{filenames[j].decode()}")