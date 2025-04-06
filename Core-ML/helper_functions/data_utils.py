from __future__ import print_function
import torch
from torch.utils.data import TensorDataset, DataLoader

from builtins import range
from six.moves import cPickle as pickle
import numpy as np
import os
from imageio import imread
import platform




from torch.utils.data import DataLoader, TensorDataset

import torch
from torch.utils.data import DataLoader, TensorDataset

import torch
from torch.utils.data import DataLoader, TensorDataset

from torch.utils.data import DataLoader, TensorDataset
import torch

def get_custom_loaders(data_dict, batch_size):
    """Create DataLoaders from small_data (train and val data)."""
    X_train, y_train = data_dict["X_train"], data_dict["y_train"]
    X_val, y_val = data_dict["X_val"], data_dict["y_val"]

    # ✅ Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    # ✅ Create TensorDatasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    # ✅ Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader





def get_cifar10_loaders(batch_size=128):
    """Create PyTorch DataLoaders for CIFAR-10 dataset."""
    data = get_CIFAR10_data()
    
    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(data["X_train"], dtype=torch.float32)
    y_train_tensor = torch.tensor(data["y_train"], dtype=torch.long)
    X_test_tensor = torch.tensor(data["X_test"], dtype=torch.float32)
    y_test_tensor = torch.tensor(data["y_test"], dtype=torch.long)

    # Create TensorDatasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader

def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == "2":
        return pickle.load(f)
    elif version[0] == "3":
        return pickle.load(f, encoding="latin1")
    raise ValueError("invalid python version: {}".format(version))



def add_symmetric_noise(y, noise_rate=0.6, num_classes=10):
    """Add symmetric noise by flipping labels randomly."""
    n_samples = y.shape[0]
    n_noisy = int(noise_rate * n_samples)

    # Randomly select indices to corrupt
    noisy_indices = np.random.choice(n_samples, n_noisy, replace=False)
    new_labels = np.random.randint(num_classes, size=n_noisy)

    y_noisy = y.copy()
    y_noisy[noisy_indices] = new_labels

    return y_noisy

def get_CIFAR10_noisy_data(noise_rate=0.6):
    """Load CIFAR-10 data and add symmetric noise to training labels."""
    from helper_functions.data_utils import get_CIFAR10_data  # Import original function

    # ✅ Get original clean CIFAR-10 data
    data = get_CIFAR10_data()

    # ✅ Add symmetric noise to training labels if noise_rate > 0
    if noise_rate > 0.0:
        print(f"Adding symmetric noise with noise rate: {noise_rate}")
        data["y_train"] = add_symmetric_noise(data["y_train"], noise_rate=noise_rate)

    # 🚨 Check if X_test and y_test exist and add them manually if missing
    if "X_test" not in data or "y_test" not in data:
        data["X_test"] = data["X_val"]  # ✅ Use validation as test if missing
        data["y_test"] = data["y_val"]
        print("⚠️ Using validation set as test set by default.")

    return data



def load_CIFAR_batch(filename):
    """ Load a single batch of CIFAR-10 """
    with open(filename, "rb") as f:
        datadict = load_pickle(f)
        X = datadict["data"]
        Y = datadict["labels"]
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y


def load_CIFAR10(ROOT):
    """ Load all CIFAR-10 data """
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, "data_batch_%d" % (b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, "test_batch"))
    return Xtr, Ytr, Xte, Yte


def get_CIFAR10_data(
    num_training=49000, num_validation=1000, num_test=1000, subtract_mean=True
):
    """ Load and preprocess CIFAR-10 for training, validation, and testing """
    # Define CIFAR-10 data path in Google Drive
    cifar10_dir = "/content/drive/MyDrive/SAIDL/core ML/data/cifar-10-batches-py"

    # Load CIFAR-10 data
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize data: subtract mean image
    if subtract_mean:
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image

    # Transpose to have channel-first format
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    # Package data into a dictionary
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }



def load_models(models_dir):
    """ Load saved models from disk """
    models = {}
    for model_file in os.listdir(models_dir):
        with open(os.path.join(models_dir, model_file), "rb") as f:
            try:
                models[model_file] = load_pickle(f)["model"]
            except pickle.UnpicklingError:
                continue
    return models


def load_imagenet_val(num=None):
    """ Load a handful of validation images from ImageNet """
    imagenet_fn = os.path.join(os.getcwd(), "datasets/imagenet_val_25.npz")
    if not os.path.isfile(imagenet_fn):
        print("File %s not found" % imagenet_fn)
        print("Run the following:")
        print("cd cs231n/datasets")
        print("bash get_imagenet_val.sh")
        raise FileNotFoundError("Need to download imagenet_val_25.npz")

    # Modify the default parameters of np.load to allow pickle
    np_load_old = np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
    f = np.load(imagenet_fn)
    np.load = np_load_old

    X = f["X"]
    y = f["y"]
    class_names = f["label_map"].item()
    if num is not None:
        X = X[:num]
        y = y[:num]
    return X, y, class_names


def load_tiny_imagenet(path, dtype=np.float32, subtract_mean=True):
    """ Load TinyImageNet dataset """
    # Load wnids.txt to get class IDs
    with open(os.path.join(path, "wnids.txt"), "r") as f:
        wnids = [x.strip() for x in f]

    wnid_to_label = {wnid: i for i, wnid in enumerate(wnids)}

    # Load words.txt for class names
    with open(os.path.join(path, "words.txt"), "r") as f:
        wnid_to_words = dict(line.split("\t") for line in f)
        for wnid, words in wnid_to_words.items():
            wnid_to_words[wnid] = [w.strip() for w in words.split(",")]
    class_names = [wnid_to_words[wnid] for wnid in wnids]

    # Load training data
    X_train = []
    y_train = []
    for i, wnid in enumerate(wnids):
        if (i + 1) % 20 == 0:
            print("Loading training data for synset %d / %d" % (i + 1, len(wnids)))
        boxes_file = os.path.join(path, "train", wnid, "%s_boxes.txt" % wnid)
        with open(boxes_file, "r") as f:
            filenames = [x.split("\t")[0] for x in f]
        num_images = len(filenames)

        X_train_block = np.zeros((num_images, 3, 64, 64), dtype=dtype)
        y_train_block = wnid_to_label[wnid] * np.ones(num_images, dtype=np.int64)
        for j, img_file in enumerate(filenames):
            img_file = os.path.join(path, "train", wnid, "images", img_file)
            img = imread(img_file)
            if img.ndim == 2:
                img.shape = (64, 64, 1)
            X_train_block[j] = img.transpose(2, 0, 1)
        X_train.append(X_train_block)
        y_train.append(y_train_block)

    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    # Load validation data
    with open(os.path.join(path, "val", "val_annotations.txt"), "r") as f:
        img_files = []
        val_wnids = []
        for line in f:
            img_file, wnid = line.split("\t")[:2]
            img_files.append(img_file)
            val_wnids.append(wnid)
        num_val = len(img_files)
        y_val = np.array([wnid_to_label[wnid] for wnid in val_wnids])
        X_val = np.zeros((num_val, 3, 64, 64), dtype=dtype)
        for i, img_file in enumerate(img_files):
            img_file = os.path.join(path, "val", "images", img_file)
            img = imread(img_file)
            if img.ndim == 2:
                img.shape = (64, 64, 1)
            X_val[i] = img.transpose(2, 0, 1)

    # Load test data
    img_files = os.listdir(os.path.join(path, "test", "images"))
    X_test = np.zeros((len(img_files), 3, 64, 64), dtype=dtype)
    for i, img_file in enumerate(img_files):
        img_file = os.path.join(path, "test", "images", img_file)
        img = imread(img_file)
        if img.ndim == 2:
            img.shape = (64, 64, 1)
        X_test[i] = img.transpose(2, 0, 1)

    y_test = None
    y_test_file = os.path.join(path, "test", "test_annotations.txt")
    if os.path.isfile(y_test_file):
        with open(y_test_file, "r") as f:
            img_file_to_wnid = {}
            for line in f:
                line = line.split("\t")
                img_file_to_wnid[line[0]] = line[1]
        y_test = [wnid_to_label[img_file_to_wnid[img_file]] for img_file in img_files]
        y_test = np.array(y_test)

    # Subtract mean image
    if subtract_mean:
        mean_image = X_train.mean(axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "class_names": class_names,
    }
