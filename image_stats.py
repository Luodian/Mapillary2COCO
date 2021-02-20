from PIL import Image
import multiprocessing as mp
import numpy as np
import os
import pandas as pd


def get_image_data(dataset_root, dir_name, file_name):
    """

    Args:
        dataset_root: root directory of the dataset
        dir_name: directory of the image data
        file_name: image file name

    Returns: image data as an numpy array

    Example Usage:
    >>> get_image_data("/home/ec2-user/SageMaker", "data/samples/image", "_1Gn_xkw7sa_i9GU4mkxxQ.jpg")
    """
    image_path = "{}/{}/{}".format(dataset_root, dir_name, file_name)
    image = Image.open(image_path)
    image_data = np.array(image, dtype=np.uint8)
    return image_data


def compute_mean_std(image_data):
    """

    Args:
        image_data (numpy array)

    Returns:
        mean value of the image data
        if the data is m * n * 3 (RGB) then return a vector of 3
        if the data is m * n (grayscale) then return a single value
    """
    dim = image_data.shape
    if len(dim) == 3:
        r = image_data[:, :, 0]
        g = image_data[:, :, 1]
        b = image_data[:, :, 2]
        return list(map(np.mean, [r, g, b])), list(map(np.std, [r, g, b]))
    elif len(dim) == 2:
        return np.mean(image_data), np.std(image_data)
    else:
        raise ValueError("Given image's dimension is unexpected.")


def run_stat(dataset_root, dir_name, file_name):
    image_data = get_image_data(dataset_root, dir_name, file_name)
    mean, std = compute_mean_std(image_data)
    return mean, std


def run_stats_on_datasets(dataset_root, dir_name):
    """

    Args:
        dataset_root: root directory of the dataset
        dir_name: directory of the image data

    Returns:
        statistics of the image dataset, saved as a csv file

    Example Usage:
    >>> run_stats_on_datasets("/home/ec2-user/SageMaker", "data/samples/image")
    """
    path = "{}/{}".format(dataset_root, dir_name)
    files, means, stds = [], [], []

    for f in sorted(os.listdir(path)):
        if f.endswith("jpg"):
            mean, std = run_stat(dataset_root, dir_name, f)
            files.append(f)
            means.append(mean)
            stds.append(std)
    df = pd.DataFrame({"file_name": files, "mean": means, "std": stds})
    df.to_csv("stats.csv", index=False)
