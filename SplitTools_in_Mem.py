"""
    This file shows how to load and use the dataset
"""

from __future__ import print_function

import json
import os

import numpy as np

# matplotlib.use('Agg')
import imageio
from PIL import Image
import multiprocessing as mp
import pycococreatortools
import datetime
import fnmatch

INFO = {
    "description": "Mapillary",
    "url": "https://github.com/waspinator/pycococreator",
    "version": "0.1.0",
    "year": 2018,
    "contributor": "Luodian",
    "date_created": datetime.datetime.utcnow().isoformat(" "),
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
    }
]

CATEGORIES = [
    {
        "id": 1,
        "name": "Bird",
        "supercategory": "animal--bird",
    },
    {
        "id": 2,
        "name": "Ground_Animal",
        "supercategory": "animal--ground-animal",
    },
    {
        "id": 3,
        "name": "Crosswalk_Plain",
        "supercategory": "construction--flat--crosswalk-plain",
    },
    {
        "id": 4,
        "name": "Person",
        "supercategory": "human",
    },
    {
        "id": 5,
        "name": "Bicyclist",
        "supercategory": "human--rider",
    },
    {
        "id": 6,
        "name": "Motorcyclist",
        "supercategory": "human--rider",
    },
    {
        "id": 7,
        "name": "Other_Rider",
        "supercategory": "human--rider",
    },
    {
        "id": 8,
        "name": "Lane_Marking_-_Crosswalk",
        "supercategory": "marking--crosswalk-zebra",
    },
    {
        "id": 9,
        "name": "Banner",
        "supercategory": "object",
    },
    {
        "id": 10,
        "name": "Bench",
        "supercategory": "object",
    },
    {
        "id": 11,
        "name": "Bike_Rack",
        "supercategory": "object",
    },
    {
        "id": 12,
        "name": "Billboard",
        "supercategory": "object",
    },
    {
        "id": 13,
        "name": "Catch_Basin",
        "supercategory": "object",
    },
    {
        "id": 14,
        "name": "CCTV_Camera",
        "supercategory": "object",
    },
    {
        "id": 15,
        "name": "Fire_Hydrant",
        "supercategory": "object",
    },
    {
        "id": 16,
        "name": "Junction_Box",
        "supercategory": "object",
    },
    {
        "id": 17,
        "name": "Mailbox",
        "supercategory": "object",
    },
    {
        "id": 18,
        "name": "Manhole",
        "supercategory": "object",
    },
    {
        "id": 19,
        "name": "Phone_Booth",
        "supercategory": "object",
    },
    {
        "id": 20,
        "name": "Street_Light",
        "supercategory": "object",
    },
    {
        "id": 21,
        "name": "Pole",
        "supercategory": "object",
    },
    {
        "id": 22,
        "name": "Traffic_Sign_Frame",
        "supercategory": "object",
    },
    {
        "id": 23,
        "name": "Utility_Pole",
        "supercategory": "object",
    },
    {
        "id": 24,
        "name": "Traffic_Light",
        "supercategory": "object",
    },
    {
        "id": 25,
        "name": "Traffic_Sign_(Back)",
        "supercategory": "object",
    },
    {
        "id": 26,
        "name": "Traffic_Sign_(Front)",
        "supercategory": "object",
    },
    {
        "id": 27,
        "name": "Trash_Can",
        "supercategory": "object",
    },
    {
        "id": 28,
        "name": "Bicycle",
        "supercategory": "object",
    },
    {
        "id": 29,
        "name": "Boat",
        "supercategory": "object",
    },
    {
        "id": 30,
        "name": "Bus",
        "supercategory": "object",
    },
    {
        "id": 31,
        "name": "Car",
        "supercategory": "object",
    },
    {
        "id": 32,
        "name": "Caravan",
        "supercategory": "object",
    },
    {
        "id": 33,
        "name": "Motorcycle",
        "supercategory": "object",
    },
    {
        "id": 34,
        "name": "Other_Vehicle",
        "supercategory": "object",
    },
    {
        "id": 35,
        "name": "Trailer",
        "supercategory": "object",
    },
    {
        "id": 36,
        "name": "Truck",
        "supercategory": "object",
    },
    {
        "id": 37,
        "name": "Wheeled_Slow",
        "supercategory": "object",
    },
]


def split_to_coco_creator(input_instance_array, labels):
    labelid_matrix_name = []

    label_image_info = np.array(input_instance_array / 256, dtype=np.uint8)

    instance_image_info = np.array(input_instance_array % 256, dtype=np.uint8)

    unique_label_info = np.unique(label_image_info)
    unique_instance_info = np.unique(instance_image_info)

    for label_id, label in enumerate(labels):

        if (label_id in (unique_label_info)) and (label["instances"] == True):

            each_label_array = np.zeros(
                (input_instance_array.shape[0], input_instance_array.shape[1]),
                dtype=np.uint8,
            )

            each_label_array[label_image_info == label_id] = 255

            for instance_id in range(256):
                if instance_id in unique_instance_info:
                    each_instance_array = np.zeros(
                        (input_instance_array.shape[0], input_instance_array.shape[1]),
                        dtype=np.uint8,
                    )

                    each_instance_array[instance_image_info == instance_id] = 255

                    final_instance_array = np.bitwise_and(
                        each_instance_array, each_label_array
                    )

                    if np.unique(final_instance_array).size == 2:
                        labelid_matrix_name.append(
                            {
                                "label_id": label_id,
                                "instance_id": instance_id,
                                "label_name": label["readable"],
                                "image": final_instance_array,
                            }
                        )

    return labelid_matrix_name


def convert_class_id(annotation_filename):
    class_id = 0
    if "Bird" in annotation_filename:
        class_id = 1
    elif "Ground Animal" in annotation_filename:
        class_id = 2
    elif "Crosswalk - Plain" in annotation_filename:
        class_id = 3
    elif "Person" in annotation_filename:
        class_id = 4
    elif "Bicyclist" in annotation_filename:
        class_id = 5
    elif "Motorcyclist" in annotation_filename:
        class_id = 6
    elif "Other Rider" in annotation_filename:
        class_id = 7
    elif "Lane Marking - Crosswalk" in annotation_filename:
        class_id = 8
    elif "Banner" in annotation_filename:
        class_id = 9
    elif "Bench" in annotation_filename:
        class_id = 10
    elif "Bike Rack" in annotation_filename:
        class_id = 11
    elif "Billboard" in annotation_filename:
        class_id = 12
    elif "Catch Basin" in annotation_filename:
        class_id = 13
    elif "CCTV Camera" in annotation_filename:
        class_id = 14
    elif "Fire Hydrant" in annotation_filename:
        class_id = 15
    elif "Junction Box" in annotation_filename:
        class_id = 16
    elif "Mailbox" in annotation_filename:
        class_id = 17
    elif "Manhole" in annotation_filename:
        class_id = 18
    elif "Phone Booth" in annotation_filename:
        class_id = 19
    elif "Street Light" in annotation_filename:
        class_id = 20
    elif "Pole" in annotation_filename:
        class_id = 21
    elif "Traffic Sign Frame" in annotation_filename:
        class_id = 22
    elif "Utility Pole" in annotation_filename:
        class_id = 23
    elif "Traffic Light" in annotation_filename:
        class_id = 24
    elif "Traffic Sign (Back)" in annotation_filename:
        class_id = 25
    elif "Traffic Sign (Front)" in annotation_filename:
        class_id = 26
    elif "Trash Can" in annotation_filename:
        class_id = 27
    elif "Bicycle" in annotation_filename:
        class_id = 28
    elif "Boat" in annotation_filename:
        class_id = 29
    elif "Bus" in annotation_filename:
        class_id = 30
    elif "Car" in annotation_filename:
        class_id = 31
    elif "Caravan" in annotation_filename:
        class_id = 32
    elif "Motorcycle" in annotation_filename:
        class_id = 33
    elif "Other Vehicle" in annotation_filename:
        class_id = 34
    elif "Trailer" in annotation_filename:
        class_id = 35
    elif "Truck" in annotation_filename:
        class_id = 36
    elif "Wheeled Slow" in annotation_filename:
        class_id = 37

    return class_id


def each_sub_proc(file_name, dir_name, dataset_root, image_id, labels, each_image_json):
    print("File name-{}-{}".format(file_name, image_id))
    file_name = file_name[:-4]
    instance_path = "../{}/instances/{}.png".format(dir_name, file_name)
    instance_image = Image.open(instance_path)
    instance_array = np.array(instance_image, dtype=np.uint16)
    image_label_instance_infomatrix = split_to_coco_creator(instance_array, labels)

    image_info = pycococreatortools.create_image_info(
        image_id, file_name + ".jpg", instance_image.size
    )
    each_image_json["images"].append(image_info)

    segmentation_id = 1
    for item in image_label_instance_infomatrix:
        class_id = convert_class_id(item["label_name"])
        category_info = {"id": class_id, "is_crowd": 0}
        binary_mask = item["image"]
        annotation_info = pycococreatortools.create_annotation_info(
            segmentation_id,
            image_id,
            category_info,
            binary_mask,
            instance_image.size,
            tolerance=2,
        )
        if annotation_info is not None:
            each_image_json["annotations"].append(annotation_info)
            segmentation_id = segmentation_id + 1

    if each_image_json["images"] is []:
        print("Image {} doesn't contain one image".format(image_id))
    save_path = "{}/{}/massive_annotations/image{}_info.json".format(
        dataset_root, dir_name, image_id
    )
    print("Saving to {}".format(save_path))
    with open(save_path, "w") as fp:
        json.dump(each_image_json, fp)


def load_datasets_and_proc(dataset_root, dir_name, files):
    np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
    each_image_json = dict({"images": [], "annotations": []})

    pool = mp.Pool(os.cpu_count())
    with open("./config.json") as config_file:
        config = json.load(config_file)

    labels = config["labels"]
    for idx, image_filename in enumerate(files):
        pool.apply_async(
            each_sub_proc,
            args=(
                image_filename,
                dir_name,
                dataset_root,
                idx + 1,
                labels,
                each_image_json,
            ),
        )

    pool.close()
    pool.join()


def readout_each_image(dataset_root, dir_name, seq):
    json_saved_path = "{}/mapillary/{}/massive_annotations/image{}_info.json".format(
        dataset_root, dir_name, seq
    )
    with open(json_saved_path) as fp:
        json_div = json.load(fp)

    return json_div


def main(dir_name, dataset_root, sample_type):
    dir_path = "../{}/instances".format(dir_name)
    files = os.listdir(dir_path)

    # dataset_root = "/home/boli/detectron2/mapillary"

    # Pre-create needed image paths
    if (
        os.path.exists("{}/{}/massive_annotations".format(dataset_root, dir_name))
        is False
    ):
        os.makedirs("{}/{}/massive_annotations".format(dataset_root, dir_name))

    load_datasets_and_proc(dataset_root, dir_name, files)

    combined_annotations = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": [],
    }

    for idx in range(int(len(files))):
        each_image_json = readout_each_image(dataset_root, dir_name, idx + 1)
        combined_annotations["images"].extend(each_image_json["images"])
        combined_annotations["annotations"].extend(each_image_json["annotations"])

    combined_annotations["annotations"] = sorted(
        combined_annotations["annotations"],
        key=lambda item: item.__getitem__("image_id"),
    )

    for idx in range(len(combined_annotations["annotations"])):
        combined_annotations["annotations"][idx]["id"] = idx + 1

    combined_json_path = "{}/{}/instances_shape_{}2020.json".format(
        dataset_root, dir_name, sample_type
    )
    with open(combined_json_path, "w") as fp:
        json.dump(combined_annotations, fp)


if __name__ == "__main__":
    dir_name = "data/training/v2.0"
    dataset_root = "/home/ec2-user/SageMaker"
    main(dir_name, dataset_root, "training")
    main(dir_name, dataset_root, "validation")
