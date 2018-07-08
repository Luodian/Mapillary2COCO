#!/usr/bin/env python3

import datetime
import json
import os
import re
import fnmatch
from PIL import Image
import numpy as np
from pycococreatortools import pycococreatortools

ROOT_DIR = 'train'
IMAGE_DIR = os.path.join(ROOT_DIR, "shapes_train2018")
ANNOTATION_DIR = os.path.join(ROOT_DIR, "annotations")

INFO = {
    "description": "Mapillary",
    "url": "https://github.com/waspinator/pycococreator",
    "version": "0.1.0",
    "year": 2018,
    "contributor": "PanXingJia/LiBo",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'Bird',
        'supercategory': 'animal--bird',
    },
    {
        'id': 2,
        'name': 'Ground_Animal',
        'supercategory': 'animal--ground-animal',
    },
    {
        'id': 3,
        'name': 'Crosswalk_Plain',
        'supercategory': 'construction--flat--crosswalk-plain',
    },
    {
        'id': 4,
        'name': 'Person',
        'supercategory': 'human',
    },
    {
        'id': 5,
        'name': 'Bicyclist',
        'supercategory': 'human--rider',
    },
        {
        'id': 6,
        'name': 'Motorcyclist',
        'supercategory': 'human--rider',
    },
        {
        'id': 7,
        'name': 'Other_Rider',
        'supercategory': 'human--rider',
    },
        {
        'id': 8,
        'name': 'Lane_Marking_-_Crosswalk',
        'supercategory': 'marking--crosswalk-zebra',
    },
        {
        'id': 9,
        'name': 'Banner',
        'supercategory': 'object',
    },
    {
        'id': 10,
        'name': 'Bench',
        'supercategory': 'object',
    },
    {
        'id': 11,
        'name': 'Bike_Rack',
        'supercategory': 'object',
    },
    {
        'id': 12,
        'name': 'Billboard',
        'supercategory': 'object',
    },
    {
        'id': 13,
        'name': 'Catch_Basin',
        'supercategory': 'object',
    },
    {
        'id': 14,
        'name': 'CCTV_Camera',
        'supercategory': 'object',
    },
    {
        'id': 15,
        'name': 'Fire_Hydrant',
        'supercategory': 'object',
    },
    {
        'id': 16,
        'name': 'Junction_Box',
        'supercategory': 'object',
    },
    {
        'id': 17,
        'name': 'Mailbox',
        'supercategory': 'object',
    },
    {
        'id': 18,
        'name': 'Manhole',
        'supercategory': 'object',
    },
    {
        'id': 19,
        'name': 'Phone_Booth',
        'supercategory': 'object',
    },
    {
        'id': 20,
        'name': 'Street_Light',
        'supercategory': 'object',
    },
    {
        'id': 21,
        'name': 'Pole',
        'supercategory': 'object',
    },
    {
        'id': 22,
        'name': 'Traffic_Sign_Frame',
        'supercategory': 'object',
    },
    {
        'id': 23,
        'name': 'Utility_Pole',
        'supercategory': 'object',
    },
    {
        'id': 24,
        'name': 'Traffic_Light',
        'supercategory': 'object',
    },
    {
        'id': 25,
        'name': 'Traffic_Sign_(Back)',
        'supercategory': 'object',
    },
    {
        'id': 26,
        'name': 'Traffic_Sign_(Front)',
        'supercategory': 'object',
    },
    {
        'id': 27,
        'name': 'Trash_Can',
        'supercategory': 'object',
    },
    {
        'id': 28,
        'name': 'Bicycle',
        'supercategory': 'object',
    },
    {
        'id': 29,
        'name': 'Boat',
        'supercategory': 'object',
    },
    {
        'id': 30,
        'name': 'Bus',
        'supercategory': 'object',
    },
    {
        'id': 31,
        'name': 'Car',
        'supercategory': 'object',
    },
    {
        'id': 32,
        'name': 'Caravan',
        'supercategory': 'object',
    },
    {
        'id': 33,
        'name': 'Motorcycle',
        'supercategory': 'object',
    },
    {
        'id': 34,
        'name': 'Other_Vehicle',
        'supercategory': 'object',
    },
    {
        'id': 35,
        'name': 'Trailer',
        'supercategory': 'object',
    },
    {
        'id': 36,
        'name': 'Truck',
        'supercategory': 'object',
    },
    {
        'id': 37,
        'name': 'Wheeled_Slow',
        'supercategory': 'object',
    },
]

def filter_for_jpeg(root, files):
    file_types = ['*.jpeg', '*.jpg']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    
    return files

def filter_for_annotations(root, files, image_filename):
    file_types = ['*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    file_name_prefix = basename_no_extension + '.*'
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]

    return files

def main():

    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 1
    segmentation_id = 1
    
    # filter for jpeg images
    for root, _, files in os.walk(IMAGE_DIR):
        image_files = filter_for_jpeg(root, files)

        # go through each image
        for image_filename in image_files:
            image = Image.open(image_filename)
            image_info = pycococreatortools.create_image_info(
                image_id, os.path.basename(image_filename), image.size)
            coco_output["images"].append(image_info)

            # filter for associated png annotations
            for root, _, files in os.walk(ANNOTATION_DIR):
                annotation_files = filter_for_annotations(root, files, image_filename)

                # go through each associated annotation
                for annotation_filename in annotation_files:
                    print(annotation_filename)
                    if 'Bird' in annotation_filename:
                        class_id = 1
                    elif 'Ground_Animal' in annotation_filename:
                        class_id = 2
                    elif 'Crosswalk_Plain' in annotation_filename:
                        class_id = 3
                    elif 'Person' in annotation_filename:
                        class_id = 4
                    elif 'Bicyclist' in annotation_filename:
                        class_id = 5
                    elif 'Motorcyclist' in annotation_filename:
                        class_id = 6
                    elif 'Other_Rider' in annotation_filename:
                        class_id = 7
                    elif 'Lane_Marking_-_Crosswalk' in annotation_filename:
                        class_id = 8
                    elif 'Banner' in annotation_filename:
                        class_id = 9
                    elif 'Bench' in annotation_filename:
                        class_id = 10
                    elif 'Bike_Rack' in annotation_filename:
                        class_id = 11
                    elif 'Billboard' in annotation_filename:
                        class_id = 12
                    elif 'Catch_Basin' in annotation_filename:
                        class_id = 13
                    elif 'CCTV_Camera' in annotation_filename:
                        class_id = 14
                    elif 'Fire_Hydrant' in annotation_filename:
                        class_id = 15
                    elif 'Junction_Box' in annotation_filename:
                        class_id = 16
                    elif 'Mailbox' in annotation_filename:
                        class_id = 17
                    elif 'Manhole' in annotation_filename:
                        class_id = 18
                    elif 'Phone_Booth' in annotation_filename:
                        class_id = 19
                    elif 'Street_Light' in annotation_filename:
                        class_id = 20
                    elif 'Pole' in annotation_filename:
                        class_id = 21
                    elif 'Traffic_Sign_Frame' in annotation_filename:
                        class_id = 22
                    elif 'Utility_Pole' in annotation_filename:
                        class_id = 23
                    elif 'Traffic_Light' in annotation_filename:
                        class_id = 24
                    elif 'Traffic_Sign_(Back)' in annotation_filename:
                        class_id = 25
                    elif 'Traffic_Sign_(Front)' in annotation_filename:
                        class_id = 26
                    elif 'Trash_Can' in annotation_filename:
                        class_id = 27
                    elif 'Bicycle' in annotation_filename:
                        class_id = 28
                    elif 'Boat' in annotation_filename:
                        class_id = 29
                    elif 'Bus' in annotation_filename:
                        class_id = 30
                    elif 'Car' in annotation_filename:
                        class_id = 31
                    elif 'Caravan' in annotation_filename:
                        class_id = 32
                    elif 'Motorcycle' in annotation_filename:
                        class_id = 33
                    elif 'Other_Vehicle' in annotation_filename:
                        class_id = 34
                    elif 'Trailer' in annotation_filename:
                        class_id = 35
                    elif 'Truck' in annotation_filename:
                        class_id = 36
                    elif 'Wheeled_Slow' in annotation_filename:
                        class_id = 37

                    category_info = {'id': class_id, 'is_crowd': 'crowd' in image_filename}
                    binary_mask = np.asarray(Image.open(annotation_filename)
                        .convert('1')).astype(np.uint8)
                    
                    annotation_info = pycococreatortools.create_annotation_info(
                        segmentation_id, image_id, category_info, binary_mask,
                        image.size, tolerance=2)

                    if annotation_info is not None:
                        coco_output["annotations"].append(annotation_info)
                    segmentation_id = segmentation_id + 1

            image_id = image_id + 1

    with open('{}/instances_shape_train2018.json'.format(ROOT_DIR), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)


if __name__ == "__main__":
    main()
