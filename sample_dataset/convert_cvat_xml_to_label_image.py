import argparse
import numpy as np
import os
import cv2
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
import shutil


def create_path_list(xml_directory):
    # Read xml labels, which should be on the format %06d.xml
    xml_path_list = []
    for r, d, f in os.walk(xml_directory):
        for file in f:
            if '.xml' in file:
                xml_path_list.append(os.path.join(r, file))
    xml_path_list.sort()

    return xml_path_list


def convert_xml_to_label_image(xml_path, label_classes_dict, label_priority_order):
    # Parse XML
    tree = ET.parse(xml_path)
    root = tree.getroot()
    imagesize_element = root.find('imagesize')
    # Get image size
    nrows = int(imagesize_element.find("nrows").text)
    ncols = int(imagesize_element.find("ncols").text)

    # Get dict of polygons for each label
    objects_dict = defaultdict(list)
    for obj in root.iter('object'):
        label = obj.find("name").text
        pts = []
        for pt in obj.find("polygon").iter("pt"):
            x_str = pt.find("x").text
            y_str = pt.find("y").text
            pts.append([int(float(x_str)), int(float(y_str))])

        objects_dict[label].append(pts)

    # Create blank image
    label_image = np.zeros((nrows, ncols), dtype=np.uint8)
    for key in label_priority_order:
        for pts in objects_dict[key]:
            class_value = label_classes_dict[key]

            pts_np = np.asarray(pts, dtype=np.int32)

            pts_reshaped = pts_np.reshape((-1, 1, 2))
            cv2.fillPoly(img=label_image, pts=[pts_reshaped], color=class_value)

    return label_image


def main(xml_directory, label_directory):

    # Get list of xml annotations
    xml_annotation_path_list = create_path_list(xml_directory)

    label_classes_dict = defaultdict(int)
    label_classes_dict["person"] = 1
    label_classes_dict["airplane"] = 2
    label_classes_dict["ground"] = 3
    label_classes_dict["sky"] = 4

    # Writing order of polygons :
    # When a pixel belongs to several polygons, the label image will
    # only contain the class id of the last class in the list below 
    label_priority_order = ["sky", "ground", "airplane", "person"]
    # label_priority_order = ["person"]

    # Create dataset directories
    dataset_labels_path = Path(label_directory)
    dataset_labels_path.mkdir(parents=True, exist_ok=True)

    # Loop over elements
    image_counter = 1
    for xml_path in xml_annotation_path_list:
        # Convert xml into an image of labels
        label_image = convert_xml_to_label_image(xml_path, label_classes_dict, label_priority_order)
        # Save label_image
        label_image_path = os.path.join(dataset_labels_path, f"{image_counter:06}.png")
        cv2.imwrite(label_image_path, label_image)
        image_counter = image_counter + 1


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "xml_directory", help='Specify the path to the directory containing xml annotations. Expected format is export from CVAT "LabelMe ZIP 3.0 for images"')
    parser.add_argument(
        "label_directory", help='Specify the path to the destination directory which will contain the labels')

    args = parser.parse_args()
    main(args.xml_directory, args.label_directory)