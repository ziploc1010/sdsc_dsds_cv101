import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import datetime
from pycococreatortools import pycococreatortools
from PIL import Image

dataDir = '/content/gdrive/My Drive/deeplearning/data/drinks/'

input_filename = 'segmentation_test.json'

output_filename = 'segmentation_test_coco.json'

def load_json(data_path, jsfile):
    with open(os.path.join(data_path, jsfile), 'r') as f:
        js = json.load(f)

    return js
    
drinks_json = load_json(dataDir,input_filename)

js = drinks_json["_via_img_metadata"]

INFO = {
    "description": "Drinks.",
    "url": "https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras",
    "version": "0.1.0",
    "year": 2019,
    "contributor": "Rowel Atienza",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "PROPRIETARY",
        "url": "PROPRIETARY"
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'water',
        'supercategory': 'drinks',
    },
    {
        'id': 2,
        'name': 'soda',
        'supercategory': 'drinks',
    },
    {
        'id': 3,
        'name': 'juice',
        'supercategory': 'drinks',
    },
]


coco_output = {
    "info": INFO,
    "licenses": LICENSES,
    "categories": CATEGORIES,
    "images": [],
    "annotations": []
}

segmentation_id = 1
image_id = 1

for i,key in enumerate(js.keys()):
    if i %100==0:
        print(i)
    entry = js[key]
    
    filename = entry['filename']
    image_filename = os.path.join(dataDir, filename)
    image = plt.imread(image_filename)
    
    image_shape = image.shape
    image_shape_tuple = (image_shape[0], image_shape[1])   

    image_info = pycococreatortools.create_image_info(image_id, os.path.basename(image_filename), image_shape)

    coco_output["images"].append(image_info)
    
    masks = []
    regions = entry["regions"]
    for region in regions:
        shape = region["shape_attributes"]
        x = shape["all_points_x"]
        y = shape["all_points_y"]
        name = region["region_attributes"]
        class_id = int(name["Name"])
        fmt = "%s,%s,%s,%s"
        #line = fmt % (filename, x, y, class_id)
        # print(line)
        xy = np.array([x, y], dtype=np.int32)
        xy = np.transpose(xy)
        xy = np.reshape(xy, [1, -1, 2])
        #mask = { class_id : xy }
        #masks.append(mask)
        
        category_info = {'id': class_id, 'is_crowd': False}
        
        bg = np.ones(image_shape_tuple, dtype="uint8")
        bg.fill(255)

        img = np.zeros_like(image)

        cv2.fillPoly(img, xy, (255,255,255))

        im = Image.fromarray(img)
        
        binary_mask = np.asarray(im.convert('1')).astype(np.uint8)

        annotation_info = pycococreatortools.create_annotation_info(
                segmentation_id, image_id, category_info, binary_mask)
        
        coco_output["annotations"].append(annotation_info)
        
        segmentation_id = segmentation_id + 1
    image_id = image_id + 1
    
    
with open(dataDir + output_filename, 'w') as output_json_file:
    json.dump(coco_output, output_json_file)


