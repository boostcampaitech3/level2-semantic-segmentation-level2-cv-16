import os
import json
import shutil

"""
Set json_dir
"""
DATAROOT = "/opt/ml/input/data/"
TRAINJSON = os.path.join(DATAROOT, "train_revised_final.json")
VALIDJSON = os.path.join(DATAROOT, "val_revised_final.json")
TESTJSON = os.path.join(DATAROOT, "test.json")

"""
Redistribution image by train/valid/test

rename img file name by img_id
"""


def _rename_images(json_dir, image_dir):
    with open(json_dir, "r", encoding="utf8") as outfile:
        json_data = json.load(outfile)
    image_datas = json_data["images"]

    for image_data in image_datas:
        shutil.copyfile(
            os.path.join("/opt/ml/input/data/images", image_data["file_name"]),
            os.path.join(image_dir, f"{image_data['id']:04}.jpg"),
        )


"""
Wrap func
"""


def make(json, path):
    imagePath = "/opt/ml/input/data/" + path
    os.makedirs(imagePath, exist_ok=True)
    _rename_images(json, imagePath)


"""
Main
"""


def __main__():
    make(TRAINJSON, "images/training")
    make(VALIDJSON, "images/validation")
    make(TESTJSON, "test")


if __name__ == "__main__":
    __main__()
