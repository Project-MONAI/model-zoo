import json


class CreateImageLabelList:
    def __init__(self, filename):
        self.filename = filename
        fid = open(self.filename, "r")
        self.json_dict = json.load(fid)

    def create_dataset(self, grp):
        image_list = []
        label_list = []
        image_label_list = self.json_dict[grp]
        for _ in image_label_list:
            image_list.append(_["image"])
            label_list.append(_["label"])
        return image_list, label_list
