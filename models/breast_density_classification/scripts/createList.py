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


def create_sample_imagelist(json_filename):
    fid = open(json_filename, "r")
    file_dict = json.load(fid)
    train_data = file_dict["Train"][0:40]
    validation_data = file_dict["Validation"][0:20]
    test_data = file_dict["Test"][0:20]

    data_dict = {}
    data_dict["Train"] = train_data
    data_dict["Validation"] = validation_data
    data_dict["Test"] = test_data


def main():
    json_filelist = "/raid/Data/MayoClinicData/BreastDensity/patient_img_label_FFDM.json"
    CreateImageLabelList(json_filelist)
    create_sample_imagelist(json_filelist)


if __name__ == "__main__":
    main()
