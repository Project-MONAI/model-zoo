import json
import os


class CreateImageLabelList():
    def __init__(self, filename):
        self.filename = filename

    def create_dataset(self):
        fid = open(self.filename, 'r')
        json_dict = json.load(fid)
        image_list = []
        label_list = []

        training_list = json_dict['Train']
        validation_list = json_dict['Validation']
        test_list = json_dict['Test']

        for _ in training_list:
            image_list.append(_['image'])
            label_list.append(_['label'])

        return image_list, label_list


def main():
    json_filelist = '/raid/Data/MayoClinicData/BreastDensity/patient_img_label_FFDM.json'
    CreateImageLabelList(json_filelist)

if __name__ == '__main__':
    main()
