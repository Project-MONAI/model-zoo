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

def CreateSampleImageList(json_filename):
    fid = open(json_filename, 'r')
    file_dict = json.load(fid)
    train_data = file_dict['Train'][0:40]
    validation_data = file_dict['Validation'][0:20]
    test_data = file_dict['Test'][0:20]

    data_dict = {}
    data_dict['Train'] = train_data
    data_dict['Validation'] = validation_data
    data_dict['Test'] = test_data

    fid1 = open('./sample_image_list_FFDM.json', 'w')
    json.dump(data_dict, fid1, indent=1)


def main():
    json_filelist = '/raid/Data/MayoClinicData/BreastDensity/patient_img_label_FFDM.json'
    CreateImageLabelList(json_filelist)
    CreateSampleImageList(json_filelist)

if __name__ == '__main__':
    main()
