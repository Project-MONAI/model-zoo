import os

from monai.bundle.config_parser import ConfigParser

os.environ["CUDA_DEVICE_IRDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = ConfigParser()
parser.read_config("../configs/inference.json")
data = parser.get_parsed_content("data")
device = parser.get_parsed_content("device")
network = parser.get_parsed_content("network_def")

inference = parser.get_parsed_content("evaluator")
inference.run()

print(type(network))


datalist = parser.get_parsed_content("test_imagelist")
print(datalist)

inference = parser.get_parsed_content("evaluator")
inference.run()
