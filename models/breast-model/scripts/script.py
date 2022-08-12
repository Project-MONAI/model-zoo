from monai.bundle.config_parser import ConfigParser

parser = ConfigParser()
parser.read_config('../configs/metadata.json')
parser.read_config('../configs/train.json')

net = parser.get_parsed_content("network_def")

device = parser.get_parsed_content("device")

loss = parser.get_parsed_content('loss')
print(loss)
network = parser.get_parsed_content("network")
print(type(loss))
data = parser.get_parsed_content("data")
imageList = parser.get_parsed_content("imagelist")
labellist = parser.get_parsed_content("labellist")
trainer = parser.get_parsed_content("trainer")
trainer.run()