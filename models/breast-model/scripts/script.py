from monai.bundle.config_parser import ConfigParser

parser = ConfigParser()
parser.read_config('../configs/metadata.json')
parser.read_config('../configs/train.json')

net = parser.get_parsed_content("network_def")

device = parser.get_parsed_content("device")

network = parser.get_parsed_content("network_def")

trainer = parser.get_parsed_content("trainer")
trainer.run()