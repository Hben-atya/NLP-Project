import argparse
import yaml

# Config Parser
parser = argparse.ArgumentParser(description='Configuration of MoCo model and Linear Classifier')
parser.add_argument('--config', default='config.yaml', type=str,
                    help='Path to yaml config file. defualt: config.yaml')
args = parser.parse_args()

with open(args.config, encoding="utf8") as f:
    global config_args
    config_args = yaml.load(f, Loader=yaml.FullLoader)
