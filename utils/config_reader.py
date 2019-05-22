import yaml

def conf_reader():
    with open("config.yaml", 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    return cfg
