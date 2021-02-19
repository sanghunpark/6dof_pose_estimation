
def get_config(config):
    import yaml
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)

def options():
    import argparse
    parser = argparse.ArgumentParser(description='6DoF Pose Estimation Arguments')
    parser.add_argument('-c','--config', type=str, default='./config.yaml', help='e.g.> --config=\'./config.yaml\'')
    parser.add_argument('-g','--gpu', action='store_true')
    
    args = parser.parse_args()
    print(args)
    config = get_config(args.config)
    return args, config