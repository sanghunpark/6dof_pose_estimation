
# PyTorch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# My library
from data.linemod import Linemod

def get_config(config):
    import yaml
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)

def options():
    import argparse
    parser = argparse.ArgumentParser(description='6DoF Pose Estimation Arguments')
    parser.add_argument('-c','--config', type=str, default='./config.yaml', help='e.g.> --config=\'./config.yaml\'')
    args = parser.parse_args()
    print(args)
    config = get_config(args.config)
    return args, config

def train(args, config):
    data_root = config['data_root']
    
    # get dataset
    transf = transforms.Compose([
        transforms.Resize(config['img_size']),
        transforms.ToTensor()
    ])
    data = Linemod(data_root,
                   objects = ['ape'],
                   split='train',
                   transform=transf)
    data_loader = DataLoader(data,
                             batch_size = config['batch_size'],
                             shuffle=True,
                             num_workers=config['num_workers'],
                             drop_last=True)

    )

    
if __name__ == '__main__':
    args, config = options()
    train(args, config)