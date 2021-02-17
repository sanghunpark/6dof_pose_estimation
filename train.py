import warnings
warnings.filterwarnings("ignore")

# PyTorch
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim

# My library
from data.linemod import Linemod
from trainer import Trainer
from model.posenet import PoseNet
class WappedDataParallel(nn.DataParallel):
        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)

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

def train(args, config):
    data_root = config['data_root']    
    device = torch.device('cuda:0' if args.gpu else 'cpu')

    # get model
    model = PoseNet(config, device)

    # multi-GPU traning
    if args.gpu:
        gpu_ids = list(range(torch.cuda.device_count()))
        model = WappedDataParallel(model,
                                device_ids=gpu_ids,
                                output_device=gpu_ids[0])
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(),
                            lr=config['lr'],
                            betas=(config['beta1'],
                            config['beta2']),
                            eps=config['eps'])

    # get dataset
    transf = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor()
    ])
    train_data_loader = DataLoader(Linemod(data_root,                   
                                    n_class = config['n_class'],
                                    split='train',
                                    transform=transf),
                                    batch_size=(config['batch_size']),
                                    shuffle=True,
                                    num_workers=config['num_workers'],
                                    drop_last=True)
    
    val_data_loader = DataLoader(Linemod(data_root,                   
                                    n_class = config['n_class'],
                                    split='test',
                                    transform=transf),
                                    batch_size=(config['batch_size']),
                                    shuffle=True,
                                    num_workers=config['num_workers'],
                                    drop_last=True)

    trainer = Trainer(train_data_loader=train_data_loader,
                      val_data_loader=val_data_loader,
                      device=device,
                      model=model,
                      optimizer=optimizer,
                      config=config)
    trainer.train()
    
if __name__ == '__main__':
    args, config = options()
    train(args, config)