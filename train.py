import warnings
warnings.filterwarnings("ignore")

# PyTorch
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim

# My library
from data.linemod import Linemod
from model.posenet import PoseNet
from model.blocks import WappedDataParallel
from utils.config import options
from utils.trainer import Trainer

def train(args, config):
    data_root = config['data_root']    
    device = torch.device('cuda:0' if args.gpu else 'cpu')

    # get model
    model = PoseNet(config, device)

    # multi-GPU training
    if args.gpu:
        gpu_ids = list(range(torch.cuda.device_count()))
        model = WappedDataParallel(model,
                                device_ids=gpu_ids,
                                output_device=gpu_ids[0])
    model = model.to(device)

    # get dataset and set data loader
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
                                    batch_size=config['batch_size'],
                                    shuffle=True,
                                    num_workers=config['num_workers'],
                                    drop_last=True)

    # get optimizer and trainer
    optimizer = optim.Adam(model.parameters(),
                            lr=config['lr'],
                            betas=(config['beta1'],
                            config['beta2']),
                            eps=config['eps'])

    test_data = next(iter(val_data_loader))
    trainer = Trainer(config=config,
                      train_data_loader=train_data_loader,
                      val_data_loader=val_data_loader,
                      test_data=test_data,
                      device=device,
                      model=model,
                      optimizer=optimizer)
    trainer.train()
    
if __name__ == '__main__':
    args, config = options()
    train(args, config)