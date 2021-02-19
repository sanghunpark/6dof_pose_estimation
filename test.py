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
from utils.trainer import Trainer
from utils.config import options

def test(args, config):
    data_root = config['data_root']
    device = torch.device('cuda:0' if args.gpu else 'cpu')

    # get model
    model = PoseNet(config, device)
    if args.gpu:
        gpu_ids = list(range(torch.cuda.device_count()))
        model = WappedDataParallel(model,
                                device_ids=gpu_ids,
                                output_device=gpu_ids[0])
    model = model.to(device)
    model = Trainer.load_weight(config['checkpoint'], model)

    # get dataset and set data loader
    transf = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor()
    ])
    val_data_loader = DataLoader(Linemod(data_root,                   
                                    n_class = config['n_class'],
                                    split='test',
                                    transform=transf),
                                    batch_size=1,
                                    shuffle=True,
                                    num_workers=config['num_workers'],
                                    drop_last=True)

    for idx, data in enumerate(val_data_loader):
        out = model(data['rgb'].to(device))
        pr_conf = out['cf']

    
if __name__ == '__main__':
    args, config = options()
    test(args, config)