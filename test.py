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
from utils.utils import draw_keypoints, get_keypoints

# 
import cv2 as cv

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
        _N_KEYPOINT = 9
        rgb = data['rgb'].to(device)
        out = model(rgb)
        pr_conf = out['cf']
        
        # compute keypoints from confidence map
        label = data['label'].to(device)
        gt_pnts = label[:,1:2*_N_KEYPOINT+1].view(-1, _N_KEYPOINT, 2) # Bx(2*n_points+3) > Bx(n_points)x2
        pr_pnts = get_keypoints(pr_conf)

        imgs = draw_keypoints(rgb, gt_pnts, pr_pnts)
        cv.imshow('ret', imgs[0].transpose(1,2,0))

        # compute key points from vector fields

        cv.waitKey(0)


    
if __name__ == '__main__':
    args, config = options()
    test(args, config)