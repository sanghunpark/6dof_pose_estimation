import warnings
warnings.filterwarnings("ignore")

# PyTorch
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim

# My library
from data.dataset import Dataset6Dof
from model.posenet import PoseNet
from model.blocks import WappedDataParallel
from utils.trainer import Trainer
from utils.config import options
from utils.utils import compute_2D_points, compute_3D_points, \
draw_keypoints, draw_bouding_box

# 
import cv2 as cv
import kornia
import numpy as np

def test(args, config, dim='2D', mode='vf'):
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
    model.eval()

    # get dataset and set data loader
    transf = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor()])
    dataset = Dataset6Dof(data_root,                   
                    n_class = config['n_class'],
                    split='test',
                    transform=transf)
    val_data_loader = DataLoader(dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=config['num_workers'],
                                drop_last=True)

    for idx, data in enumerate(val_data_loader):        
        rgb = data['rgb'].to(device)
        label = data['label'].to(device)
        out = model(rgb)
        _N_KEYPOINT = 9  
        gt_pnts = label[:,1:2*_N_KEYPOINT+1].view(-1, _N_KEYPOINT, 2) # Bx(2*n_points+3) > Bx(n_points)x2
        
        if dim == '2D':
            pr_pnts = compute_2D_points(out, device, config, mode)
        elif dim == '3D':
            ## 3D (projected 2D points from 3D points)
            proj_2d_pr = compute_3D_points(dataset, out, device, config, mode)       
            pr_pnts = torch.from_numpy(proj_2d_pr).permute(1,0).unsqueeze(0)            

        # draw 2D points 
        # imgs = draw_keypoints(rgb, gt_pnts,  pr_pnts)
        img = rgb[0].permute(1,2,0).detach().cpu().numpy()
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        img = draw_bouding_box(img, gt_pnts, color=(0,1,0))
        img = draw_bouding_box(img, pr_pnts, color=(0,0,1))
        cv.imshow(dim+' '+mode, img )
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    args, config = options()
    test(args,
         config,
         dim='2D',
         mode='cf_vf')