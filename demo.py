import warnings

from torchvision.transforms.transforms import ToPILImage
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

def demo(args, config, dim='2D', mode='vf'):
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

    # get dataset and set data loader (to get 3D keypoints)
    transf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor()])
    dataset = Dataset6Dof(data_root,                   
                    n_class = config['n_class'],
                    split='test',
                    transform=transf)

    # get image from a webcam
    cap = cv.VideoCapture(0)
    batch_idx = torch.LongTensor(range(1)).to(device)
    while(True):
        ret, frame = cap.read()
        rgb = transf(frame).unsqueeze(0).to(device)    
        # rgb = torch.Tensor(frame).permute(2,0,1).unsqueeze(0).to(device)        
        out = model(rgb)
        if dim == '2D':
            pr_pnts = compute_2D_points(rgb, out, device, config, mode)
        elif dim == '3D':
            ## 3D (projected 2D points from 3D points)
            proj_2d_pr = compute_3D_points(dataset, out, device, config, mode)       
            pr_pnts = torch.from_numpy(proj_2d_pr).permute(1,0).unsqueeze(0)

        # draw 2D points 
        # imgs = draw_keypoints(rgb, gt_pnts,  pr_pnts)
        img = draw_bouding_box(frame, pr_pnts, color=(0.2,1,0))
        cv.imshow(dim+' '+mode, img)
        cv.imshow('confidence map', out['cf'][0,0:1,:,:].permute(1,2,0).cpu().detach().numpy())        
        cls_idx = torch.argmax(out['cl'], dim=1).to(device)
        sg_1 = out['sg'][batch_idx,cls_idx,:,:]
        cv.imshow('segmentation', sg_1.permute(1,2,0).cpu().detach().numpy())        

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # release a webcam
    cap.release()
    cv.destroyAllWindows()

    
if __name__ == '__main__':
    args, config = options()
    demo(args,
         config,
         dim='2D',
         mode='cf')