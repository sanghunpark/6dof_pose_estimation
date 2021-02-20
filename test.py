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
from utils.utils import draw_keypoints, get_keypoints_cf, get_keypoints_vf, get_vector_field

# 
import cv2 as cv
import kornia

def visualize_2D_points(rgb, label, out, device):
    _N_KEYPOINT = 9
    # compute keypoints from confidence map        
    gt_pnts = label[:,1:2*_N_KEYPOINT+1].view(-1, _N_KEYPOINT, 2) # Bx(2*n_points+3) > Bx(n_points)x2
    pr_conf = out['cf']
    pr_pnts = get_keypoints_cf(pr_conf)

    imgs_cf = draw_keypoints(rgb, gt_pnts, pr_pnts)
    cv.imshow('cf', cv.cvtColor(imgs_cf[0].transpose(1,2,0), cv.COLOR_RGB2BGR) )

    # compute key points from vector fields
    pr_vf = out['vf']
    pr_sg = out['sg']
    pr_cl = out['cl']
    B, _, H, W = rgb.size()
    pos = (kornia.create_meshgrid(H, W).permute(0,3,1,2).to(device) + 1) / 2 # (1xHxWx2) > (1x2xHxW), [-1,1] > [0, 1]  
    batch_idx = torch.LongTensor(range(1)).to(device)
    cls_idx = torch.argmax(pr_cl, dim=1).to(device)
    pr_sg_1 = pr_sg[batch_idx,cls_idx,:,:].unsqueeze(1)
    pr_pnts = get_keypoints_vf(pr_vf, pr_sg_1, pos, k=config['k'])
    
    # test
    # gt_vf = get_vector_field(gt_pnts, pos.clone()).view(B, 2*_N_KEYPOINT, H, W)
    # pr_pnts = get_keypoints_vf(gt_vf, pr_sg_1, pos, k=config['k']) 

    imgs_vf = draw_keypoints(rgb, gt_pnts, pr_pnts)
    cv.imshow('vf', cv.cvtColor(imgs_vf[0].transpose(1,2,0), cv.COLOR_RGB2BGR) )
    cv.waitKey(0)

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
        transforms.ToTensor()])
    val_data_loader = DataLoader(Linemod(data_root,                   
                                    n_class = config['n_class'],
                                    split='test',
                                    transform=transf),
                                    batch_size=1,
                                    shuffle=True,
                                    num_workers=config['num_workers'],
                                    drop_last=True)

    for idx, data in enumerate(val_data_loader):        
        rgb = data['rgb'].to(device)
        label = data['label'].to(device)
        out = model(rgb)
        
        visualize_2D_points(rgb, label, out, device)

    
if __name__ == '__main__':
    args, config = options()
    test(args, config)