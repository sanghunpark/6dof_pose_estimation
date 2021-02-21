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
from utils.utils import draw_keypoints, get_keypoints_cf, get_keypoints_vf, get_vector_field, \
get_3D_corners, pnp, get_camera_intrinsic, compute_projection
from utils.meshply import MeshPly

# 
import cv2 as cv
import kornia
import numpy as np

def visualize_2D_points(rgb, label, out, device, config):
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

def visualize_3D_points(dataset, out, device, config):
    # compute key points from vector fields
    pr_vf = out['vf']
    pr_sg = out['sg']
    pr_cl = out['cl']
    B, _, H, W = pr_sg.size()
    pos = (kornia.create_meshgrid(H, W).permute(0,3,1,2).to(device) + 1) / 2 # (1xHxWx2) > (1x2xHxW), [-1,1] > [0, 1]  
    batch_idx = torch.LongTensor(range(1)).to(device)
    cls_idx = torch.argmax(pr_cl, dim=1).to(device)
    pr_sg_1 = pr_sg[batch_idx,cls_idx,:,:].unsqueeze(1)
    pr_pnts = get_keypoints_vf(pr_vf, pr_sg_1, pos, k=config['k'])

    pr_cl = out['cl']
    cls_idx = torch.argmax(pr_cl, dim=1).to(device)
    mesh_file = dataset.get_mesh_file(cls_idx)
    mesh = MeshPly(mesh_file)
    vertices  = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose()
    corners3D = get_3D_corners(vertices)

    ppx, ppy, fx, fy = dataset.get_camera_info(cls_idx)
    intrinsic_calibration = get_camera_intrinsic(ppx, ppy, fx, fy)
    K = np.array(intrinsic_calibration, dtype='float32')

    corners2D_pr = pr_pnts[0].cpu().numpy()
    print(np.array(np.transpose(np.concatenate((np.zeros((3, 1)), corners3D[:3, :]), axis=1)), dtype='float32').shape)
    R_pr, t_pr = pnp(np.array(np.transpose(np.concatenate((np.zeros((3, 1)), corners3D[:3, :]), axis=1)), dtype='float32'), corners2D_pr, K)

    # project 3D point into 2D image
    Rt_pr = np.concatenate((R_pr, t_pr), axis=1)
    proj_2d_pr = compute_projection(corners3D, Rt_pr, intrinsic_calibration)
    return proj_2d_pr

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
    dataset = Linemod(data_root,                   
                    n_class = config['n_class'],
                    split='test',
                    transform=transf)
    val_data_loader = DataLoader(dataset,
                                batch_size=1,
                                shuffle=True,
                                num_workers=config['num_workers'],
                                drop_last=True)

    for idx, data in enumerate(val_data_loader):        
        rgb = data['rgb'].to(device)
        label = data['label'].to(device)
        out = model(rgb)
        
        proj_2d_pr = visualize_3D_points(dataset, out, device, config)
        # visualize_2D_points(rgb, label, out, device, config)
        _N_KEYPOINT = 9
        # compute keypoints from confidence map        
        gt_pnts = label[:,1:2*_N_KEYPOINT+1].view(-1, _N_KEYPOINT, 2) # Bx(2*n_points+3) > Bx(n_points)x2
        
        imgs_vf = draw_keypoints(rgb, gt_pnts[:,3:,:], proj_2d_pr)

    
if __name__ == '__main__':
    args, config = options()
    test(args, config)