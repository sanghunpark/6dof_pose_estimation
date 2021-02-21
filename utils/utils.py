# sys
import sys

# PyTorch 
import torch

# etc
import numpy as np
import random
import cv2 as cv
import kornia
# from utils.meshply import MeshPly
import trimesh

# OpenPose Eq(7)
# https://arxiv.org/abs/1812.08008
def get_confidence_map(x, p, sigma):
    ''' x: ground truth (Bx9x2) [0,1], p: grid position '''
    x = x.unsqueeze(-1).unsqueeze(-1) 
    p = p.unsqueeze(0)
    return torch.exp(-torch.linalg.norm(x-p, ord=2, dim=2) / (sigma**2))

# PVNet Eq(1)
# http://arxiv.org/abs/1812.11788
def get_vector_field(x, p):
    # p = (kornia.create_meshgrid(H, W).permute(0,3,1,2).to(device) + 1) / 2 # [-1,1] > [0, 1]
    x = x.unsqueeze(-1).unsqueeze(-1) # Bx(n_points)x2x1x1
    p = p.unsqueeze(1) # Bx1x2xHxW
    return (x-p)/ torch.linalg.norm(x-p, ord=2, dim=2, keepdim=True)

def get_keypoints_cf(confidence_map): # Bx(n_point)xHxW->Bx(n_points)x2
    B, C, H, W = confidence_map.size()
    m = confidence_map.view(B*C, -1).argmax(dim=1).view(B,C,1) # find locations of max value in 1-dimension
    return torch.cat(((m % H)/W, (m // H)/H), dim=2) # Indices (x, y) of max values, Bx(n_points)x2, return position [0,1]

def get_keypoints_vf(vector_field, obj_seg, p, k):
    ''' vector_field: Bx(2*n_pnts)xHxW '''
    ''' segmentation: Bx(n_objects=1)xHxW '''
    # normalize segmentation
    B, n_cls, H, W = obj_seg.size() # Bx(n_cls=1)xHxW
    B, C, H, W = vector_field.size() # Bx(2*n_pnts)xHxW
    n_pnts = int(C/2)
    obj_seg = obj_seg.view(B, n_cls, -1)
    obj_seg -= torch.min(obj_seg, dim=2, keepdim=True)[0]
    obj_seg /= torch.max(obj_seg, dim=2, keepdim=True)[0]
    obj_seg = obj_seg.view(B, n_cls, H, W)
    mask = (obj_seg > 0.5)

    # mask vector fields by oject segmentation
    vector_field = vector_field.view(B,n_pnts,2,H,W)
    mask = mask.expand_as(vector_field)
    vf = torch.masked_select(vector_field, mask).view(B,n_pnts,2,-1).unsqueeze(-2) # Bx(n_pnts)x2x1x(n_sample)
    vf_pos = torch.masked_select(p, mask).view(B,n_pnts,2,-1).unsqueeze(-2) # Bx(n_pnts)x2x1x(n_sample)
    x_pos = p.unsqueeze(1).reshape(1,1,2,H*W).unsqueeze(-1) # (B:1)x(1)x2x(HxW)x(n_sample:1)

    n_sample = vf.size(-1)
    k = min(n_sample, k)
    k_idx = random.sample(range(n_sample), k)
    vf = vf[:,:,:,:,k_idx] # sample for hough voting
    vf_pos = vf_pos[:,:,:,:,k_idx]
    
    # compute cosine similarity (hough voting manner)
    diff = x_pos - vf_pos # Bxn_pntsx2x(HxW)x(k)
    diff_norm = torch.linalg.norm(diff, ord=2, dim=2, keepdim=True)
    dot = vf[:,:,0:1,:,:]*diff[:,:,0:1,:,:] + vf[:,:,1:2,:,:]*diff[:,:,1:2,:,:]
    score = torch.div(dot, diff_norm+ +sys.float_info.epsilon) # to avoid dividing by zero
    score = torch.sum(score, dim=4).squeeze(2)
    val, m = score.max(dim=2, keepdim=True)
    return torch.cat(((m % H)/W, (m // H)/H), dim=2) # predicted keypoints of 2D bouding box (x, y) [0, 1]

def compute_2D_points(rgb, label, out, device, config, mode='vf'):
    _N_KEYPOINT = 9
    if mode == 'cf':
        # compute keypoints from confidence map        
        gt_pnts = label[:,1:2*_N_KEYPOINT+1].view(-1, _N_KEYPOINT, 2) # Bx(2*n_points+3) > Bx(n_points)x2
        pr_conf = out['cf']
        pr_pnts = get_keypoints_cf(pr_conf)
    elif mode == 'vf':
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
    return pr_pnts 

def compute_3D_points(dataset, out, device, config, mode, gt_pnts):
    # compute key points from confidence map
    if mode == 'cf':
        pr_conf = out['cf']
        pr_pnts = get_keypoints_cf(pr_conf)
    # compute key points from vector fields
    elif mode == 'vf':
        pr_vf = out['vf']
        pr_sg = out['sg']
        pr_cl = out['cl']
        B, _, H, W = pr_sg.size()
        pos = (kornia.create_meshgrid(H, W).permute(0,3,1,2).to(device) + 1) / 2 # (1xHxWx2) > (1x2xHxW), [-1,1] > [0, 1]  
        batch_idx = torch.LongTensor(range(1)).to(device)
        cls_idx = torch.argmax(pr_cl, dim=1).to(device)
        pr_sg_1 = pr_sg[batch_idx,cls_idx,:,:].unsqueeze(1)
        pr_pnts = get_keypoints_vf(pr_vf, pr_sg_1, pos, k=config['k'])

    # get ground truth 3D points from mesh file
    pr_cl = out['cl']
    cls_idx = torch.argmax(pr_cl, dim=1).to(device)
    mesh_file = dataset.get_mesh_file(cls_idx)
    mesh = trimesh.load(mesh_file)

    # vertices  = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose()
    corners3D = get_3D_corners(mesh)    

    # Compute PnP
    ppx, ppy, fx, fy = dataset.get_camera_info(cls_idx)
    intrinsic_calibration = get_camera_intrinsic(ppx, ppy, fx, fy)
    K = np.array(intrinsic_calibration, dtype='float32')
    corners2D_pr = pr_pnts[0].cpu().numpy()
    # corners2D_pr = gt_pnts[0].cpu().numpy() # test
    corners2D_pr[:,0] *= 640 # width # to-do : get width/height from dataset
    corners2D_pr[:,1] *= 480 # height
    # R_pr, t_pr = pnp(np.array(np.transpose(np.concatenate((np.zeros((3, 1)), corners3D[:3, :]), axis=1)), dtype='float32'), corners2D_pr, K)
    R_pr, t_pr = pnp(np.array(np.transpose(corners3D[:3, :]), dtype='float32'), corners2D_pr, K)
    # project 3D point into 2D image
    Rt_pr = np.concatenate((R_pr, t_pr), axis=1)
    proj_2d_pr = compute_projection(corners3D, Rt_pr, intrinsic_calibration)
    proj_2d_pr[0,:] /= 640
    proj_2d_pr[1,:] /= 480
    return proj_2d_pr

def draw_keypoints(rgb, gt_pnts, pr_pnts):
    B, _, H, W = rgb.size()
    B, n_pnts, _ = pr_pnts.size()
    if n_pnts != gt_pnts.size(1):
        gt_pnts = gt_pnts[:,1:,:] # use only boudning box points (remove controid point)    
    assert gt_pnts.size(1) == pr_pnts.size(1), f'Erorr: number of points is different.: GT:{gt_pnts.size(1)} != PR:{pr_pnts.size(1)}'

    imgs = rgb.cpu().detach().permute(0,2,3,1).numpy() # BxCxHxW > BxHxWxC
    imgs = np.array(imgs)
   
    for b in range(B):
        img = cv.cvtColor(imgs[b].copy(), cv.COLOR_RGB2BGR)
        for i in range(n_pnts):
            x, y = gt_pnts[b][i]
            img = cv.circle(img, (x*W, y*H), 8, (i/9, (9-i)/9, (9-i)/9), 1)
            x, y = pr_pnts[b][i]
            img = cv.circle(img, (x*W, y*H), 3, (i/9, (9-i)/9, (9-i)/9), -1)
        imgs[b] = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # return torch.from_numpy(imgs).permute(0, 3, 1, 2)
    return imgs.transpose((0, 3, 1, 2))

def draw_bouding_box(img, pnts, color):
    _, H, W = img.size()
    _N_BB_PNTS = 8
    if pnts.size(1) != _N_BB_PNTS:
        pnts = pnts[:,1:,:]
    assert pnts.size(1) == _N_BB_PNTS, f'Erorr: number of points must be {_N_BB_PNTS}: PNT:{pnts.size(1)}'
    B, n_pnts, _ = pnts.size()
    img = img.cpu().detach().permute(1,2,0).numpy() # CxHxW > HxWxC
    img = np.array(img)
    img = cv.resize(img, dsize=(640, 480), interpolation=cv.INTER_AREA)
    bb_pnts = torch.cat((pnts[:,(0,1),:],
                         pnts[:,(3,2),:],
                         pnts[:,(4,5),:],
                         pnts[:,(7,6),:],
                         pnts[:,(0,2),:],
                         pnts[:,(6,4),:],
                         pnts[:,(1,3),:],
                         pnts[:,(7,5),:]), dim=1)

    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    for i in range(0, n_pnts*2, 4):
        pts = bb_pnts[:,i:i+4,:].permute(1,0,2).cpu().numpy()

        pts[:,:,0] *= 640 # W
        pts[:,:,1] *= 480 # H
        pts = np.array(pts, np.int32)
        cv.polylines(img, [pts], True, color, 1)

    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # return torch.from_numpy(imgs).permute(0, 3, 1, 2)
    # return imgs.transpose((0, 3, 1, 2))
    return img.transpose(2,0,1) #  HxWxC > CxHxW
    
## SingleShotPose
def get_3D_corners(mesh):
    Tform = mesh.apply_obb()
    vertices = mesh.bounding_box.vertices
    centroid = mesh.centroid    

    min_x = np.min(vertices[0,:])
    max_x = np.max(vertices[0,:])
    min_y = np.min(vertices[1,:])
    max_y = np.max(vertices[1,:])
    min_z = np.min(vertices[2,:])
    max_z = np.max(vertices[2,:])
    points = np.array([[centroid[0], centroid[1], centroid[2]],
                        [min_x, min_y, min_z],
                        [min_x, min_y, max_z],
                        [min_x, max_y, min_z],
                        [min_x, max_y, max_z],
                        [max_x, min_y, min_z],
                        [max_x, min_y, max_z],
                        [max_x, max_y, min_z],
                        [max_x, max_y, max_z]])
    points = trimesh.transformations.transform_points(points, np.linalg.inv(Tform))
    corners = np.concatenate((np.transpose(points), np.ones((1,9)) ), axis=0)
    print(corners.shape)
    return corners

def pnp(points_3D, points_2D, cameraMatrix):
    try:
        distCoeffs = pnp.distCoeffs
    except:
        distCoeffs = np.zeros((8, 1), dtype='float32') 
    assert points_3D.shape[0] == points_2D.shape[0], f'points 3D and points 2D must have same number of vertices: 3D:{points_3D.shape} 2D:{points_2D.shape} '
    
    # _, R_exp, t, _ = cv.solvePnPRansac(points_3D,
    #                         np.ascontiguousarray(points_2D[:,:2]).reshape((-1,1,2)),
    #                         cameraMatrix,
    #                         distCoeffs)
    
    _, R_exp, t = cv.solvePnP(points_3D,
                              np.ascontiguousarray(points_2D[:,:2]).reshape((-1,1,2)),
                              cameraMatrix,
                              distCoeffs)                            

    R, _ = cv.Rodrigues(R_exp)
    return R, t

def get_camera_intrinsic(u0, v0, fx, fy):
    return np.array([[fx, 0.0, u0], [0.0, fy, v0], [0.0, 0.0, 1.0]])

def compute_projection(points_3D, transformation, internal_calibration):
    projections_2d = np.zeros((2, points_3D.shape[1]), dtype='float32')
    camera_projection = (internal_calibration.dot(transformation)).dot(points_3D)
    projections_2d[0, :] = camera_projection[0, :]/camera_projection[2, :]
    projections_2d[1, :] = camera_projection[1, :]/camera_projection[2, :]
    return projections_2d


