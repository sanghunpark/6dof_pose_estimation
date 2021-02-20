# system
import os
import glob

# PyTorch
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

# etc
from tqdm import tqdm

# My library
from utils.loss import compute_losses
from utils.utils import draw_keypoints, get_keypoints
class Trainer:
    def __init__(self, train_data_loader, val_data_loader, test_data, device, model, optimizer, config):
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        # self.test_data = test_data/
        self.device = device
        self.config = config
        self.checkpoint_path = config['checkpoint']
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)              
        self.model = model
        self.optimizer = optimizer
        
        # loss weights
        self.cf_w = config['cf_w']
        self.pt_w = config['pt_w']
        self.vf_w = config['vf_w']
        self.sg_w = config['sg_w']
        self.cl_w = config['cl_w']

        checkpoint_dir_name = os.path.basename(self.checkpoint_dir)
        self.logger = SummaryWriter('./checkpoint/logs/' + checkpoint_dir_name)
    
    def train(self):        
        load_epoch, it = self.load() 
        max_epochs = self.config['max_epochs']
        max_iters = len(self.train_data_loader)
        epoch_bar = tqdm(desc='epoch: ', initial=load_epoch, total=max_epochs, position=2, leave=True)
        iter_bar = tqdm(desc='iter: ', total=max_iters, position=3, leave=True)
        for epoch in range(0, max_epochs):
            iter_bar.reset()
            for i, data in enumerate(self.val_data_loader):
                loss = self.iterate(data, it, epoch)
                tqdm.write(f' [{epoch+1}/{max_epochs}][{i}/{max_iters}] loss: {loss.item():.4f}') 
                it += 1
                iter_bar.update()
            epoch_bar.update()

    def log(self, mode, losses, it):
        loss_cf, loss_pt, loss_vf, loss_sg, loss_cl = losses
        self.logger.add_scalars(mode+'/loss',{'loss_cf': loss_cf,
                                            'loss_pt': loss_pt,
                                            'loss_vf': loss_vf,
                                            'loss_sg': loss_sg,
                                            'loss_cl': loss_cl}, it+1)

    def iterate(self, data, it, epoch):
        out = self.model(data['rgb'].to(self.device))
        losses = compute_losses(out, data, device=self.device, config=self.config)

        # update model
        loss_cf, loss_pt, loss_vf, loss_sg, loss_cl = losses
        loss = self.cf_w * loss_cf + \
               self.pt_w * loss_pt + \
               self.vf_w * loss_vf + \
               self.sg_w * loss_sg + \
               self.cl_w * loss_cl
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # log losses
        if (it+1) % self.config['log_freq'] == 0:
            self.log('train', losses, it)
            tqdm.write('logged losses at iteration %d' % (it+1))
        
        # log val losses & save model
        if (it+1) % self.config['checkpoint_freq'] == 0:
            losses = self.val(it)            
            self.log('val', losses, it)         
            self.save(it+1, epoch)
        
        return loss

    def val(self, it):
        self.model.eval()
        
        with torch.no_grad():
            ## get losses with validation data
            # max_iter = len(self.val_data_loader)
            # iter_bar = tqdm(desc='val iter: ', total=max_iter, position=1, leave=True) # for test
            # loss_cf = torch.zeros([1]).to(self.device)
            # loss_pt = torch.zeros([1]).to(self.device)
            # loss_vf = torch.zeros([1]).to(self.device)
            # loss_sg = torch.zeros([1]).to(self.device)
            # loss_cl = torch.zeros([1]).to(self.device)
                      
            # for i, data in enumerate(self.val_data_loader): # for test
            #     out = self.model(data['rgb'].to(self.device))
            #     loss_cf, loss_pt, loss_vf, loss_sg, loss_cl = compute_losses(out, data, device=self.device, config=self.config)
            #     loss_cf += loss_cf / max_iter
            #     loss_pt += loss_pt / max_iter
            #     loss_vf += loss_vf / max_iter
            #     loss_sg += loss_sg / max_iter
            #     loss_cl += loss_cl / max_iter                
            #     iter_bar.update()

            ## simple validation
            data = next(iter(self.val_data_loader))
            out = self.model(data['rgb'].to(self.device))
            loss_cf, loss_pt, loss_vf, loss_sg, loss_cl = compute_losses(out, data, device=self.device, config=self.config)

            # log images
            rgb = data['rgb']
            out = self.model(rgb.to(self.device))

            # confidence map
            B, n_pts, H, W = out['cf'].size() # Bx(n_points)xHxW
            self.logger.add_images('confidence', out['cf'].view(-1, 1, H, W), it+1)

            # segmentation
            B, n_cls, H, W = out['sg'].size() # Bx(n_class)xHxW
            self.logger.add_images('Segmentation', out['sg'].view(-1, 1, H, W), it+1)

            label = data['label'].to(self.device)
            gt_pnts = label[:,1:2*n_pts +1].view(-1, n_pts, 2) # Bx(2*n_points+3) > Bx(n_points)x2
            pr_pnts = get_keypoints(out['cf'])
            
            imgs = draw_keypoints(rgb, gt_pnts, pr_pnts)
            self.logger.add_images('Keypoints', imgs, it+1)


            # out['pt'] # Bx(n_points)xHxW 
            # out['vf'] # Bx(2*n_points)xHxW            
            # out['cl'] # Bx(n_class)

        self.model.train()
        return [loss_cf, loss_pt, loss_vf, loss_sg, loss_cl]

    def save(self, it, epoch):
        # create a directory for checkpoints
        if not os.path.exists(self.checkpoint_dir):
            try:
                original_umask = os.umask(0)
                os.makedirs(self.checkpoint_dir, mode=0o777)
            finally:
                os.umask(original_umask)
        
        # save model
        path = os.path.join(self.checkpoint_dir, 'model_%08d.pt' % it)

        torch.save({'model' : self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                    'iter' : it,
                    'epoch' : epoch}, path)
        tqdm.write('saved model at iteration %d , path: %s' % (it, path))

        # remove previous models
        model_list = glob.glob(self.checkpoint_dir + '/*.pt')
        for file_path in model_list:
            if path != file_path in file_path:
                os.remove(file_path)

    def load(self):
        if not os.path.isfile(self.checkpoint_path):
            print('[**Warning**] There is no checkpoint to load')
            print(self.checkpoint_path)
            return 0, 0

        # load model and optimizer
        checkpoint = torch.load(self.checkpoint_path, map_location=self.model.device)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        print('[**Notice**] Succeeded to load')
        print(self.checkpoint_path)
        return checkpoint['epoch'], checkpoint['iter']

    @staticmethod
    def load_weight(checkpoint_path, model): # to load model weights from a checkpoint in test code
        assert os.path.isfile(checkpoint_path), print('[**Error**] No checkpoint to load!')

        checkpoint = torch.load(checkpoint_path, map_location=model.device)
        model.load_state_dict(checkpoint['model'])
        return model