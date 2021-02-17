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

class Trainer:
    def __init__(self, train_data_loader, val_data_loader, device, model, optimizer, config):
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.device = device
        self.config = config
        self.checkpoint_path = config['checkpoints']
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)              
        self.model = model
        self.optimizer = optimizer
        
        # weights
        self.bb_w = config['bb_w']
        
        checkpoint_dir_name = os.path.basename(self.checkpoint_dir)
        self.logger = SummaryWriter('./checkpoint/logs/' + checkpoint_dir_name)

    def log(self, mode, losses, it):
        loss_bb, loss2 = losses # dummy
        self.logger.add_scalars(mode+'/loss',{'loss_bb' : loss_bb,
                                            'loss2' : loss2}, it+1)

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

    def iterate(self, data, it, epoch):        
        out = self.model(data['rgb'].to(self.device))        
        losses = compute_losses(out, data, self.device)

        # update model
        loss_bb, loss_temp = losses
        loss = self.bb_w * loss_bb + \
               1.0 * loss_temp

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # log losses
        if (it+1) % self.config['log_freq'] == 0:            
            self.log('train', losses, it)
            tqdm.write('logged losses at iteration %d' % (it+1))
        
        # log val losses & save model
        if (it+1) % self.config['log_freq'] == 0:
            losses = self.val()            
            self.log('val', losses, it)
                        
            test_data = next(iter(self.val_data_loader))
            dummy_img = vutils.make_grid(test_data['rgb'], normalize=True) # dummy
            self.logger.add_image('dummy_img', dummy_img, it+1)            
            self.save(it+1, epoch)
        
        return loss

    def val(self):
        self.model.eval()
        with torch.no_grad():
            max_iter = len(self.val_data_loader)
            iter_bar = tqdm(desc='val iter: ', total=max_iter, position=1, leave=True) # for test
            loss_bb_avg = torch.zeros([1]).to(self.device)
            loss2_avg= torch.zeros([1]).to(self.device)            
            for i, data in enumerate(self.val_data_loader): # for test
                out = self.model(data['rgb'].to(self.device))
                loss_bb, loss2 = compute_losses(out, data, device=self.device)
                loss_bb_avg += loss_bb / max_iter
                loss2_avg += loss2 / max_iter
                
                iter_bar.update()
        self.model.train()
        return [loss_bb_avg, loss2_avg]

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

        # load models
        checkpoint = torch.load(self.checkpoint_path, map_location=self.model.device)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        print('[**Notice**] Succeeded to load')
        print(self.checkpoint_path)
        return checkpoint['epoch'], checkpoint['iter']

