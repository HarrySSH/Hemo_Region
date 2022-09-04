import torch
import torch.nn as nn
import sys
import time
import cv2
import os
sys.path.append("..")
from torch.utils.data import DataLoader
from DataLoader.DataLoader import Img_DataLoader
from utils.utils import configure_optimizers, graph_loss, apply_along_axis, find_scale,binary_cross_entropy
class trainer_classification(nn.Module):
    def __init__(self, train_image_files, validation_image_files, gamma = 0.1,
                               init_lr = 0.001, weight_decay = 0.0005, batch_size = 32, epochs = 30, lr_decay_every_x_epochs = 10,
                 print_steps = 50, df = None, df_features = None, graph_loss = False,
                 img_transform = False, model =False,
                save_checkpoints_dir = None):
        super(trainer_classification, self).__init__()
        assert model != False, 'Please put a model!'
        assert img_transform != False, 'Please put a augumentation pipeline!'
        self.df = df
        self.df_features = df_features
        names = list(set(self.df['Cell_Types'].tolist()))
        self.graph_loss = graph_loss
        self.train_image_files = train_image_files#[x  for x in train_image_files if (x.split('/')[-2] in names)]
        self.validation_image_files = validation_image_files#[x  for x in validation_image_files if x.split('/')[-2] in names] 
    
        
        self.batch_size = batch_size
        self.epoch = epochs
        self.global_step = 0
        self.current_step = 0
        self.init_lr = init_lr
        self.lr_decay_every_x_epochs = lr_decay_every_x_epochs
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.print_steps = print_steps
        self.img_transform = img_transform
        self.model = model
        self.save_checkpoints_dir = save_checkpoints_dir
        #self.data_type = os.path.splitext(os.path.split(self.args.train_data_list)[-1])[0]

    def _dataloader(self, datalist, split='train',img_transform = False):
        dataset = Img_DataLoader(img_list=datalist, split=split, transform = img_transform, df = self.df, df_features = self.df_features)
        shuffle = True if split == 'train' else False
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=2, shuffle=shuffle)
        return dataloader

    def train_one_epoch(self, epoch, train_loader, model, optimizer, lr_scheduler):
        t0 = 0.0
        model.train()
        for inputs in train_loader:
            
            self.global_step += 1
            self.current_step +=1

            t1 = time.time()

            images, masks = inputs["image"].cuda(), inputs["label"].cuda()
            if self.df_features is not None:
                features = inputs["features"].cuda()
                mask_out, feature_pred  = model(images)
                feature_loss = nn.BCELoss()(feature_pred, features)

            else:
                mask_out = model(images)

            if self.graph_loss:
                prediction = torch.argmax(torch.flatten(mask_out, start_dim=1), dim=1)
                groundtruth = torch.argmax(torch.flatten(masks, start_dim=1), dim=1)
                scale = apply_along_axis(find_scale, torch.stack([prediction, groundtruth]), axis=-1)
                class_loss = binary_cross_entropy(mask_out, masks, weight_harry=scale.cuda())
            else:
                class_loss = nn.BCELoss()(mask_out, masks)

            #elif self.args.segmentation_loss == "bce":

            if self.df_features is not None:
                total_loss = class_loss + feature_loss
            else:
                total_loss = class_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            t0 += (time.time() - t1)

            if self.global_step % self.print_steps == 0:
                message = "Epoch: %d Step: %d LR: %.6f Total Loss: %.4f Runtime: %.2f s/%d iters." % (epoch+1, self.global_step, lr_scheduler.get_last_lr()[-1], total_loss, t0, self.current_step)
                print("==> %s" % (message))
                self.current_step = 0
                t0 = 0.0

        return total_loss




    def val_one_epoch(self, data_loader, model, epoch):
        with torch.no_grad():
            model.eval()

            for i, inputs in enumerate(data_loader):
                images, masks = inputs["image"].cuda(), inputs["label"].cuda()
                if self.df_features is not None:
                    mask_out, _ = model(images)
                else:
                    mask_out = model(images)
                
                if i == 0:
                    predictions = mask_out
                    groundtruths = masks
                else:
                    predictions = torch.cat((predictions, mask_out), dim=0)
                    groundtruths = torch.cat((groundtruths, masks), dim=0)

            indx = 0
            if self.graph_loss:
                prediction = torch.argmax(torch.flatten(predictions, start_dim=1), dim=1)
                groundtruth = torch.argmax(torch.flatten(groundtruths, start_dim=1), dim=1)
                scale = apply_along_axis(find_scale, torch.stack([prediction, groundtruth]), axis=-1)
                total_loss = binary_cross_entropy(predictions, groundtruths, weight_harry=scale.cuda())
            else:
                loss = torch.nn.BCELoss()
                total_loss = loss(predictions, groundtruths)
            #predictions = predictions.data.cpu().numpy().flatten()
            #groundtruths = groundtruths.data.cpu().numpy().flatten()

            #best_threshold, best_f1, best_iou, _, _, _ = evaluate(predictions, groundtruths, interval=0.02)


        #print("==> Epoch: %d Evaluation Threshold %.2f F1 %.4f." % (epoch+1, best_threshold, best_f1))
        print("==> Epoch: %d Loss %.6f ." % (epoch+1, total_loss.cpu().numpy() ))
        torch.cuda.empty_cache()

        return total_loss

    def train(self,model):
        print("==> Create model.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model= nn.DataParallel(model)
        model.to(device)
        #model = nn.DataParallel(model, device_ids=self.args.gpu_list)
                #model = UNet()

        model.cuda()
        print("==> List learnable parameters")
        '''
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                print("\t{}".format(name))
        '''

        print("==> Load data.")
        print(len(self.train_image_files))
        print(len(self.validation_image_files))

        train_data_loader = self._dataloader(self.train_image_files, split='train', img_transform=self.img_transform)
        val_data_loader = self._dataloader(self.validation_image_files, split='val', img_transform=self.img_transform)

        print("==> Configure optimizer.")
        optimizer, lr_scheduler = configure_optimizers(model, self.init_lr, self.weight_decay,
                                                       self.gamma, self.lr_decay_every_x_epochs)

        print("==> Start training")
        since = time.time()
        best_f1 = 0.0
        loss_list = []
        
        print('==> Creat the saving dictionary')
        if os.path.exists(self.save_checkpoints_dir):
            print('The directory exists, overided the files if duplicates')
        else:
            print('Created new dictionary for saving checkpoints')
            os.makedirs(self.save_checkpoints_dir)
        for epoch in range(self.epoch):

            self.train_one_epoch( epoch, train_data_loader, model, optimizer, lr_scheduler)
            _loss = self.val_one_epoch(val_data_loader, model, epoch)
            loss_list.append(_loss.detach().cpu().numpy())
            
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),}, self.save_checkpoints_dir+'/checkpoint_'+str(epoch)+'_iteration3.ckpt')
            if _loss.detach().cpu().numpy()<= min(loss_list):
                torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),}, self.save_checkpoints_dir+'/checkpoint_best_iteration3.ckpt')
            lr_scheduler.step()
            

        
        print("==> Runtime: %.2f minutes." % ((time.time()-since)/60.0))
        return model