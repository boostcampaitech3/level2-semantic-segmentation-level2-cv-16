import os
import argparse
from turtle import forward
import timm
from sklearn.semi_supervised import LabelSpreading
from  recycledataset import CustomDataLoader
import segmentation_models_pytorch as smp
from unetthreeplus import Unet3Plus
import torch
import numpy as np
from utils import label_accuracy_score, add_hist
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torch.nn.functional as F
import wandb
from tqdm import tqdm
import pandas as pd
from copy_pasteaug import CopyPasteV2
import random
from torch.cuda.amp import autocast,GradScaler

def main(cfg):
    pass

def collate_fn(batch):
        return tuple(zip(*batch))

def save_model(model, saved_dir, file_name='unet3+_best_model_dsv.pt'):
    check_point = {'net': model.state_dict()}
    output_path = os.path.join(saved_dir, file_name)
    torch.save(model, output_path)

def validation(epoch, model, data_loader, criterion, device):
    print(f'Start validation #{epoch}')
    model.eval().to(device)
    model = model.to(device)
    model = model
    with torch.no_grad():
        n_class = 11
        cnt = 0
        
        hist = np.zeros((n_class, n_class))
        for step, (images, masks, _) in enumerate(data_loader):
            
            images = torch.stack(images).float()
            masks = torch.stack(masks).long()  

            images, masks = images.to(device), masks.to(device)            
            
            # device 할당
            

            with autocast():
                outputs ,labels= model(images)
                # inference
                loss = 0
                loss_cls = F.binary_cross_entropy_with_logits(labels,torch.amax(F.one_hot(masks,num_classes = 11).float(),dim = (1,2)))
                for i in range(len(outputs)):
                    outputs[i] = outputs[i]*(torch.sigmoid(labels.view(-1,11,1,1))>0.5).float()
                    loss += F.cross_entropy(outputs[i],masks)*0.2*0.5
                loss += loss_cls*0.4/2
            
            cnt += 1
            outputs = torch.argmax(outputs[0], dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()
            
            hist = add_hist(hist, masks, outputs, n_class=n_class)
        
        acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
        IoU_by_class = [{classes : round(IoU,4)} for IoU, classes in zip(IoU , category_names)]
        
        avrg_loss =loss / cnt
        print(f'Validation #{epoch}  Average Loss: {round(avrg_loss.item(), 4)}, Accuracy : {round(acc, 4)}, \
                mIoU: {round(mIoU, 4)}')
        print(f'IoU by class : {IoU_by_class}')
        
    return avrg_loss, round(avrg_loss.item(), 4), round(mIoU, 4), round(acc, 4)

category_names = ['Backgroud', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']

def train(num_epochs, model, data_loader, val_loader, criterion, optimizer,scheduler, saved_dir, val_every, device):
    print(f'Start training..')
    n_class = 11
    best_miou = 0.0
    model.train()
    scaler = GradScaler()
    for epoch in range(num_epochs):

        optimizer.zero_grad()
        hist = np.zeros((n_class, n_class))
        for step, (images, masks, _) in enumerate(data_loader):
            images = torch.stack(images).float()      
            masks = torch.stack(masks).long() 
            
            # gpu 연산을 위해 device 할당
            images, masks = images.to(device), masks.to(device)
            with autocast():
                outputs ,labels= model(images)
                # inference
                loss = 0
                loss_cls = F.binary_cross_entropy_with_logits(labels,torch.amax(F.one_hot(masks,num_classes = 11).float(),dim = (1,2)))
                for i in range(len(outputs)):
                    outputs[i] = outputs[i]*(torch.sigmoid(labels.view(-1,11,1,1))>0.5).float()
                    loss += F.cross_entropy(outputs[i],masks)*0.25*0.5
                loss += loss_cls*0.4/2
                scaler.scale(loss).backward()
            # loss 계산 (cross entropy loss)
            if (step+1) % 2 == 0: 
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            outputs = torch.argmax(outputs[0], dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()
            hist = add_hist(hist, masks, outputs, n_class=n_class)
            acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
            
            # step 주기에 따른 loss 출력
            if (step + 1) % 50 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(train_loader)}], \
                        Loss: {round(loss.item()*2,4)}, Cls_Loss: {round(loss_cls.item(),4)},,mIoU: {round(mIoU,4)}')
             
        # validation 주기에 따른 loss 출력 및 best model 저장
        scheduler.step()
        if (epoch + 1) % val_every == 0:
            avrg_loss, valid_loss, valid_miou, valid_accuracy = validation(epoch + 1, model, val_loader, criterion, device)
            if valid_miou > best_miou:
                print(f"Best performance at epoch: {epoch + 1}")
                print(f"Save model in {saved_dir}")
                best_miou = valid_miou
                save_model(model, saved_dir)
        wandb.log({
            "learning_rate" : optimizer.param_groups[0]['lr'],
            "train_loss": round(loss.item(),4),
            "train_miou": round(mIoU,4),
            "valid_loss": valid_loss,
            "valid_miou": valid_miou,
            "valid_accuracy": valid_accuracy,
        })
def dense_crf(img, output_probs):
    MAX_ITER = 10
    POS_W = 3
    POS_XY_STD = 3
    Bi_W = 4
    Bi_XY_STD = 49
    Bi_RGB_STD = 5

    c = output_probs.shape[0]
    h = output_probs.shape[1]
    w = output_probs.shape[2]

    U = utils.unary_from_softmax(output_probs)
    U = np.ascontiguousarray(U)

    img = np.ascontiguousarray(img)

    d = dcrf.DenseCRF2D(w, h, c)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=POS_XY_STD, compat=POS_W)
    d.addPairwiseBilateral(sxy=Bi_XY_STD, srgb=Bi_RGB_STD, rgbim=img, compat=Bi_W)

    Q = d.inference(MAX_ITER)
    Q = np.array(Q).reshape((c, h, w))
    return Q
def test(model, data_loader, device):
    size = 256
    transform = A.Compose([A.Resize(size, size)])
    print('Start prediction.')
    
    model.eval()
    file_name_list = []
    preds_array = np.empty((0, size*size), dtype=np.long)
    
    with torch.no_grad():
        for step, (imgs, image_infos) in enumerate(tqdm(test_loader)):
            
            # inference (512 x 512)
            outs = model(torch.stack(imgs).float().to(device))
            oms = torch.argmax(outs, dim=1).detach().cpu().numpy()
            
            # resize (256 x 256)
            temp_mask = []
            for img, mask in zip(np.stack(imgs), oms):
                transformed = transform(image=img, mask=mask)
                mask = transformed['mask']
                temp_mask.append(mask)
                
            oms = np.array(temp_mask)
            
            oms = oms.reshape([oms.shape[0], size*size]).astype(int)
            preds_array = np.vstack((preds_array, oms))
            
            file_name_list.append([i['file_name'] for i in image_infos])
    print("End prediction.")
    file_names = [y for x in file_name_list for y in x]
    
    return file_names, preds_array

class hbloss(nn.Module):
    def __init__(self):
        super(hbloss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
    def forward(self,pred,target):
        return self.ce(pred,target)

class BeitEncoder(nn.Module):
    def __init__(self,beit,pyra):
        self.feature = {}
        super(BeitEncoder,self).__init__()
        self.beit = beit
        self.pyra = pyra
    
    def forward(self,x):
        return self.pyra(self.beit.internal_features(x))

if __name__ == '__main__':
    # seed 고정
    random_seed = 21
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    wandb.init(project="semantic-segmentation", entity="medic", name="SY_UNET3+_dsv")
    data_path = '../input/data'
    train_path = 'train.json'
    valid_path = 'val.json'
    test_path = 'test.json'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    aux_params=dict(
    pooling='max',             # one of 'avg', 'max'
    dropout=0.5,               # dropout ratio, default is None
    activation = None,    # activation function, default is None
    classes=11,                 # define number of output labels
    )

    model = Unet3Plus(encoder_name="tu-convnext_xlarge_in22k",encoder_depth = 4,encoder_weights='imagenet',decoder_channels = 256,
        cat_channels = 64,decoder_attention_type = 'scse',in_channels=3,classes=11,aux_params=aux_params)

    model
    model.cuda()
    preprocessing_fn = smp.encoders.get_preprocessing_fn('efficientnet-b5','imagenet')
    
    train_transform = A.Compose([
        A.RandomResizedCrop(512,512, (0.5, 1.0), p=0.5),
        A.Flip(p=0.5),
        CopyPasteV2(),
        A.ShiftScaleRotate(p = 0.5),
        A.Lambda(image=preprocessing_fn),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Lambda(image=preprocessing_fn),
        ToTensorV2()
    ])

    test_transform = A.Compose([
        A.Lambda(image=preprocessing_fn),
        ToTensorV2()
    ])
    
    train_dataset = CustomDataLoader(data_path,train_path,'train',train_transform)
    val_dataset = CustomDataLoader(data_path,valid_path,'val',val_transform)
    test_dataset = CustomDataLoader(data_path,test_path,'test',test_transform)

    epoch = 1
    train_loader = DataLoader(  dataset=train_dataset, 
                                batch_size=4,
                                shuffle=True,
                                num_workers=4,
                                collate_fn=collate_fn,
                                drop_last=True)

    val_loader = DataLoader(dataset=val_dataset, 
                            batch_size=16,
                            shuffle=False,
                            num_workers=4,
                            collate_fn=collate_fn)
    
    test_loader = DataLoader(dataset=test_dataset, 
                            batch_size=16,
                            shuffle=False,
                            num_workers=4,
                            collate_fn=collate_fn)

    # Loss function 정의

    criterion = hbloss()

    # Optimizer 정의
    optimizer = torch.optim.AdamW(params = model.parameters(), lr = 5*1e-5, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=1, eta_min=1e-7)
    
    train(60,model,train_loader,val_loader,criterion,optimizer,scheduler,saved_dir='./saved_dir',val_every=1, device = device)
    
    submission = pd.read_csv('./submission/sample_submission.csv', index_col=None)

    # test set에 대한 prediction
    file_names, preds = test(model, test_loader, device)

    # PredictionString 대입
    for file_name, string in zip(file_names, preds):
        submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}, 
                                    ignore_index=True)

    # submission.csv로 저장
    submission.to_csv("./submission/u3p_convnext.csv", index=False)

