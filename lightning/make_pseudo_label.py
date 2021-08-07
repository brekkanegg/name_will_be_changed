import os, sys
import torch
import numpy as np
import pandas as pd
import timm
import segmentation_models_pytorch as smp
from tqdm import tqdm

from inputs.cxr_dm_test_2 import CXRDataset
from inputs.augmentation import get_augmentation_v2

def get_pseudo_study_4class(test_dir, weights_dir, study_dim=640, study_act='softmax', model_type='smp'):
    
    # data
    _, val_aug = get_augmentation_v2(study_dim)
    study_dataset = CXRDataset(data_dir=test_dir, size=study_dim, mode='test', transform=val_aug)
    test_dataloader = torch.utils.data.DataLoader(
        study_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=8,
        pin_memory=False,
    )
    print('Test dataset setup!')

    # smp:
    if model_type == 'smp':
        aux_params = {
        "classes": 4,
        "pooling": "avg",
        "dropout": None,
        "activation": None,
        }
        pl_models = [smp.UnetPlusPlus(
            encoder_name="timm-efficientnet-b7",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None,
            aux_params=aux_params,
        ).cuda() for _ in range(7)]
        print('Model created!')

        load_ckpts = [torch.load(weights_dir[i])['state_dict'] for i in range(7)]
        renamed_ckpts = [{k.replace('model.', ''): v for k,v in ckpt.items()} for ckpt in load_ckpts]
        for i in range(7):
            pl_models[i].load_state_dict(renamed_ckpts[i])
            pl_models[i].eval()
        print('Model loaded!')
    
        sid_prob_dict = []
        with torch.no_grad():
            for img, img_path in tqdm(test_dataloader):
                img = img.cuda()

                if study_act == 'sigmoid':
                    out = np.mean([torch.sigmoid(plm(img)[1]).detach().cpu().numpy() for plm in pl_models], axis=0)
                    out_std = np.std([torch.sigmoid(plm(img)[1]).detach().cpu().numpy() for plm in pl_models], axis=0)

                elif study_act == 'softmax':
                    out = np.mean([torch.softmax(plm(img)[1], axis=1).detach().cpu().numpy() for plm in pl_models], axis=0)
                    out_std = np.std([torch.softmax(plm(img)[1], axis=1).detach().cpu().numpy() for plm in pl_models], axis=0)

                for idx, ip in enumerate(img_path):
                    iid = ip.split('/')[-1][:-4] 
                    # sid = meta[meta['image_id']==iid]['study_id'].iloc[0]

                    sid_prob_dict.append({
                        'id': iid+"_study",
                        'Negative for Pneumonia': out[idx][0],
                        'Typical Appearance': out[idx][1],
                        'Indeterminate Appearance': out[idx][2],
                        'Atypical Appearance std': out[idx][3], 
                        'Negative for Pneumonia std': out_std[idx][0],
                        'Typical Appearance std': out_std[idx][1],
                        'Indeterminate Appearance std': out_std[idx][2],
                        'Atypical Appearance std': out_std[idx][3], 
                        'std': np.mean(out_std[idx]), 
                                        })

            pseudo_sid = pd.DataFrame(sid_prob_dict)
            pseudo_sid.to_csv('/data/brekkanegg/kaggle/siim-covid19/image_1280/pseudo_test_study_level.csv', index=False)



    # timm
    elif model_type == 'timm':
            # timm:
        pl_models = [timm.create_model('tf_efficientnet_b7_ns', pretrained=False, in_chans=1, num_classes=4).cuda() for _ in range(7)]
        load_ckpts = [torch.load(weights_dir[i])['state_dict'] for i in range(7)]
        renamed_ckpts = [{k.replace('model.', ''): v for k,v in ckpt.items()} for ckpt in load_ckpts]
        for i in range(7):
            pl_models[i].load_state_dict(renamed_ckpts[i])
            pl_models[i].eval()
        
        sid_prob_dict = []
        with torch.no_grad():
            for img, img_path in tqdm(test_dataloader):
                img = img.cuda()
                if study_act == 'sigmoid':
                    out = np.mean([torch.sigmoid(plm(img)).detach().cpu().numpy() for plm in pl_models], axis=0)
                    out_std = np.std([torch.sigmoid(plm(img)).detach().cpu().numpy() for plm in pl_models], axis=0)

                elif study_act == 'softmax':
                    out = np.mean([torch.softmax(plm(img), axis=1).detach().cpu().numpy() for plm in pl_models], axis=0)
                    out_std = np.std([torch.softmax(plm(img), axis=1).detach().cpu().numpy() for plm in pl_models], axis=0)

                # break
                for idx, ip in enumerate(img_path):
                    iid = ip.split('/')[-1][:-4] 

                    sid_prob_dict.append({
                        'id': iid+"_study",
                        'Negative for Pneumonia': out[idx][0],
                        'Typical Appearance': out[idx][1],
                        'Indeterminate Appearance': out[idx][2],
                        'Atypical Appearance': out[idx][3], 
                        'Negative for Pneumonia std': out_std[idx][0],
                        'Typical Appearance std': out_std[idx][1],
                        'Indeterminate Appearance std': out_std[idx][2],
                        'Atypical Appearance std': out_std[idx][3], 
                                        })

            pseudo_sid = pd.DataFrame(sid_prob_dict)
            pseudo_sid.to_csv('/data/brekkanegg/kaggle/siim-covid19/image_1280/psudo_test_study_level.csv')

    # smp:


if __name__ == '__main__':
    test_dir = '/data/brekkanegg/kaggle/siim-covid19/image_1280/test'
    weights_dir = [f'/data/brekkanegg/kaggle/siim-covid19/lightning/ckpt/fold{f}_effb7upp_640_aux_0629/best.ckpt' for f in range(7)]
    get_pseudo_study_4class(test_dir, weights_dir, study_dim=640, study_act='softmax', model_type='smp')