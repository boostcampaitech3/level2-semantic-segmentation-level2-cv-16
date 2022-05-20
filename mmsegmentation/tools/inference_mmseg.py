from mmcv import Config
from mmseg.models import build_segmentor
from mmseg.datasets import build_dataloader, build_dataset
from mmcv.runner import load_checkpoint
import os
from mmcv.parallel import MMDataParallel
from mmseg.apis import single_gpu_test

import json
import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Convert MMSegmentation output to csv for AI Stages submission')
    parser.add_argument('--config_path', default='../configs/_dabin_/ocrnet/ocrnet_hr48_512x1024_80k_trash.py', help='config file path') # fix
    parser.add_argument('--root_test_dir', default='../data/test', help='path of test data')
    parser.add_argument('--work_dir', default='mmsegmentation/work_dirs/DB_ocrnet_cleansing', help='directory path which contains pth file') # fix
    parser.add_argument('--test_json_path',default='../data/test.json', help='path of test.json')
    parser.add_argument('--epoch', default='latest', help='epoch of inference')
    parser.add_argument('--result_path', default='./submission', help='directory to save csv file in')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    classes = [
        'Background', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
        'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing'
    ]
    
    cfg = Config.fromfile(args.config_path)
    
    cfg.data.test.classes = classes
    cfg.data.test.img_dir = args.root_test_dir
    cfg.data.test.pipeline[1]['img_scale'] = (512, 512)
    cfg.data.test.test_mode = True
    cfg.data.samples_per_gpu = 4
    cfg.seed=21
    cfg.gpu_ids = [1]
    cfg.work_dir = args.work_dir
    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
    cfg.model.train_cfg = None
    
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=cfg.data.workers_per_gpu, dist=False, shuffle=False)
    
    checkpoint_path = os.path.join(cfg.work_dir, f'{args.epoch}.pth')
    
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')

    model.CLASSES = classes
    model = MMDataParallel(model.cuda(), device_ids=[0])
    
    output = single_gpu_test(model, data_loader)

    with open(args.test_json_path, "r", encoding="utf8") as file:
        test_json_info = json.load(file)
        
    file_names, preds = [], []

    for img_id, pred in enumerate(output):
        file_names.append(test_json_info["images"][img_id]["file_name"])
        pred = pred.reshape(1, 512, 512)
        pred = pred.reshape((1, 256, 2, 256, 2)).max(4).max(2)
        preds.append(' '.join(str(e) for e in pred.flatten()))

    submission = pd.read_csv('./submission/sample_submission.csv', index_col=None)

    # PredictionString 대입
    for file_name, string in zip(file_names, preds):
        submission = submission.append({"image_id" : file_name, "PredictionString" : string}, 
                                   ignore_index=True)
    # submission.csv로 저장
    submission.to_csv(os.path.join(args.result_path, f'{args.epoch}.csv'), index=False)
    
    
if __name__ == "__main__":
    main()  
    
