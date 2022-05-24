# Team Medic(CV-16)


## Project Overview

Project Period
2022.04.25 ~ 2022.05.12
- Project Wrap Up Report
    
    [Semantic Segmentation_CV_팀 리포트(16조).pdf](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/ff8eda8b-e3e5-4704-90e8-8055ffadeda6/Semantic_Segmentation_CV_%ED%8C%80_%EB%A6%AC%ED%8F%AC%ED%8A%B8%2816%EC%A1%B0%29.pdf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220524%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220524T005851Z&X-Amz-Expires=86400&X-Amz-Signature=ae25baaa054223fd2267a340f32da336f42ece15bd8bf89415f861bc04e9353f&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Semantic%2520Segmentation_CV_%25ED%258C%2580%2520%25EB%25A6%25AC%25ED%258F%25AC%25ED%258A%25B8%2816%25EC%25A1%25B0%29.pdf%22&x-id=GetObject)
    

## 🔎 재활용 품목 분류를 위한 Segmentation

![image](https://user-images.githubusercontent.com/74086829/169696764-33e39980-b5fa-452e-bf19-3435a7f08fef.png)

### 😎 Members

| 권순호 | 서다빈 | 서예현 | 이상윤 | 전경민 |
| --- | --- | --- | --- | --- |
| [Github](https://github.com/tnsgh9603) | [Github](https://github.com/sodabeans) | [Github](https://github.com/justbeaver97) | [Github](https://github.com/SSANGYOON?tab=repositories) | [Github](https://github.com/seoulsky-field) |

### 🌏 Contribution

- 권순호: semantic FPN(efficientNet-b3) implementation, data cleansing, data augmentation(randomcrop, cropout, resize...), 여러 모델 ensemble
- 서다빈: Data Cleansing, HRNet + OCR, Dense ViT, implement Albumentation in MMSegmentation, Data Augmentation (related to color/RGB), ensemble
- 서예현: Data Cleansing, implementation of mmsegmentation with custom data, UperNet + BEiT/Swin, MobileNetV3, BEiT ensemble
- 이상윤: Data Cleansing, Unet 3+_with timm_backbone 의 구현 ,Data Augmentation, Experiment with pytorch.amp
- 전경민: Data Cleansing, Modeling (MMSeg에 있는 모델 + 없는 모델), 필요한 python file 제작, Hyper-parameter Tuning, Ensemble 진행

### **❓Problem Definition**

- 바야흐로 **대량 생산, 대량 소비**의 시대에 우리는 많은 물건이 대량으로 생산되고 소비되는 시대를 삶에 따라 **쓰레기 대란, 매립지 부족**과 같은 사회 문제가 발생하였다.

![image](https://user-images.githubusercontent.com/74086829/169696825-3154d653-fdbb-4375-bc07-e43d8f57e107.png)

- 버려지는 쓰레기 중 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되지만, 잘못 분리배출 되면 그대로 폐기물로 분류되어 매립 또는 소각되기 때문에 분리수거는 사회적 환경 부담 문제를 줄일 수 있는 방법으로 제안되어 왔다.
- Deep Learning을 통해 쓰레기들을 자동으로 분류할 수 있는 모델을 개발하는 것이 프로젝트의 목표이다.
- 쓰레기를 줍는 드론, 쓰레기 배출 방지 비디오 감시, 인간의 쓰레기 분류를 돕는 AR 기술과 같은 여러 기술을 통해서 조금이나마 개선이 가능할 것으로 기대한다.

### 🚨 Competition Rules

- model로부터 예측된 mask의 size는 512 x 512 지만, 대회의 원활한 운영을 위해 output을 일괄적으로 256 x 256 으로 변경하여 score를 반영하게 되었습니다.

### 💾 Datasets

- 이미지 크기 : (512, 512)

![image](https://user-images.githubusercontent.com/74086829/169696846-003bab81-2aff-40d7-8859-f18572bd39e5.png)

- 11 class : Background, General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing
    - 참고 : train_all.json/train.json/val.json에는 background에 대한 annotation이 존재하지 않으므로 background (0) class 추가 (baseline 참고)

---

![image](https://user-images.githubusercontent.com/74086829/169696852-29e2494e-d6ec-4ca0-a18e-63c5dd434a9c.png)

### 💾 Annotations

- Datasets
    - id: 파일 안에서 image 고유 id, ex) 1
    - height: 512
    - width: 512
    - filename*: ex) batch*01_vt/002.jpg
- Annotations
    - id: 파일 안에 annotation 고유 id, ex) 1
    - segmentation: masking 되어 있는 고유의 좌표
    - category_id: 객체가 해당하는 class의 id
    - image_id: annotation이 표시된 이미지 고유 id

### 💻 **Development Environment**

- GPU: Tesla V100
- OS: Ubuntu 18.04.5LTS
- CPU: Intel Xeon
- Python : 3.8.5

### 📁 Project Structure

```markdown
level2-semantic-segmentation-level2-cv-16
├─ code
│   ├─ baseline_fcn_resnet50.ipynb
│   ├─ class_dict.csv
│   ├─ requirements.txt
│   ├─ utils.py
│   ├─ recycledataset.py
|   ├─ train_unetthreeplus.py
|   ├─ unetthreeplus.py
|   ├─ copy_pasteaug.py
│   ├─ saved
│   └─ submission
│       └─ sample_submission.csv
└─ data
    ├─ test.json
    ├─ train.json
    ├─ train_all.json
    ├─ val.json
    ├─ batch_01_vt
    │   ├─ 0002.jpg
    │   ├─ ...
    │   └─ 0005.jpg
    ├─ batch_02_vt
    │   ├─ 0001.jpg
    │   ├─ ...
    │   └─ 0003.jpg
    └─ batch_03
        ├─ 0001.jpg
        ├─ ...
        └─ 0003.jpg
```

- data
    - train.json: train image에 대한 annotation file (coco format) [80%]
    - val.json: validation image에 대한 annotation file (coco format) [20%]
    - train_all.json: train/validation 구분 없는 image에 대한 annotation file (coco format) [100%]
    - test.json: test image에 대한 annotation file (coco format)

### 👨‍🏫 Evaluation Methods

- Test set의 mIoU (Mean Intersection over Union)

![image](https://user-images.githubusercontent.com/74086829/169696871-1cf7538c-f77e-49d8-8261-0a5d39ca19dd.png)

### 💯 Final Score

![image](https://user-images.githubusercontent.com/74086829/169696981-e057f2d4-aa77-4b6f-a29a-08a1084e233b.png)

## 👀 How to Start

- Downloading the github repository

```powershell
git clone https://github.com/boostcampaitech3/level2-semantic-segmentation-level2-cv-16.git
cd level2-semantic-segmentation-level2-cv-16.git
```

- Installing the requirements for training

```powershell
pip install -r requirements.txt
```

- Using Stratified K-fold

```powershell
python stratified_kfold/kfold.py
```

- Hard-vote Ensemble

```powershell
python hard_vote_ensemble/ensemble.py
```

1. mmsegmentation ( [Link](https://www.notion.so/MMSegmentation-b2c1e27103ba48769ac3b7ad0876bc7c) )
- Installing prerequisites (without Albumentations)

```powershell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7/index.html
pip install mmsegmentation
```

- To use Albumentations with MMSegmentation

```powershell
cd mmsegmentation
conda create -n mmseg python=3.8
conda activate mmseg
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7/index.html
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
conda install -c conda-forge albumentations
conda install -c conda-forge wandb
pip install -e .
```

- Changing json format to png format ( [Link](https://github.com/boostcampaitech3/level2-semantic-segmentation-level2-cv-16/tree/Yehyun/json2png_format) )
    1. `/json2png_format/make_json_mask.ipynb` 실행
    2. `/json2png_format/make_json_image.py` 실행
- Training the model

```powershell
cd mmsegmentation
python tools/train.py <<directory_of_config_file>>
```

2. segmentation_modules_pytorch

```powershell
python train_unetthreeplus.py --data_path <<parentdir_path_of_datasets>>\
--train_path train.json --valid_path val.json --test_path test.json \
--encodername < ex) tu-efficientnet_b4 >
```

### 📄 [Experiments & Submission Report](https://www.notion.so/W15-17-Semantic-Segmentation-Project-Team-Medic-85cfbdc9fdaa4fa2a9d78b5b00a58d18)
