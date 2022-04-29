_base_ = [
    "../_base_/models/deeplabv3plus_r50-d8.py",
    "../_base_/datasets/trash_dataset.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_80k.py",
]
# pretrained = 'https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r101-d8_512x1024_80k_cityscapes/deeplabv3plus_r101-d8_512x1024_80k_cityscapes_20200606_114143-068fcfe9.pth'

# model = dict(
#     backbone=dict(
#         init_cfg=dict(type='Pretrained', checkpoint=pretrained)
#     ),
# )
