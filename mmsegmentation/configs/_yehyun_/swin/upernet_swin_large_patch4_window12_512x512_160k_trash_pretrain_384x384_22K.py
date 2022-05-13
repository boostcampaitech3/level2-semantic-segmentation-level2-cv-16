_base_ = [
    "upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_" "pretrain_224x224_1K.py"
]

model = dict(
    pretrained="pretrain/swin_large_patch4_window12_384_22k.pth",
    backbone=dict(
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
    ),
    decode_head=dict(in_channels=[192, 384, 768, 1536], num_classes=11),
    auxiliary_head=dict(in_channels=768, num_classes=11),
)

data = dict(samples_per_gpu=4)
