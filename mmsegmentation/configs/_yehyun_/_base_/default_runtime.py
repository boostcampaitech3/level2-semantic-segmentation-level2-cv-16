import wandb

# yapf:disable
log_config = dict(
    interval=150,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=True),
        dict(
            type="WandbLoggerHook",
            init_kwargs=dict(
                entity="medic", project="semantic-segmentation", name="YH_DLv3_baseline"
            ),
        )
        # dict(type='TensorboardLoggerHook')
    ],
)

# yapf:enable
dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = None
resume_from = None
workflow = [("train", 1)]
cudnn_benchmark = True
