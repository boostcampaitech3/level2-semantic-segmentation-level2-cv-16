python train.py --config ../configs/_dabin_/ocrnet/ocrnet_hr48_cosann.py \
--work-dir ../work_dirs/DB_ocrnet_cosann_0 --seed 0;

python train.py --config ../configs/_dabin_/ocrnet/ocrnet_hr48_cosann.py \
--work-dir ../work_dirs/DB_ocrnet_cosann_16 --seed 7;

python train.py --config ../configs/_dabin_/ocrnet/ocrnet_hr48_cosann.py \
--work-dir ../work_dirs/DB_ocrnet_cosann_21 --seed 21;

python train.py --config ../configs/_dabin_/ocrnet/ocrnet_hr48_cosann.py \
--work-dir ../work_dirs/DB_ocrnet_cosann_42 --seed 42;

python train.py --config ../configs/_dabin_/ocrnet/ocrnet_hr48_cosann.py \
--work-dir ../work_dirs/DB_ocrnet_cosann_84 --seed 84;