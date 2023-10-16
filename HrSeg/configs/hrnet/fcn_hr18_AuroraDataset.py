_base_ = [
    # '../_base_/models/pspnet_r50-d8.py',
    # '../_base_/models/pspnet_unet_s5-d16.py',
    '../_base_/models/fcn_hr18.py',
    '../_base_/datasets/AuroraDataset_pipeline.py',
    '../_base_/default_runtime.py', 
    '../_base_/schedules/schedule_160k.py'
]
crop_size = (240, 240) # 输入图像尺寸，根据自己数据集情况修改
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)

