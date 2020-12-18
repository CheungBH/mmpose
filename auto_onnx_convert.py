import os

cfg = '/media/hkuit164/Elements/mmpose/configs/hand/resnet/panoptic/res18_panoptic_256x256_DUC.py'
weight = '/media/hkuit164/Elements/mmpose/work_dirs/res18_panoptic_256x256_DUC/epoch_20.pth'
output_path = '/media/hkuit164/Elements/mmpose/work_dirs/res18_panoptic_256x256_DUC/model.onnx'

cmd = 'python pytorch2onnx.py  {} {} --verify --output-file {}'.format(cfg,weight,output_path)
os.system(cmd)
