import os

model_config = {
    "work_dirs/mobilenetv2_mpii_256x256_DUC/latest.pth":"configs/top_down/mobilenet_v2/mpii/mobilenetv2_mpii_256x256_DUC.py",\

}

img_path = "img/person"
dest_path = img_path + "_vis"

cmds = []

for model, config in model_config.items():
    model_name = model.replace("work_dirs/", "").replace("/", "-")[:-4]
    for img_name in os.listdir(img_path):
        cmd = "python demo/top_down_img_demo_with_mmdet.py demo/mmdetection_cfg/faster_rcnn_r50_fpn_1x_coco.py \
    http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    {} {} --img-root {} --img {} --out-img-root {} --show".\
            format(config, model, img_path, img_name, os.path.join(dest_path, model_name))
        # cmds.append(cmd)
        os.system(cmd)
        print(cmd)

# print(cmds)
