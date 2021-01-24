import os

model_config = {
    "work_dirs/mobilenetv2_coco_512x512/latest.pth": "configs/bottom_up/mobilenet/coco/mobilenetv2_coco_512x512.py",
    # "work_dirs/res18_coco_512x512/latest.pth": "configs/bottom_up/resnet/coco/res50_coco_512x512.py",
    "work_dirs/res50_coco_512x512/latest.pth": "configs/bottom_up/resnet/coco/res50_coco_512x512.py"
}

img_path = "img/multi_people"
dest_path = img_path + "_vis"

cmds = []

for model, config in model_config.items():
    model_name = model.replace("work_dirs/", "").replace("/", "-")[:-4]
    if "bottom" in config:
        base_cmd = "python demo/bottom_up_img_demo.py "
    elif "top" in config:
        base_cmd = "python demo/top_down_img_demo_with_mmdet.py demo/mmdetection_cfg/faster_rcnn_r50_fpn_1x_coco.py " \
                   "http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
    else:
        raise ValueError("please assign your dataset type in config! (bottom-up or top-down)")
    for img_name in os.listdir(img_path):
        cmd = base_cmd + " {} {} --img-root {} --img {} --out-img-root {} --show".\
            format(config, model, img_path, img_name, os.path.join(dest_path, model_name))
        os.system(cmd)
        print(cmd)

