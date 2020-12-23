import os

configs = [

]

cmds = ["CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/bottom_up/resnet/coco/res50_coco_512x512.py 4",
        "CUDA_VISIBLE_DEVICES=0,1,2 ./tools/dist_train.sh configs/bottom_up/resnet/coco/res18_coco_512x512.py 3"
        ]

# for config in configs:
#     cmds.append("python tools/train.py {}".format(config))

for cmd in cmds:
    print(cmd)
    os.system(cmd)

