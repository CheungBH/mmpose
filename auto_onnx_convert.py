import os

model_config = {
    "work_dirs/mobilenetv2_mpii_256x256_DUC/latest.pth":
        "configs/top_down/mobilenet_v2/mpii/mobilenetv2_mpii_256x256_DUC.py",
}

for model, config in model_config.items():
    out_path = os.path.join("/".join(model.split("/")[:-1]), "model.onnx")
    cmd = 'python pytorch2onnx.py  {} {} --verify --output-file {}'.format(config, model, out_path)
    print(out_path)
