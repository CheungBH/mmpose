import os

model_config = {
    "work_dirs/mobilenetv2_mpii_256x256_DUC/latest.pth":
        "configs/top_down/mobilenet_v2/mpii/mobilenetv2_mpii_256x256_DUC.py",
    "work_dirs/mobilenetv2_mpii_256x256_1DUC/latest.pth":
        "configs/top_down/mobilenet_v2/mpii/mobilenetv2_mpii_256x256_1DUC.py",

}

eval = "PCKh"


def get_new_folder(path):
    items = path.split("/")
    old_path = "/".join(items[:-1])
    items[-2] += "_test"
    return "/".join(items[:-1]), old_path


for idx, (model, config) in enumerate(model_config.items()):
    print("-------------------------------------------------------\nEvaluating Model {}: {}".format(idx+1, model))
    new_config = config.replace(".py", "_test.py")
    os.system("cp {} {}".format(config, new_config))
    cmd = "python tools/test.py {} {} --eval {}".format(new_config, model, eval)
    os.system(cmd)
    new_folder, old_folder = get_new_folder(model)
    os.system("cp {} {}".format(os.path.join(new_folder, "pred.mat"), os.path.join(old_folder, "pred_test.mat")))
    os.system("rm -r {}".format(new_folder))
    os.system("rm {}".format(new_config))

# get_new_folder("work_dirs/mobilenetv2_mpii_256x256_DUC/latest.pth"))
