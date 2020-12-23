import os

folders = ["work_dirs/mobilenetv2_mpii_256x256", "work_dirs/shufflenetv2_mpii_256x256"]

cmds = []

for folder in folders:
    os.makedirs(os.path.join(folder, "plot"), exist_ok=True)
    for file in os.listdir(folder):
        if ".json" in file and "best" not in file:
            log_json = os.path.join(folder, file)
    out_img = os.path.join(os.path.join(folder, "plot", "loss.png"))
    cmds.append("python tools/analysis/analyze_logs.py plot_curve {} --keys mse_loss loss --out {}".format(log_json, out_img))

    out_img = os.path.join(os.path.join(folder, "plot", "acc.png"))
    cmds.append("python tools/analysis/analyze_logs.py plot_curve {} --keys acc_pose --out {}".format(log_json, out_img))

for cmd in cmds:
    os.system(cmd)

