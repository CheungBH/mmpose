import os

src_folder = "work_dirs"
onnx_sim_names = [os.path.join(src_folder, item, "model_sim.onnx") for item in os.listdir(src_folder)]
param_names = [os.path.join(src_folder, item, "model.param") for item in os.listdir(src_folder)]
bin_names = [os.path.join(src_folder, item, "model.bin") for item in os.listdir(src_folder)]


for onnx, param, bin in zip(onnx_sim_names, param_names, bin_names):
    cmd = "./onnx2ncnn {} {} {}".format(onnx, param, bin)
    os.system(cmd)
    print(cmd)
