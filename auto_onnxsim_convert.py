import os

src_folder = "work_dirs"
onnx_names = [os.path.join(src_folder, item, "model.onnx") for item in os.listdir(src_folder)]
onnxsim_names = [os.path.join(src_folder, item, "model_sim.onnx") for item in os.listdir(src_folder)]

for src, dest in zip(onnx_names, onnxsim_names):
    cmd = "python -m onnxsim {} {}".format(src, dest)
    os.system(cmd)
    print(cmd)
