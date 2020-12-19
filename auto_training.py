import os

configs = [

]

cmds = []

for config in configs:
    cmds.append("python tools/train.py {}".format(config))

for cmd in cmds:
    print(cmd)
    os.system(cmd)

