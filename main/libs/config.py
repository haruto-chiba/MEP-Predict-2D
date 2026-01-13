import yaml
import numpy as np
from itertools import product
import torch.optim as optim
import torch.nn as nn
import copy

from libs.model import (
    ResNet18,
    SmallInputResNet18,
    SelfMadeModel,
    ResNet50_3D,
    ResNet101_3D,
    ResNet152_3D,
)

SIGMOID = False


def load_yaml(yamlpath, yamldict=None):
    if yamldict["model"][0]:
        yml = copy.deepcopy(yamldict)
        del yml["testname"]
        del yml["timestamp"]
    else:
        with open(yamlpath, "r") as f:
            yml = yaml.safe_load(f)

    keys = yml.keys()
    values = yml.values()

    return [dict(zip(keys, v)) for v in product(*values)]


def get_config(yamlpath, yamldict=None):
    all_cfgs = []
    if yamldict["model"] != None:
        cfg_names = load_yaml(yamlpath, yamldict)
    else:
        cfg_names = load_yaml(yamlpath)
    all_cfg_names = []
    for cfg_name in cfg_names:
        for _ in range(int(cfg_name["lr_num"])):
            cfg = {}

            # Model
            if cfg_name["model"] == "ResNet18":
                if SIGMOID:
                    cfg["model"] = ResNet18(num_classes=1, sigmoid=True)
                else:
                    cfg["model"] = ResNet18(num_classes=1, sigmoid=False)
            elif cfg_name["model"] == "SmallResNet":
                if SIGMOID:
                    cfg["model"] = SmallInputResNet18(num_classes=1, sigmoid=True)
                else:
                    cfg["model"] = SmallInputResNet18(num_classes=1, sigmoid=False)
            elif cfg_name["model"] == "SelfMadeModel":
                if SIGMOID:
                    cfg["model"] = SelfMadeModel(num_classes=1, sigmoid=True)
                else:
                    cfg["model"] = SelfMadeModel(num_classes=1, sigmoid=False)

            elif cfg_name["model"] == "ResNet50_3D":
                if SIGMOID:
                    cfg["model"] = ResNet50_3D(num_classes=1, sigmoid=True)
                else:
                    cfg["model"] = ResNet50_3D(num_classes=1, sigmoid=False)
            elif cfg_name["model"] == "ResNet101_3D":
                if SIGMOID:
                    cfg["model"] = ResNet101_3D(num_classes=1, sigmoid=True)
                else:
                    cfg["model"] = ResNet101_3D(num_classes=1, sigmoid=False)
            elif cfg_name["model"] == "ResNet152_3D":
                if SIGMOID:
                    cfg["model"] = ResNet152_3D(num_classes=1, sigmoid=True)
                else:
                    cfg["model"] = ResNet152_3D(num_classes=1, sigmoid=False)

            # Criterion
            if cfg_name["criterion"] == "MSE":
                cfg["criterion"] = nn.MSELoss()
            elif cfg_name["criterion"] == "MAE":
                cfg["criterion"] = nn.L1Loss()
            elif cfg_name["criterion"] == "BCEWithLogitsLoss":
                cfg["criterion"] = nn.BCEWithLogitsLoss()

            # Learning Rate
            lr = 10 ** np.random.uniform(
                np.log10(cfg_name["lr_min"]), np.log10(cfg_name["lr_max"])
            )

            cfg["lr"] = lr
            cfg_name["lr"] = lr

            # Weight Decay
            cfg["weight_decay"] = cfg_name["weight_decay"]

            # Optimizer
            if cfg_name["optimizer"] == "AdamW":
                cfg["optimizer"] = optim.AdamW(
                    cfg["model"].parameters(),
                    lr=cfg["lr"],
                    weight_decay=cfg["weight_decay"],
                )

            # Epochs
            cfg["epochs"] = cfg_name["epochs"]

            # Batch Size
            cfg["batchsize"] = cfg_name["batchsize"]

            all_cfgs.append(cfg)
            all_cfg_names.append(copy.deepcopy(cfg_name))

    return all_cfg_names, all_cfgs


def get_config_test(yamlpath):
    cfg_names = load_yaml(yamlpath)
    print(cfg_names)


if __name__ == "__main__":
    path = "params.yaml"
    all_cfg_names, cfg = load_yaml(path)
    print(all_cfg_names)
