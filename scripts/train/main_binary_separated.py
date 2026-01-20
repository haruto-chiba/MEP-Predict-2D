import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import utils as utils
import pandas as pd
from torch.utils.data import DataLoader
from datetime import datetime
import os
import argparse

g = utils.set_seed(0)


def parse_args():
    parser = argparse.ArgumentParser(description="このプログラムの説明（なくてもよい）")
    parser.add_argument("--testname")
    parser.add_argument("--timestamp")
    parser.add_argument("--muscle", type=int)
    parser.add_argument("--model")
    parser.add_argument("--criterion")
    parser.add_argument("--optimizer")
    parser.add_argument("--lr_min", type=float)
    parser.add_argument("--lr_max", type=float)
    parser.add_argument("--lr_num", type=int)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batchsize", type=int)

    args = parser.parse_args()

    configs = {
        "testname": args.testname,
        "timestamp": args.timestamp,
        "muscle": [args.muscle],
        "model": [args.model],
        "criterion": [args.criterion],
        "optimizer": [args.optimizer],
        "lr_min": [args.lr_min],
        "lr_max": [args.lr_max],
        "lr_num": [args.lr_num],
        "weight_decay": [args.weight_decay],
        "epochs": [args.epochs],
        "batchsize": [args.batchsize],
    }

    return configs


arg_configs = parse_args()
print(arg_configs)

MEP_EXCEL_PATH = "data/MEPs/All_data_upper1000_binary.xlsx"
EFIELD_PATH = "data/Efields_interpolated"
PARAMETER_PATH = "params.yaml"


if arg_configs["model"][0]:
    TIMESTAMP = arg_configs["timestamp"]
    TESTNAME = arg_configs["testname"]

    if arg_configs["lr_min"][0] == 0.0004:
        LR_NAME = "LRe-4~3"
    elif arg_configs["lr_min"][0] == 0.004:
        LR_NAME = "LRe-3~2"
    else:
        raise ValueError

    MEMO = f"Binary_Standardized_{arg_configs['model'][0]}_{arg_configs['criterion'][0]}_{LR_NAME}"


else:
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    TESTNAME = "Takase"
    MEMO = f"Binary_Standardized_SelfMadeModel_LRe-5~4"

OUTPUT_SERVER = True
S = (
    f"../../../../../mnt/share/Takase/Result_DeepLearning2D/outputs_{TESTNAME}"
    if OUTPUT_SERVER
    else ""
)

standardized_parameter = {
    "Aoki": {
        "Kinugasa": [0.4858878186119927, 0.28700012352055027],
        "Masuda": [0.48641003757052953, 0.2966698596725366],
        "Takase": [0.466375328763326, 0.285437149913909],
        "Soma": [0.4735550129148695, 0.2840493781641669],
        "Teragiwa": [0.4756742489920722, 0.28574133205626184],
        "Yatsuda": [0.4734373771879408, 0.2856288585963328],
    },
    "Kinugasa": {
        "Aoki": [0.4858878186119927, 0.28700012352055027],
        "Masuda": [0.4882242011014747, 0.2889878298531165],
        "Takase": [0.46813927999149624, 0.2775442507395921],
        "Soma": [0.47533695833890827, 0.27606699472893925],
        "Teragiwa": [0.47746150578472546, 0.2777982794112282],
        "Yatsuda": [0.47521902778559627, 0.2776967082565608],
    },
    "Masuda": {
        "Aoki": [0.48641003757052953, 0.2966698596725366],
        "Kinugasa": [0.4882242011014747, 0.2889878298531165],
        "Takase": [0.46866280776947794, 0.2875890663614488],
        "Soma": [0.47586048611688997, 0.28615050130372544],
        "Teragiwa": [0.4779850335627072, 0.2878172738677873],
        "Yatsuda": [0.475742555563578, 0.2877233103942014],
    },
    "Takase": {
        "Aoki": [0.466375328763326, 0.285437149913909],
        "Kinugasa": [0.46813927999149624, 0.2775442507395921],
        "Masuda": [0.46866280776947794, 0.2875890663614488],
        "Soma": [0.4557755650069115, 0.2736828564774979],
        "Teragiwa": [0.45790011245272877, 0.27557997456365674],
        "Yatsuda": [0.4556576344535995, 0.27531830296570603],
    },
    "Soma": {
        "Aoki": [0.4735550129148695, 0.2840493781641669],
        "Kinugasa": [0.47533695833890827, 0.27606699472893925],
        "Masuda": [0.47586048611688997, 0.28615050130372544],
        "Takase": [0.4557755650069115, 0.2736828564774979],
        "Teragiwa": [0.4650977908001408, 0.274360881700204],
        "Yatsuda": [0.4628553128010116, 0.27415691763249433],
    },
    "Teragiwa": {
        "Aoki": [0.4756742489920722, 0.28574133205626184],
        "Kinugasa": [0.47746150578472546, 0.2777982794112282],
        "Masuda": [0.4779850335627072, 0.2878172738677873],
        "Takase": [0.45790011245272877, 0.27557997456365674],
        "Soma": [0.4650977908001408, 0.274360881700204],
        "Yatsuda": [0.4649798602468288, 0.27599628493647604],
    },
    "Yatsuda": {
        "Aoki": [0.4734373771879408, 0.2856288585963328],
        "Kinugasa": [0.47521902778559627, 0.2776967082565608],
        "Masuda": [0.475742555563578, 0.2877233103942014],
        "Takase": [0.4556576344535995, 0.27531830296570603],
        "Soma": [0.4628553128010116, 0.27415691763249433],
        "Teragiwa": [0.4649798602468288, 0.27599628493647604],
    },
}


def main():
    df = pd.read_excel(MEP_EXCEL_PATH)
    muscle = arg_configs["muscle"][0]

    dataset = utils.Standardized_Dataset(EFIELD_PATH, df, muscle)
    configs_name, configs = utils.get_config(PARAMETER_PATH, arg_configs)
    for i, config in enumerate(configs):
        for valname in [
            "Aoki",
            "Kinugasa",
            "Masuda",
            "Takase",
            "Yatsuda",
            "Soma",
            "Teragiwa",
        ]:
            if TESTNAME == valname:
                continue
            save_dir = f"{S}/muscle{muscle}/{TIMESTAMP}_{MEMO}/config{i}/{valname}"
            print(save_dir)
            os.makedirs(save_dir, exist_ok=True)
            utils.save_config(
                f"{S}/muscle{muscle}/{TIMESTAMP}_{MEMO}/config{i}",
                configs_name[i],
            )
            train_dataset, val_dataset, test_dataset = utils.Split_dataset_test_val(
                dataset, valname, TESTNAME
            )

            dataset.set_standardize_param(
                mu=standardized_parameter[TESTNAME][valname][0],
                sigma=standardized_parameter[TESTNAME][valname][1],
            )

            train_dataloader = DataLoader(
                train_dataset,
                batch_size=config["batchsize"],
                num_workers=8,
                shuffle=True,
                pin_memory=True,
                worker_init_fn=utils.seed_worker,
                persistent_workers=True,
                generator=g,
            )
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=1,
                shuffle=False,
                worker_init_fn=utils.seed_worker,
                generator=g,
            )

            trainer = utils.Trainer(
                config["model"],
                config["criterion"],
                config["optimizer"],
                patience=20,
            )

            history, bestModel, bestLoss, bestResult = trainer.train(
                train_dataloader, val_dataloader, config["epochs"]
            )

            utils.save_training_logs(save_dir, history, bestModel, bestLoss, bestResult)


if __name__ == "__main__":
    main()
