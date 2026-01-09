from libs.seed import set_seed, seed_worker

g = set_seed(0)

import utils as utils
import pandas as pd
from torch.utils.data import DataLoader
from datetime import datetime
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="このプログラムの説明（なくてもよい）")
    parser.add_argument("--testname")
    parser.add_argument("--timestamp")
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

MEP_EXCEL_PATH = "data/MEPs/All_data_upper1000.xlsx"
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
    MEMO = f"Random_Standardized_{arg_configs['model'][0]}_{arg_configs['criterion'][0]}_{LR_NAME}"
else:
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    TESTNAME = "Takase"
    MEMO = f"Random_Standardized_SelfMadeModel_LRe-5~4"

OUTPUT_SERVER = True
S = (
    f"../../../../../mnt/share/Takase/Result_DeepLearning2D/outputs_{TESTNAME}"
    if OUTPUT_SERVER
    else ""
)

# standardize_param = {
# "Teragiwa": [0.4698521905773838, 0.2519356287737737],
# "Takase": [0.4620869127245686, 0.2793042346296634],
# }

standardize_param = {
    "Aoki": [0.4768899706734551, 0.28754104563516675],
    "Kinugasa": [0.47838126750424853, 0.2809807611731357],
    "Masuda": [0.4788173584925882, 0.2892574428888596],
    "Takase": [0.4620869127245686, 0.2793042346296634],
    "Soma": [0.4680824736026843, 0.2782254497309693],
    "Teragiwa": [0.4698521905773838, 0.27970422085458174],
    "Yatsuda": [0.4679842391751864, 0.27957061268735184],
}


def main():
    df = pd.read_excel(MEP_EXCEL_PATH)
    for muscle in range(6):
        dataset = utils.Standardized_Dataset(EFIELD_PATH, df, muscle)
        configs_name, configs = utils.get_config(PARAMETER_PATH, arg_configs)

        for i, config in enumerate(configs):
            save_dir = f"{S}/muscle{muscle}/{TIMESTAMP}_{MEMO}/config{i}"
            print(save_dir)
            os.makedirs(save_dir, exist_ok=True)
            utils.save_config(
                f"{S}/muscle{muscle}/{TIMESTAMP}_{MEMO}/config{i}",
                configs_name[i],
            )
            train_dataset, val_dataset, test_dataset = (
                utils.Random_Split_dataset_test_val(dataset, TESTNAME, ratio=0.2)
            )

            dataset.set_standardize_param(
                mu=standardize_param[TESTNAME][0], sigma=standardize_param[TESTNAME][1]
            )

            train_dataloader = DataLoader(
                train_dataset,
                batch_size=config["batchsize"],
                num_workers=8,
                shuffle=True,
                pin_memory=True,
                worker_init_fn=seed_worker,
                persistent_workers=True,
                generator=g,
            )
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=1,
                shuffle=False,
                worker_init_fn=seed_worker,
                generator=g,
            )

            trainer = utils.Trainer(
                config["model"], config["criterion"], config["optimizer"]
            )

            history, bestModel, bestLoss, bestResult = trainer.train(
                train_dataloader, val_dataloader, config["epochs"]
            )

            utils.save_training_logs(save_dir, history, bestModel, bestLoss, bestResult)
        exit()


if __name__ == "__main__":
    main()
