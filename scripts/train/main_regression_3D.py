from libs.seed import set_seed, seed_worker

g = set_seed(0)

import utils as utils
import pandas as pd
from torch.utils.data import DataLoader
import os
import datetime
import warnings

warnings.simplefilter("ignore")

MEP_EXCEL_PATH = "data/MEPs/All_data_upper1000.xlsx"
EFIELD_PATH = "../../../../../mnt/share/Takase/midterm/Precortex_Clipped_Efield"
PARAMETER_PATH = "params.yaml"
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
TESTNAME = "Teragiwa"
MEMO = "Standardized_ResNet101_3D"


standardized_parameter = {
    "Aoki": [3.6134451851638696, 16.03927988020625],
    "Kinugasa": [3.537732119259702, 15.698294595286734],
    "Masuda": [3.575465841293335, 15.843230764553228],
    "Soma": [3.3340477701082674, 14.69111782041492],
    "Takase": [3.4216744258593277, 15.392898210220906],
    "Yatsuda": [3.545191117414931, 15.837346911562081],
}

start_time = datetime.datetime.now()


def main():
    df = pd.read_excel(MEP_EXCEL_PATH)
    for muscle in range(6):
        dataset = utils.Standardized_Dataset_3D(EFIELD_PATH, df, muscle)
        configs_name, configs = utils.get_config(PARAMETER_PATH)

        for i, config in enumerate(configs):
            for valname in ["Aoki", "Kinugasa", "Masuda", "Takase", "Yatsuda", "Soma"]:
                save_dir = (
                    f"outputs_3D/muscle{muscle}/{TIMESTAMP}_{MEMO}/config{i}/{valname}"
                )

                time = start_time = datetime.datetime.now()
                print(
                    f'{save_dir}    {start_time.strftime("%Y-%m-%d %H:%M:%S")} -> {time.strftime("%Y-%m-%d %H:%M:%S")}'
                )
                os.makedirs(save_dir, exist_ok=True)
                utils.save_config(
                    f"outputs_3D/muscle{muscle}/{TIMESTAMP}_{MEMO}/config{i}",
                    configs_name[i],
                )
                train_dataset, val_dataset, test_dataset = utils.Split_dataset_test_val(
                    dataset, valname, TESTNAME
                )

                dataset.set_standardize_param(
                    mu=standardized_parameter[valname][0],
                    sigma=standardized_parameter[valname][1],
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

                utils.save_training_logs(
                    save_dir, history, bestModel, bestLoss, bestResult
                )
        exit()


if __name__ == "__main__":
    main()
