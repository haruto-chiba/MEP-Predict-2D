from libs.seed import set_seed, seed_worker

g = set_seed(0)

import utils
import pandas as pd
import glob
from natsort import natsorted
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings

import argparse

MODE = "RANDOM"


def parse_args():
    parser = argparse.ArgumentParser(description="このプログラムの説明（なくてもよい）")
    parser.add_argument("--testname")
    parser.add_argument("--foldername")

    args = parser.parse_args()

    configs = {
        "testname": args.testname,
        "foldername": args.foldername,
    }

    return configs


arg_configs = parse_args()

warnings.simplefilter("ignore")

SIGMOID = False
MODEL_LIST = [
    utils.ResNet18,
    utils.SmallInputResNet18,
    utils.SelfMadeModel,
    utils.ResNet50_3D,
    utils.ResNet101_3D,
    utils.ResNet152_3D,
]
TESTNAME = arg_configs["testname"]
FOLDERNAME = arg_configs["foldername"]

S = f"../../../../../mnt/share/Takase/Result_DeepLearning2D/outputs_{TESTNAME}"

MEP_EXCEL_PATH = "data/MEPs/All_data_upper1000.xlsx"
EFIELD_PATH = "data/Efields_interpolated"

MUSCLE = 0

standardized_param_separated = {
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

standardize_param_random = {
    "Aoki": [0.4768899706734551, 0.28754104563516675],
    "Kinugasa": [0.47838126750424853, 0.2809807611731357],
    "Masuda": [0.4788173584925882, 0.2892574428888596],
    "Takase": [0.4620869127245686, 0.2793042346296634],
    "Soma": [0.4680824736026843, 0.2782254497309693],
    "Teragiwa": [0.4698521905773838, 0.27970422085458174],
    "Yatsuda": [0.4679842391751864, 0.27957061268735184],
}


def modelpath_to_outputfolderpath(modelpath):
    testfolderpath = "/".join(
        modelpath.replace(f"outputs_{TESTNAME}", f"test_{TESTNAME}").split("/")[:-1]
    )
    os.makedirs(testfolderpath, exist_ok=True)

    return testfolderpath


def load_model(modelPath, modelList, num_classes=1):
    state_dict = torch.load(modelPath, map_location="cpu")
    loaded_model = None

    for modelClass in modelList:
        try:
            if SIGMOID:
                loaded_model = modelClass(num_classes=num_classes, sigmoid=True)
                loaded_model.load_state_dict(state_dict)
            else:
                loaded_model = modelClass(num_classes=num_classes, sigmoid=False)
                loaded_model.load_state_dict(state_dict)

            # print(f"✅ Success: Model loaded as {modelClass.__name__}")
            return loaded_model
        except RuntimeError as e:
            continue
        except Exception as e:
            print(f"❌ An unexpected error occurred with {modelClass.__name__}: {e}")
            continue

    raise ValueError(
        f"❌ Error: The checkpoint file at {modelPath} did not match any of the provided model classes."
    )


def save_output(outputDir, history):
    os.makedirs(outputDir, exist_ok=True)

    df = pd.DataFrame(history)

    outputs = df["output"]
    targets = df["target"]
    outputs = (outputs - outputs.min()) / (outputs.max() - outputs.min())
    targets = (targets - targets.min()) / (targets.max() - targets.min())
    mse = ((outputs - targets) ** 2).sum() / len(outputs)

    df.to_csv(os.path.join(outputDir, f"{mse:.4f}.csv"), index=False)
    return


if MODE == "SEPARATED":
    model_paths = glob.glob(f"{S}/muscle0/{FOLDERNAME}/*/*/*.pt")
elif MODE == "RANDOM":
    model_paths = glob.glob(f"{S}/muscle0/{FOLDERNAME}/*/*.pt")
model_paths = natsorted(model_paths)


print(modelpath_to_outputfolderpath(model_paths[0]))
print(f"modelpath length: {len(model_paths)}")


def main():
    df = pd.read_excel(MEP_EXCEL_PATH)
    dataset = utils.Standardized_Dataset(EFIELD_PATH, df, MUSCLE)

    _, _, test_dataset = utils.Split_dataset_test_val(dataset, "Aoki", TESTNAME)

    for modelpath in tqdm(model_paths):
        output_folder = modelpath_to_outputfolderpath(modelpath)
        model = load_model(modelpath, MODEL_LIST)

        if MODE == "SEPARATED":
            valname = modelpath.split("/")[-2]
            dataset.set_standardize_param(
                mu=standardized_param_separated[TESTNAME][valname][0],
                sigma=standardized_param_separated[TESTNAME][valname][1],
            )
        elif MODE == "RANDOM":
            dataset.set_standardize_param(
                mu=standardize_param_random[TESTNAME][0],
                sigma=standardize_param_random[TESTNAME][1],
            )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            worker_init_fn=seed_worker,
            generator=g,
        )

        tester = utils.Tester(model)
        history = tester.test(test_dataloader)
        save_output(output_folder, history)


main()
