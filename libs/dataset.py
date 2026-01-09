import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, Subset
import time
import os
import random


class Binary_Dataset(Dataset):
    def __init__(self, efield_folder_path, mep_df, muscle):
        self.mep_df = mep_df
        self.muscle = muscle
        self.efields = []
        self.targets = []
        self.names = []
        self.dict = {
            "Aoki": 1.7462796,
            "Kinugasa": 2.007464,
            "Masuda": 1.7225424,
            "Soma": 2.5477223,
            "Takase": 2.2916286,
            "Teragiwa": 2.3143313,
            "Yatsuda": 1.8813754,
        }

        print("▶️  Loading Start ...")
        s = time.time()
        for row in tqdm(self.mep_df.to_dict(orient="records")):
            path = row["group_Sample_name"]
            name = path.split("_")[0]
            efield = (
                np.load(os.path.join(efield_folder_path, path + ".npy"))
                / self.dict[name]
            )
            target = target = np.array(row[f"Muscle{muscle}"])

            self.efields.append(efield)
            self.targets.append(target)
            self.names.append(path)

        e = time.time()
        print(f"✅ Loading End in {int((e-s)//60)}m {int((e-s)%60)}s")

    def __len__(self):
        return len(self.mep_df)

    def __getitem__(self, index):
        return (
            torch.tensor(self.efields[index]).unsqueeze(0),
            torch.tensor(self.targets[index]).unsqueeze(-1),
            self.names[index],
        )


class Standardized_Dataset(Dataset):
    def __init__(
        self, efield_folder_path, mep_df, muscle, mu=0, sigma=1.0, normalization=False
    ):
        self.mep_df = mep_df
        self.muscle = muscle
        self.efields = []
        self.targets = []
        self.names = []
        self.mu = mu
        self.sigma = sigma
        self.normalization = normalization
        self.normalization_dict = {}

        print("▶️  Loading Start ...")
        s = time.time()
        for row in tqdm(self.mep_df.to_dict(orient="records")):
            path = row["group_Sample_name"]
            name = path.split("_")[0]
            efield = np.load(os.path.join(efield_folder_path, path + ".npy"))
            target = np.array(row[f"Muscle{muscle}"])

            self.efields.append(efield)
            self.targets.append(target)
            self.names.append(path)

        e = time.time()
        print(f"✅ Loading End in {int((e-s)//60)}m {int((e-s)%60)}s")

    def __len__(self):
        return len(self.mep_df)

    def __getitem__(self, index):
        if self.normalization:

            name = self.names[index].split("_")[0]
            normalized_target = (
                self.targets[index] - self.normalization_dict[name][0]
            ) / (self.normalization_dict[name][1] - self.normalization_dict[name][0])

            return (
                torch.tensor(
                    ((self.efields[index] / 10**4) - self.mu) / self.sigma
                ).unsqueeze(0),
                torch.tensor(normalized_target).unsqueeze(-1),
                self.names[index],
            )
        else:
            return (
                torch.tensor((self.efields[index] - self.mu) / self.sigma).unsqueeze(0),
                torch.tensor(self.targets[index]).unsqueeze(-1),
                self.names[index],
            )

    def set_standardize_param(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def set_normalization_param(self, normalization_dict):
        self.normalization = True
        self.normalization_dict = normalization_dict


class Standardized_Dataset_3D(Dataset):
    def __init__(
        self, efield_folder_path, mep_df, muscle, mu=0, sigma=1.0, normalization=False
    ):
        self.mep_df = mep_df
        self.muscle = muscle
        self.efields = []
        self.targets = []
        self.names = []
        self.mu = mu
        self.sigma = sigma
        self.normalization = normalization
        self.normalization_dict = {}

        print("▶️  Loading Start ...")
        s = time.time()
        for row in tqdm(self.mep_df.to_dict(orient="records")):
            path = row["group_Sample_name"]
            name = path.split("_")[0]
            efield = np.load(os.path.join(efield_folder_path, path + ".npy")).astype(
                np.float32
            )
            target = np.array(row[f"Muscle{muscle}"]).astype(np.float32)

            self.efields.append(efield)
            self.targets.append(target)
            self.names.append(path)

        e = time.time()
        print(f"✅ Loading Finished in {int((e-s)//60)}m {int((e-s)%60)}s")

    def __len__(self):
        return len(self.mep_df)

    def __getitem__(self, index):
        if self.normalization:

            name = self.names[index].split("_")[0]
            normalized_targets = (
                self.targets[index] - self.normalization_dict[name][0]
            ) / (self.normalization_dict[name][1] - self.normalization_dict[name][0])

            return (
                torch.tensor(
                    ((self.efields[index] / 10**4) - self.mu) / self.sigma
                ).unsqueeze(0),
                torch.tensor(normalized_targets).unsqueeze(-1),
                self.names[index],
            )
        else:
            return (
                torch.tensor(
                    ((self.efields[index] / 10**4) - self.mu) / self.sigma
                ).unsqueeze(0),
                torch.tensor(self.targets[index]).unsqueeze(-1),
                self.names[index],
            )

    def set_standardize_param(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def set_normalization_param(self, normalization_dict):
        self.normalization = True
        self.normalization_dict = normalization_dict


class Standardized_Dataset_Binary(Dataset):
    def __init__(self, efield_folder_path, mep_df, muscle, mu=0, sigma=1.0):
        self.mep_df = mep_df
        self.muscle = muscle
        self.efields = []
        self.targets = []
        self.names = []
        self.mu = mu
        self.sigma = sigma

        print("▶️  Loading Start ...")
        s = time.time()
        for row in tqdm(self.mep_df.to_dict(orient="records")):
            path = row["group_Sample_name"]
            name = path.split("_")[0]
            efield = np.load(os.path.join(efield_folder_path, path + ".npy"))
            target = np.array(row[f"Muscle{muscle}"])

            self.efields.append(efield)
            self.targets.append(target)
            self.names.append(path)

        e = time.time()
        print(f"✅ Loading End in {int((e-s)//60)}m {int((e-s)%60)}s")

    def __len__(self):
        return len(self.mep_df)

    def __getitem__(self, index):
        return (
            torch.tensor((self.efields[index] - self.mu) / self.sigma).unsqueeze(0),
            torch.tensor(self.targets[index]).unsqueeze(-1),
            self.names[index],
        )

    def set_standardize_param(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma


def Split_dataset_test_val(dataset, val_name, test_name):
    indexes = list(range(len(dataset)))
    if test_name == "Aoki":
        test_idx = indexes[:237]
    elif test_name == "Kinugasa":
        test_idx = indexes[237:477]
    elif test_name == "Masuda":
        test_idx = indexes[477:717]
    elif test_name == "Takase":
        test_idx = indexes[717:957]
    elif test_name == "Teragiwa":
        test_idx = indexes[957:1197]
    elif test_name == "Yatsuda":
        test_idx = indexes[1197:1437]
    elif test_name == "Soma":
        test_idx = indexes[1437:1677]

    if val_name == "Aoki":
        val_idx = indexes[:237]
    elif val_name == "Kinugasa":
        val_idx = indexes[237:477]
    elif val_name == "Masuda":
        val_idx = indexes[477:717]
    elif val_name == "Takase":
        val_idx = indexes[717:957]
    elif val_name == "Teragiwa":
        val_idx = indexes[957:1197]
    elif val_name == "Yatsuda":
        val_idx = indexes[1197:1437]
    elif val_name == "Soma":
        val_idx = indexes[1437:1677]

    train_idx = list(set(indexes) - set(val_idx) - set(test_idx))

    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
    )


def Random_Split_dataset_test_val(dataset, test_name, ratio):
    indexes = list(range(len(dataset)))
    val_idx = []

    Aoki_indexes = indexes[:237]
    Kinugasa_indexes = indexes[237:477]
    Masuda_indexes = indexes[477:717]
    Takase_indexes = indexes[717:957]
    Teragiwa_indexes = indexes[957:1197]
    Yatsuda_indexes = indexes[1197:1437]
    Soma_indexes = indexes[1437:1677]

    if test_name == "Aoki":
        test_idx = Aoki_indexes
    elif test_name == "Kinugasa":
        test_idx = Kinugasa_indexes
    elif test_name == "Masuda":
        test_idx = Masuda_indexes
    elif test_name == "Takase":
        test_idx = Takase_indexes
    elif test_name == "Teragiwa":
        test_idx = Teragiwa_indexes
    elif test_name == "Yatsuda":
        test_idx = Yatsuda_indexes
    elif test_name == "Soma":
        test_idx = Soma_indexes

    """

    random.shuffle(Aoki_indexes)
    random.shuffle(Kinugasa_indexes)
    random.shuffle(Masuda_indexes)
    random.shuffle(Takase_indexes)
    random.shuffle(Teragiwa_indexes)
    random.shuffle(Yatsuda_indexes)
    random.shuffle(Soma_indexes)

    val_idx.extend(Aoki_indexes[: int(240 * ratio)])
    val_idx.extend(Kinugasa_indexes[: int(240 * ratio)])
    val_idx.extend(Masuda_indexes[: int(240 * ratio)])
    val_idx.extend(Takase_indexes[: int(240 * ratio)])
    val_idx.extend(Teragiwa_indexes[: int(240 * ratio)])
    val_idx.extend(Yatsuda_indexes[: int(240 * ratio)])
    val_idx.extend(Soma_indexes[: int(240 * ratio)])
    """

    val_idx.extend(random.sample(Aoki_indexes[:120], int(240 * ratio / 2)))
    val_idx.extend(random.sample(Aoki_indexes[120:], int(240 * ratio / 2)))
    val_idx.extend(random.sample(Kinugasa_indexes[:120], int(240 * ratio / 2)))
    val_idx.extend(random.sample(Kinugasa_indexes[120:], int(240 * ratio / 2)))
    val_idx.extend(random.sample(Masuda_indexes[:120], int(240 * ratio / 2)))
    val_idx.extend(random.sample(Masuda_indexes[120:], int(240 * ratio / 2)))
    val_idx.extend(random.sample(Takase_indexes[:120], int(240 * ratio / 2)))
    val_idx.extend(random.sample(Takase_indexes[120:], int(240 * ratio / 2)))
    val_idx.extend(random.sample(Yatsuda_indexes[:120], int(240 * ratio / 2)))
    val_idx.extend(random.sample(Yatsuda_indexes[120:], int(240 * ratio / 2)))
    val_idx.extend(random.sample(Soma_indexes[:120], int(240 * ratio / 2)))
    val_idx.extend(random.sample(Soma_indexes[120:], int(240 * ratio / 2)))
    # print(val_idx)
    random.shuffle(val_idx)

    train_idx = list(set(indexes) - set(val_idx) - set(test_idx))

    random.shuffle(train_idx)

    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
    )


if __name__ == "__main__":
    df = pd.read_excel("data/MEPs/All_data_upper1000.xlsx")

    dataset = Standardized_Dataset(
        "../../../../../mnt/share/Takase/midterm/Precortex_Clipped_Efield", df, 0
    )

    print(dataset[2])
