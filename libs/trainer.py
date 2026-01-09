import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict
from tqdm import tqdm
import copy


class EarlyStopping:
    def __init__(self, patience: int = 5, delta: float = 0.0):
        """早期終了のクラス

        Args:
            patience (int): 改善が見られなくても許容するエポック数
            delta (float): 何をもって改善とするかの閾値
        """
        self.patience = patience
        self.delta = delta
        self.best_loss = np.inf  # 初期値として最良損失を無限大に設定
        self.counter = 0
        self.early_stop = False
        self.best_model = None
        self.best_result = None

    def __call__(self, val_loss, model, result):
        """早期終了の判断"""
        if val_loss < self.best_loss - self.delta:
            print(f"{self.best_loss} -> {val_loss}")
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model)
            self.best_result = copy.deepcopy(result)
            self.counter = 0  # 改善があったのでカウンターをリセット
        else:
            self.counter += 1  # 改善がなかった場合、カウンターを増加
            if self.counter >= self.patience:
                self.early_stop = True  # 許容エポック数を超えたら早期終了

        return self.early_stop


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        patience: int = 5,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.earlyStopping = EarlyStopping(patience=patience)
        self.device = device
        self.history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_results": [],
        }

    def train_one_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        train_epoch_loss = 0

        for images, targets, name in train_loader:
            images, targets = images.float().to(self.device), targets.float().to(
                self.device
            )
            self.optimizer.zero_grad()

            outputs = self.model(images)

            loss = self.criterion(outputs, targets)

            loss.backward()
            self.optimizer.step()

            train_epoch_loss += loss.item()

        train_epoch_loss /= len(train_loader)

        return train_epoch_loss

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        val_epoch_loss = 0
        results = []
        with torch.no_grad():
            for images, targets, name in val_loader:
                images, targets = images.float().to(self.device), targets.float().to(
                    self.device
                )

                outputs = self.model(images)

                loss = self.criterion(outputs, targets)
                val_epoch_loss += loss.item()
                results.append(
                    [name, outputs[0].to("cpu").item(), targets[0].to("cpu").item()]
                )
        val_epoch_loss /= len(val_loader)
        earlyStopping_flag = self.earlyStopping(val_epoch_loss, self.model, results)

        return val_epoch_loss, results, earlyStopping_flag

    def train(self, train_loader, val_loader, num_epochs):
        for epoch in range(num_epochs):
            train_loss = self.train_one_epoch(train_loader)
            val_loss, val_results, stopping_flag = self.validate(val_loader)
            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_results"].append(val_results)

            if stopping_flag:
                break

        return (
            self.history,
            self.earlyStopping.best_model.to("cpu"),
            self.earlyStopping.best_loss,
            self.earlyStopping.best_result,
        )
