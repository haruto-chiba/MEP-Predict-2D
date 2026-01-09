import torch
from torch.utils.data import DataLoader
from typing import Dict


class Tester:
    def __init__(self, model, device="cpu"):
        self.device = device
        self.model = model.to(self.device)

        self.all_predictions = []
        self.all_targets = []
        self.all_names = []

    def test(self, testLoader: DataLoader) -> Dict[str, list]:
        self.model.eval()
        self.all_predictions.clear()
        self.all_targets.clear()
        self.all_names.clear()

        with torch.no_grad():
            for images, targets, names in testLoader:
                images = images.float().to(self.device)
                targets = targets.float().to(self.device)

                outputs = self.model(images)

                self.all_predictions.extend(outputs.cpu().tolist()[0])
                self.all_targets.extend(targets.cpu().tolist()[0])
                self.all_names.extend(names)

        return {
            "output": self.all_predictions,
            "target": self.all_targets,
            "name": self.all_names,
        }
