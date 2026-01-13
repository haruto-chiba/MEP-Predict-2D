import pandas as pd
import torch
import os
import yaml


def save_config(save_dir, cfg_name):
    with open(os.path.join(save_dir, "parameter.yaml"), "w") as f:
        yaml.dump(
            cfg_name, f, default_flow_style=False, sort_keys=False, allow_unicode=True
        )

    return


def save_training_logs(save_dir, history, best_model, best_loss, best_result):
    # Model
    torch.save(
        best_model.state_dict(),
        os.path.join(save_dir, f"best_model_loss={best_loss:.4f}.pt"),
    )

    # History
    with open(
        os.path.join(save_dir, f"training_log_loss={best_loss:.4f}.log"), "w"
    ) as f:
        epochs = history["epoch"]
        train_losses = history["train_loss"]
        val_losses = history["val_loss"]
        val_results = history["val_results"]

        for i in range(len(epochs)):
            f.write(
                f"\nEpoch: {epochs[i]+1}\tTrain_loss: {train_losses[i]:.4f}\tVal_loss: {val_losses[i]:.4f}\n"
            )
            for name, output, target in val_results[i]:
                f.write(f"{name[0]:<30}\t{output:.3f}\t{target:.3f}\n")

    # Best result
    df = pd.DataFrame(best_result, columns=["name", "output", "target"])
    df.to_csv(
        os.path.join(save_dir, f"best_result_loss={best_loss:.4f}.csv"), index=False
    )

    return
