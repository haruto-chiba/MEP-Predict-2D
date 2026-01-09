#!/bin/bash

# --- 実行するPythonスクリプトの基本コマンド ---
PYTHON_COMMAND="/bin/python3 /home/gomez16/Haruto/Pytorch_env/DeepLearning2D/main_random.py"

# --- Aoki ---
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
$PYTHON_COMMAND --testname Aoki --timestamp "$CURRENT_TIME" --model ResNet18 --criterion MSE --optimizer AdamW --lr_min 0.0004 --lr_max 0.004 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
$PYTHON_COMMAND --testname Aoki --timestamp "$CURRENT_TIME" --model ResNet18 --criterion MAE --optimizer AdamW --lr_min 0.0004 --lr_max 0.004 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64

CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
$PYTHON_COMMAND --testname Aoki --timestamp "$CURRENT_TIME" --model SmallResNet --criterion MSE --optimizer AdamW --lr_min 0.0004 --lr_max 0.004 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
$PYTHON_COMMAND --testname Aoki --timestamp "$CURRENT_TIME" --model SmallResNet --criterion MAE --optimizer AdamW --lr_min 0.0004 --lr_max 0.004 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64

CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
$PYTHON_COMMAND --testname Aoki --timestamp "$CURRENT_TIME" --model ResNet18 --criterion MSE --optimizer AdamW --lr_min 0.004 --lr_max 0.04 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
$PYTHON_COMMAND --testname Aoki --timestamp "$CURRENT_TIME" --model ResNet18 --criterion MAE --optimizer AdamW --lr_min 0.0004 --lr_max 0.04 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64

CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
$PYTHON_COMMAND --testname Aoki --timestamp "$CURRENT_TIME" --model SmallResNet --criterion MSE --optimizer AdamW --lr_min 0.004 --lr_max 0.04 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
$PYTHON_COMMAND --testname Aoki --timestamp "$CURRENT_TIME" --model SmallResNet --criterion MAE --optimizer AdamW --lr_min 0.004 --lr_max 0.04 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64
# ------------

# --- Kinugasa ---
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
$PYTHON_COMMAND --testname Kinugasa --timestamp "$CURRENT_TIME" --model ResNet18 --criterion MSE --optimizer AdamW --lr_min 0.0004 --lr_max 0.004 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
$PYTHON_COMMAND --testname Kinugasa --timestamp "$CURRENT_TIME" --model ResNet18 --criterion MAE --optimizer AdamW --lr_min 0.0004 --lr_max 0.004 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64

CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
$PYTHON_COMMAND --testname Kinugasa --timestamp "$CURRENT_TIME" --model SmallResNet --criterion MSE --optimizer AdamW --lr_min 0.0004 --lr_max 0.004 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
$PYTHON_COMMAND --testname Kinugasa --timestamp "$CURRENT_TIME" --model SmallResNet --criterion MAE --optimizer AdamW --lr_min 0.0004 --lr_max 0.004 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64

CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
$PYTHON_COMMAND --testname Kinugasa --timestamp "$CURRENT_TIME" --model ResNet18 --criterion MSE --optimizer AdamW --lr_min 0.004 --lr_max 0.04 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
$PYTHON_COMMAND --testname Kinugasa --timestamp "$CURRENT_TIME" --model ResNet18 --criterion MAE --optimizer AdamW --lr_min 0.0004 --lr_max 0.04 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64

CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
$PYTHON_COMMAND --testname Kinugasa --timestamp "$CURRENT_TIME" --model SmallResNet --criterion MSE --optimizer AdamW --lr_min 0.004 --lr_max 0.04 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
$PYTHON_COMMAND --testname Kinugasa --timestamp "$CURRENT_TIME" --model SmallResNet --criterion MAE --optimizer AdamW --lr_min 0.004 --lr_max 0.04 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64
# ----------------

# --- Masuda ---
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
$PYTHON_COMMAND --testname Masuda --timestamp "$CURRENT_TIME" --model ResNet18 --criterion MSE --optimizer AdamW --lr_min 0.0004 --lr_max 0.004 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
$PYTHON_COMMAND --testname Masuda --timestamp "$CURRENT_TIME" --model ResNet18 --criterion MAE --optimizer AdamW --lr_min 0.0004 --lr_max 0.004 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64

CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
$PYTHON_COMMAND --testname Masuda --timestamp "$CURRENT_TIME" --model SmallResNet --criterion MSE --optimizer AdamW --lr_min 0.0004 --lr_max 0.004 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
$PYTHON_COMMAND --testname Masuda --timestamp "$CURRENT_TIME" --model SmallResNet --criterion MAE --optimizer AdamW --lr_min 0.0004 --lr_max 0.004 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64

CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
$PYTHON_COMMAND --testname Masuda --timestamp "$CURRENT_TIME" --model ResNet18 --criterion MSE --optimizer AdamW --lr_min 0.004 --lr_max 0.04 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
$PYTHON_COMMAND --testname Masuda --timestamp "$CURRENT_TIME" --model ResNet18 --criterion MAE --optimizer AdamW --lr_min 0.0004 --lr_max 0.04 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64

CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
$PYTHON_COMMAND --testname Masuda --timestamp "$CURRENT_TIME" --model SmallResNet --criterion MSE --optimizer AdamW --lr_min 0.004 --lr_max 0.04 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
$PYTHON_COMMAND --testname Masuda --timestamp "$CURRENT_TIME" --model SmallResNet --criterion MAE --optimizer AdamW --lr_min 0.004 --lr_max 0.04 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64
# -------------

# --- Soma ---
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
$PYTHON_COMMAND --testname Soma --timestamp "$CURRENT_TIME" --model ResNet18 --criterion MSE --optimizer AdamW --lr_min 0.0004 --lr_max 0.004 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
$PYTHON_COMMAND --testname Soma --timestamp "$CURRENT_TIME" --model ResNet18 --criterion MAE --optimizer AdamW --lr_min 0.0004 --lr_max 0.004 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64

CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
$PYTHON_COMMAND --testname Soma --timestamp "$CURRENT_TIME" --model SmallResNet --criterion MSE --optimizer AdamW --lr_min 0.0004 --lr_max 0.004 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
$PYTHON_COMMAND --testname Soma --timestamp "$CURRENT_TIME" --model SmallResNet --criterion MAE --optimizer AdamW --lr_min 0.0004 --lr_max 0.004 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64

CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
$PYTHON_COMMAND --testname Soma --timestamp "$CURRENT_TIME" --model ResNet18 --criterion MSE --optimizer AdamW --lr_min 0.004 --lr_max 0.04 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
$PYTHON_COMMAND --testname Soma --timestamp "$CURRENT_TIME" --model ResNet18 --criterion MAE --optimizer AdamW --lr_min 0.0004 --lr_max 0.04 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64

CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
$PYTHON_COMMAND --testname Soma --timestamp "$CURRENT_TIME" --model SmallResNet --criterion MSE --optimizer AdamW --lr_min 0.004 --lr_max 0.04 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
$PYTHON_COMMAND --testname Soma --timestamp "$CURRENT_TIME" --model SmallResNet --criterion MAE --optimizer AdamW --lr_min 0.004 --lr_max 0.04 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64
# ------------

# --- Yatsuda ---
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
$PYTHON_COMMAND --testname Yatsuda --timestamp "$CURRENT_TIME" --model ResNet18 --criterion MSE --optimizer AdamW --lr_min 0.0004 --lr_max 0.004 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
$PYTHON_COMMAND --testname Yatsuda --timestamp "$CURRENT_TIME" --model ResNet18 --criterion MAE --optimizer AdamW --lr_min 0.0004 --lr_max 0.004 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64

CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
$PYTHON_COMMAND --testname Yatsuda --timestamp "$CURRENT_TIME" --model SmallResNet --criterion MSE --optimizer AdamW --lr_min 0.0004 --lr_max 0.004 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
$PYTHON_COMMAND --testname Yatsuda --timestamp "$CURRENT_TIME" --model SmallResNet --criterion MAE --optimizer AdamW --lr_min 0.0004 --lr_max 0.004 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64

CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
$PYTHON_COMMAND --testname Yatsuda --timestamp "$CURRENT_TIME" --model ResNet18 --criterion MSE --optimizer AdamW --lr_min 0.004 --lr_max 0.04 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
$PYTHON_COMMAND --testname Yatsuda --timestamp "$CURRENT_TIME" --model ResNet18 --criterion MAE --optimizer AdamW --lr_min 0.0004 --lr_max 0.04 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64

CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
$PYTHON_COMMAND --testname Yatsuda --timestamp "$CURRENT_TIME" --model SmallResNet --criterion MSE --optimizer AdamW --lr_min 0.004 --lr_max 0.04 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
$PYTHON_COMMAND --testname Yatsuda --timestamp "$CURRENT_TIME" --model SmallResNet --criterion MAE --optimizer AdamW --lr_min 0.004 --lr_max 0.04 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64
# ---------------
echo "--- 全ての実行が完了しました ---"