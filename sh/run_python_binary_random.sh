#!/bin/bash

# --- 実行するPythonスクリプトの基本コマンド ---
PYTHON_COMMAND="/bin/python3 /home/gomez16/Haruto/Pytorch_env/DeepLearning2D/main_binary_random.py"

# --- Kinugasa ---
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
$PYTHON_COMMAND --testname Kinugasa --timestamp "$CURRENT_TIME" --model ResNet18 --criterion BCEWithLogitsLoss --optimizer AdamW --lr_min 0.0004 --lr_max 0.004 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
$PYTHON_COMMAND --testname Kinugasa --timestamp "$CURRENT_TIME" --model SmallResNet --criterion BCEWithLogitsLoss --optimizer AdamW --lr_min 0.0004 --lr_max 0.004 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64

CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
$PYTHON_COMMAND --testname Kinugasa --timestamp "$CURRENT_TIME" --model ResNet18 --criterion BCEWithLogitsLoss --optimizer AdamW --lr_min 0.004 --lr_max 0.04 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
$PYTHON_COMMAND --testname Kinugasa --timestamp "$CURRENT_TIME" --model SmallResNet --criterion BCEWithLogitsLoss --optimizer AdamW --lr_min 0.004 --lr_max 0.04 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64

# --- Masuda ---
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
$PYTHON_COMMAND --testname Masuda --timestamp "$CURRENT_TIME" --model ResNet18 --criterion BCEWithLogitsLoss --optimizer AdamW --lr_min 0.0004 --lr_max 0.004 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
$PYTHON_COMMAND --testname Masuda --timestamp "$CURRENT_TIME" --model SmallResNet --criterion BCEWithLogitsLoss --optimizer AdamW --lr_min 0.0004 --lr_max 0.004 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64

CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
$PYTHON_COMMAND --testname Masuda --timestamp "$CURRENT_TIME" --model ResNet18 --criterion BCEWithLogitsLoss --optimizer AdamW --lr_min 0.004 --lr_max 0.04 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
$PYTHON_COMMAND --testname Masuda --timestamp "$CURRENT_TIME" --model SmallResNet --criterion BCEWithLogitsLoss --optimizer AdamW --lr_min 0.004 --lr_max 0.04 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64

# --- Soma ---
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
$PYTHON_COMMAND --testname Soma --timestamp "$CURRENT_TIME" --model ResNet18 --criterion BCEWithLogitsLoss --optimizer AdamW --lr_min 0.0004 --lr_max 0.004 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
$PYTHON_COMMAND --testname Soma --timestamp "$CURRENT_TIME" --model SmallResNet --criterion BCEWithLogitsLoss --optimizer AdamW --lr_min 0.0004 --lr_max 0.004 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64

CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
$PYTHON_COMMAND --testname Soma --timestamp "$CURRENT_TIME" --model ResNet18 --criterion BCEWithLogitsLoss --optimizer AdamW --lr_min 0.004 --lr_max 0.04 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
$PYTHON_COMMAND --testname Soma --timestamp "$CURRENT_TIME" --model SmallResNet --criterion BCEWithLogitsLoss --optimizer AdamW --lr_min 0.004 --lr_max 0.04 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64

# --- Takase ---
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
$PYTHON_COMMAND --testname Takase --timestamp "$CURRENT_TIME" --model ResNet18 --criterion BCEWithLogitsLoss --optimizer AdamW --lr_min 0.0004 --lr_max 0.004 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
$PYTHON_COMMAND --testname Takase --timestamp "$CURRENT_TIME" --model SmallResNet --criterion BCEWithLogitsLoss --optimizer AdamW --lr_min 0.0004 --lr_max 0.004 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64

CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
$PYTHON_COMMAND --testname Takase --timestamp "$CURRENT_TIME" --model ResNet18 --criterion BCEWithLogitsLoss --optimizer AdamW --lr_min 0.004 --lr_max 0.04 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
$PYTHON_COMMAND --testname Takase --timestamp "$CURRENT_TIME" --model SmallResNet --criterion BCEWithLogitsLoss --optimizer AdamW --lr_min 0.004 --lr_max 0.04 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64

# --- Teragiwa ---
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
$PYTHON_COMMAND --testname Teragiwa --timestamp "$CURRENT_TIME" --model ResNet18 --criterion BCEWithLogitsLoss --optimizer AdamW --lr_min 0.0004 --lr_max 0.004 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
$PYTHON_COMMAND --testname Teragiwa --timestamp "$CURRENT_TIME" --model SmallResNet --criterion BCEWithLogitsLoss --optimizer AdamW --lr_min 0.0004 --lr_max 0.004 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64

CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
$PYTHON_COMMAND --testname Teragiwa --timestamp "$CURRENT_TIME" --model ResNet18 --criterion BCEWithLogitsLoss --optimizer AdamW --lr_min 0.004 --lr_max 0.04 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
$PYTHON_COMMAND --testname Teragiwa --timestamp "$CURRENT_TIME" --model SmallResNet --criterion BCEWithLogitsLoss --optimizer AdamW --lr_min 0.004 --lr_max 0.04 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64

# --- Yatsuda ---
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
$PYTHON_COMMAND --testname Yatsuda --timestamp "$CURRENT_TIME" --model ResNet18 --criterion BCEWithLogitsLoss --optimizer AdamW --lr_min 0.0004 --lr_max 0.004 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
$PYTHON_COMMAND --testname Yatsuda --timestamp "$CURRENT_TIME" --model SmallResNet --criterion BCEWithLogitsLoss --optimizer AdamW --lr_min 0.0004 --lr_max 0.004 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64

CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
$PYTHON_COMMAND --testname Yatsuda --timestamp "$CURRENT_TIME" --model ResNet18 --criterion BCEWithLogitsLoss --optimizer AdamW --lr_min 0.004 --lr_max 0.04 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
$PYTHON_COMMAND --testname Yatsuda --timestamp "$CURRENT_TIME" --model SmallResNet --criterion BCEWithLogitsLoss --optimizer AdamW --lr_min 0.004 --lr_max 0.04 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64