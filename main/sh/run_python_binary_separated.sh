#!/bin/bash

# --- 実行するPythonスクリプトの基本コマンド ---
PYTHON_COMMAND="/bin/python3 /home/gomez16/Haruto/Pytorch_env/DeepLearning2D/main_binary_separated.py"

# --- 1. ResNet18 MSE の学習 ---
echo "--- 1. ResNet18 MSE の学習を開始します ---"

# 現在時刻を 'YYYYMMDD_HHMMSS' 形式で取得し、変数に格納
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
echo "タイムスタンプ (ResNet18): $CURRENT_TIME"

# タイムスタンプを引数に追加してコマンドを実行
# main.py側で --timestamp という引数を受け取れるようにしておく必要があります



$PYTHON_COMMAND --testname Masuda --timestamp "$CURRENT_TIME" --model SmallResNet --criterion BCEWithLogitsLoss --optimizer AdamW --lr_min 0.0004 --lr_max 0.004 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64

$PYTHON_COMMAND --testname Masuda --timestamp "$CURRENT_TIME" --model ResNet18 --criterion BCEWithLogitsLoss --optimizer AdamW --lr_min 0.0004 --lr_max 0.004 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64

$PYTHON_COMMAND --testname Masuda --timestamp "$CURRENT_TIME" --model SmallResNet --criterion BCEWithLogitsLoss --optimizer AdamW --lr_min 0.004 --lr_max 0.04 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64

$PYTHON_COMMAND --testname Masuda --timestamp "$CURRENT_TIME" --model ResNet18 --criterion BCEWithLogitsLoss --optimizer AdamW --lr_min 0.004 --lr_max 0.04 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64


$PYTHON_COMMAND --testname Soma --timestamp "$CURRENT_TIME" --model SmallResNet --criterion BCEWithLogitsLoss --optimizer AdamW --lr_min 0.0004 --lr_max 0.004 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64

$PYTHON_COMMAND --testname Soma --timestamp "$CURRENT_TIME" --model ResNet18 --criterion BCEWithLogitsLoss --optimizer AdamW --lr_min 0.0004 --lr_max 0.004 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64

$PYTHON_COMMAND --testname Soma --timestamp "$CURRENT_TIME" --model SmallResNet --criterion BCEWithLogitsLoss --optimizer AdamW --lr_min 0.004 --lr_max 0.04 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64

$PYTHON_COMMAND --testname Soma --timestamp "$CURRENT_TIME" --model ResNet18 --criterion BCEWithLogitsLoss --optimizer AdamW --lr_min 0.004 --lr_max 0.04 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64



$PYTHON_COMMAND --testname Takase --timestamp "$CURRENT_TIME" --model SmallResNet --criterion BCEWithLogitsLoss --optimizer AdamW --lr_min 0.0004 --lr_max 0.004 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64

$PYTHON_COMMAND --testname Takase --timestamp "$CURRENT_TIME" --model ResNet18 --criterion BCEWithLogitsLoss --optimizer AdamW --lr_min 0.0004 --lr_max 0.004 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64

$PYTHON_COMMAND --testname Takase --timestamp "$CURRENT_TIME" --model SmallResNet --criterion BCEWithLogitsLoss --optimizer AdamW --lr_min 0.004 --lr_max 0.04 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64

$PYTHON_COMMAND --testname Takase --timestamp "$CURRENT_TIME" --model ResNet18 --criterion BCEWithLogitsLoss --optimizer AdamW --lr_min 0.004 --lr_max 0.04 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64


$PYTHON_COMMAND --testname Teragiwa --timestamp "$CURRENT_TIME" --model SmallResNet --criterion BCEWithLogitsLoss --optimizer AdamW --lr_min 0.0004 --lr_max 0.004 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64

$PYTHON_COMMAND --testname Teragiwa --timestamp "$CURRENT_TIME" --model ResNet18 --criterion BCEWithLogitsLoss --optimizer AdamW --lr_min 0.0004 --lr_max 0.004 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64

$PYTHON_COMMAND --testname Teragiwa --timestamp "$CURRENT_TIME" --model SmallResNet --criterion BCEWithLogitsLoss --optimizer AdamW --lr_min 0.004 --lr_max 0.04 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64

$PYTHON_COMMAND --testname Teragiwa --timestamp "$CURRENT_TIME" --model ResNet18 --criterion BCEWithLogitsLoss --optimizer AdamW --lr_min 0.004 --lr_max 0.04 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64


$PYTHON_COMMAND --testname Yatsuda --timestamp "$CURRENT_TIME" --model SmallResNet --criterion BCEWithLogitsLoss --optimizer AdamW --lr_min 0.0004 --lr_max 0.004 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64

$PYTHON_COMMAND --testname Yatsuda --timestamp "$CURRENT_TIME" --model ResNet18 --criterion BCEWithLogitsLoss --optimizer AdamW --lr_min 0.0004 --lr_max 0.004 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64

$PYTHON_COMMAND --testname Yatsuda --timestamp "$CURRENT_TIME" --model SmallResNet --criterion BCEWithLogitsLoss --optimizer AdamW --lr_min 0.004 --lr_max 0.04 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64

$PYTHON_COMMAND --testname Yatsuda --timestamp "$CURRENT_TIME" --model ResNet18 --criterion BCEWithLogitsLoss --optimizer AdamW --lr_min 0.004 --lr_max 0.04 --lr_num 20 --weight_decay 0.001 --epochs 100 --batchsize 64