#!/bin/bash

# --- 実行するPythonスクリプトの基本コマンド ---
PYTHON_COMMAND="/bin/python3 /home/gomez16/Haruto/Pytorch_env/MEP-Predict-2D-modifiable-muscle/scripts/test/main_binary_test.py"


$PYTHON_COMMAND --testname Aoki --muscle 1 --mode separated --foldername 20260119_170546_Binary_Standardized_SmallResNet_BCEWithLogitsLoss_LRe-4~3