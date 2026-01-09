#!/bin/bash

# --- 実行するPythonスクリプトの基本コマンド ---
PYTHON_COMMAND="/bin/python3 /home/gomez16/Haruto/Pytorch_env/DeepLearning2D/main_binary_test.py"


$PYTHON_COMMAND --testname Kinugasa --foldername 20251209_181858_Binary_Random_Standardized_ResNet18_BCEWithLogitsLoss_LRe-4~3
$PYTHON_COMMAND --testname Kinugasa --foldername 20251209_182535_Binary_Random_Standardized_SmallResNet_BCEWithLogitsLoss_LRe-4~3
$PYTHON_COMMAND --testname Kinugasa --foldername 20251209_183638_Binary_Random_Standardized_ResNet18_BCEWithLogitsLoss_LRe-3~2
$PYTHON_COMMAND --testname Kinugasa --foldername 20251209_184457_Binary_Random_Standardized_SmallResNet_BCEWithLogitsLoss_LRe-3~2

$PYTHON_COMMAND --testname Masuda --foldername 20251209_185628_Binary_Random_Standardized_ResNet18_BCEWithLogitsLoss_LRe-4~3
$PYTHON_COMMAND --testname Masuda --foldername 20251209_190204_Binary_Random_Standardized_SmallResNet_BCEWithLogitsLoss_LRe-4~3
$PYTHON_COMMAND --testname Masuda --foldername 20251209_191154_Binary_Random_Standardized_ResNet18_BCEWithLogitsLoss_LRe-3~2
$PYTHON_COMMAND --testname Masuda --foldername 20251209_191758_Binary_Random_Standardized_SmallResNet_BCEWithLogitsLoss_LRe-3~2

$PYTHON_COMMAND --testname Soma --foldername 20251209_192739_Binary_Random_Standardized_ResNet18_BCEWithLogitsLoss_LRe-4~3
$PYTHON_COMMAND --testname Soma --foldername 20251209_193409_Binary_Random_Standardized_SmallResNet_BCEWithLogitsLoss_LRe-4~3
$PYTHON_COMMAND --testname Soma --foldername 20251209_194304_Binary_Random_Standardized_ResNet18_BCEWithLogitsLoss_LRe-3~2
$PYTHON_COMMAND --testname Soma --foldername 20251209_194947_Binary_Random_Standardized_SmallResNet_BCEWithLogitsLoss_LRe-3~2

$PYTHON_COMMAND --testname Takase --foldername 20251209_200015_Binary_Random_Standardized_ResNet18_BCEWithLogitsLoss_LRe-4~3
$PYTHON_COMMAND --testname Takase --foldername 20251209_200640_Binary_Random_Standardized_SmallResNet_BCEWithLogitsLoss_LRe-4~3
$PYTHON_COMMAND --testname Takase --foldername 20251209_201637_Binary_Random_Standardized_ResNet18_BCEWithLogitsLoss_LRe-3~2
$PYTHON_COMMAND --testname Takase --foldername 20251209_202350_Binary_Random_Standardized_SmallResNet_BCEWithLogitsLoss_LRe-3~2

$PYTHON_COMMAND --testname Teragiwa --foldername 20251209_203521_Binary_Random_Standardized_ResNet18_BCEWithLogitsLoss_LRe-4~3
$PYTHON_COMMAND --testname Teragiwa --foldername 20251209_204204_Binary_Random_Standardized_SmallResNet_BCEWithLogitsLoss_LRe-4~3
$PYTHON_COMMAND --testname Teragiwa --foldername 20251209_205138_Binary_Random_Standardized_ResNet18_BCEWithLogitsLoss_LRe-3~2
$PYTHON_COMMAND --testname Teragiwa --foldername 20251209_205951_Binary_Random_Standardized_SmallResNet_BCEWithLogitsLoss_LRe-3~2

$PYTHON_COMMAND --testname Yatsuda --foldername 20251209_211138_Binary_Random_Standardized_ResNet18_BCEWithLogitsLoss_LRe-4~3
$PYTHON_COMMAND --testname Yatsuda --foldername 20251209_211805_Binary_Random_Standardized_SmallResNet_BCEWithLogitsLoss_LRe-4~3
$PYTHON_COMMAND --testname Yatsuda --foldername 20251209_212856_Binary_Random_Standardized_ResNet18_BCEWithLogitsLoss_LRe-3~2
$PYTHON_COMMAND --testname Yatsuda --foldername 20251209_213705_Binary_Random_Standardized_SmallResNet_BCEWithLogitsLoss_LRe-3~2


echo "--- 全ての実行が完了しました ---"