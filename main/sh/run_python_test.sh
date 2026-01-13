#!/bin/bash

# --- 実行するPythonスクリプトの基本コマンド ---
PYTHON_COMMAND="/bin/python3 /home/gomez16/Haruto/Pytorch_env/DeepLearning2D/main_test.py"

$PYTHON_COMMAND --testname Aoki --foldername 20251110_180931_Standardized_ResNet18_MSE_LRe-4~3
$PYTHON_COMMAND --testname Aoki --foldername 20251110_183707_Standardized_ResNet18_MAE_LRe-4~3
$PYTHON_COMMAND --testname Aoki --foldername 20251110_190315_Standardized_SmallResNet_MSE_LRe-4~3
$PYTHON_COMMAND --testname Aoki --foldername 20251110_194811_Standardized_SmallResNet_MAE_LRe-4~3
$PYTHON_COMMAND --testname Aoki --foldername 20251110_202953_Standardized_ResNet18_MSE_LRe-3~2
$PYTHON_COMMAND --testname Aoki --foldername 20251110_205625_Standardized_ResNet18_MAE_LRe-3~2
$PYTHON_COMMAND --testname Aoki --foldername 20251110_212324_Standardized_SmallResNet_MSE_LRe-3~2
$PYTHON_COMMAND --testname Aoki --foldername 20251110_220953_Standardized_SmallResNet_MAE_LRe-3~2
$PYTHON_COMMAND --testname Kinugasa --foldername 20251110_225511_Standardized_ResNet18_MSE_LRe-4~3
$PYTHON_COMMAND --testname Kinugasa --foldername 20251110_232442_Standardized_ResNet18_MAE_LRe-4~3
$PYTHON_COMMAND --testname Kinugasa --foldername 20251110_235253_Standardized_SmallResNet_MSE_LRe-4~3
$PYTHON_COMMAND --testname Kinugasa --foldername 20251111_003732_Standardized_SmallResNet_MAE_LRe-4~3
$PYTHON_COMMAND --testname Kinugasa --foldername 20251111_012256_Standardized_ResNet18_MSE_LRe-3~2
$PYTHON_COMMAND --testname Kinugasa --foldername 20251111_014826_Standardized_ResNet18_MAE_LRe-3~2
$PYTHON_COMMAND --testname Kinugasa --foldername 20251111_021430_Standardized_SmallResNet_MSE_LRe-3~2
$PYTHON_COMMAND --testname Kinugasa --foldername 20251111_025851_Standardized_SmallResNet_MAE_LRe-3~2

echo "--- 全ての実行が完了しました ---"