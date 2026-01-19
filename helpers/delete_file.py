"""
モデルファイルが容量を食っていたので削除するヘルパー関数
"""

import os

TESTNAME = "Yatsuda"


def delete_file(root_dir):
    after_size = 0
    count = 0
    for root, _, files in os.walk(root_dir):
        for f in files:
            if not f.endswith(".pth"):
                continue

            pt_path = os.path.join(root, f)

            pt_size = os.path.getsize(pt_path)
            after_size += pt_size

            os.remove(pt_path)
            count += 1

            # print(f"removed: {pt_path}")

    def mb(x):
        return x / 1024 / 1024 / 1024

    print("==== result ====")
    print(f"files        : {count}")
    print(f"delete       : {mb(after_size):.2f} GB")


# 使用例
delete_file(f"../../../../../mnt/share/Takase/Results")
