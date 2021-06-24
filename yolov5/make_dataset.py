import numpy as np
import pandas as pd
from glob import glob
import shutil, os
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from tqdm import tqdm
import random

# import seaborn as sns


# Make dataset for yolo

# fold = 0

USER = "minki"
DATA_DIR = f"/data/{USER}/kaggle/siim-covid19"
SEED = 92819

IMAGE_SIZE = 512

random.seed(SEED)
np.random.seed(SEED)


### Make yolov5 labels in txt format
df = pd.read_csv(DATA_DIR + "/orig/train_image_level.csv")

df1 = pd.read_csv(DATA_DIR + "/train_meta.csv")
df1 = df1[df1["split"] == "train"].reset_index(drop=True)
for i in range(df1.shape[0]):
    df1.loc[i, "image_id"] = df1.loc[i, "image_id"] + "_image"

df1.columns = ["id", "dim0", "dim1", "split"]
df = pd.merge(df, df1, on="id", how="left")

for i in range(df.shape[0]):
    # FIXME:
    a = df.loc[i, "id"]
    f = open(DATA_DIR + f"/image_{IMAGE_SIZE}/train/{a}.txt", "w")
    b = df.loc[i, "label"].split()
    b_len = int(len(b) / 6)
    if b[0] == "none":
        f.close()
        continue
    dim1 = df.loc[i, "dim1"]
    dim0 = df.loc[i, "dim0"]
    for j in range(b_len):
        x_mid = str((float(b[6 * j + 2]) + float(b[6 * j + 4])) / 2 / dim1)
        y_mid = str((float(b[6 * j + 3]) + float(b[6 * j + 5])) / 2 / dim0)
        w = str((float(b[6 * j + 4]) - float(b[6 * j + 2])) / dim1)
        h = str((float(b[6 * j + 5]) - float(b[6 * j + 3])) / dim0)
        f.write("0" + " " + x_mid + " " + y_mid + " " + w + " " + h + " ")
        f.write("\n")

    f.close()
###

train_df = pd.read_csv(DATA_DIR + "/orig/train_image_level.csv")

# Remove more than 2 ids
more_than_2_ids = []

for i in range(len(train_df)):
    row = train_df.iloc[i]
    sid = row["StudyInstanceUID"]
    sid_df = train_df[train_df["StudyInstanceUID"] == sid]
    if len(sid_df) >= 2:
        more_than_2_ids.append(sid)

train_df = train_df[~train_df["StudyInstanceUID"].isin(more_than_2_ids)]
train_df.reset_index(inplace=True)

#         break

df = train_df

gkf = GroupKFold(n_splits=7)
df["fold"] = -1
for fold, (train_idx, val_idx) in enumerate(
    gkf.split(df, groups=df.StudyInstanceUID.tolist())
):
    df.loc[val_idx, "fold"] = fold

train_df = df


def createImagesTxt(_images, filepath, data_dir):
    images_dir = data_dir + f"/image_{IMAGE_SIZE}/train/"
    rows = []
    for img_id in _images:
        rows.append(images_dir + img_id + ".png")

    f = open(filepath, "w")
    f.write("\n".join(rows))
    f.close()


os.makedirs(DATA_DIR + "/yolov5", exist_ok=True)

for fold in range(7):
    train_files = []
    val_files = []
    val_files += list(train_df[train_df.fold == fold].id.unique())
    train_files += list(train_df[train_df.fold != fold].id.unique())
    print(len(train_files), len(val_files))

    train_path = DATA_DIR + f"/yolov5/train{fold}_{IMAGE_SIZE}.txt"
    val_path = DATA_DIR + f"/yolov5/val{fold}_{IMAGE_SIZE}.txt"
    createImagesTxt(train_files, train_path, DATA_DIR)
    createImagesTxt(val_files, val_path, DATA_DIR)
