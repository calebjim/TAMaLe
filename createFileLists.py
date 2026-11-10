import os
import pandas as pd
import numpy as np
from obspy import read

# Path to .mseed files (using CS for now because CT has three sensors)
dir_path1 = "dischma_data/season_2020_21/CS"
dir_path2 = "dischma_data/season_2021_22/CS"

# Create list of .mseed files
file_list1 = os.listdir(dir_path1)
file_list2 = os.listdir(dir_path2)

# Load labels.csv file
csv_labels = pd.read_csv("dischma_data/labels.csv")

# Creates one long list of all file pathes that are accounted for in labels.csv
file_path = []
for i in range(len(file_list1)):
    file_path.append(dir_path1 + "/" + file_list1[i])
    if (file_list1[i][12:25]) != (csv_labels['Event_id'][i]):
        print(f"Error: File {i} in {dir_path1} does not match Event_id in labels.csv")
        break
for i in range(len(file_list2)):
    file_path.append(dir_path2 + "/" + file_list2[i])
    if (file_list2[i][12:25]) != (csv_labels['Event_id'][len(file_list1)+i]):
        print(f"Error: File {i} in {dir_path2} does not match Event_id in labels.csv")
        break

# Splits up all files by label in labels.csv
noise_files = []
eq_files = []
ava_files = []
for i in range(len(file_path)):
    if csv_labels['label'][i] == 0:
        noise_files.append(file_path[i])
    elif csv_labels['label'][i] == 2:
        eq_files.append(file_path[i])
    elif csv_labels['label'][i] == 1:
        ava_files.append(file_path[i])

# Write pathes to all files into 3 separate .txt files
with open("noise_files.txt", "w") as noise_file:
    for file in noise_files:
        noise_file.write(file + "\n")
print(f'Wrote {len(noise_files)} noise file pathes to "noise_files.txt"')

with open("eq_files.txt", "w") as eq_file:
    for file in eq_files:
        eq_file.write(file + "\n")
print(f'Wrote {len(eq_files)} earthquake file pathes to "eq_files.txt"')

with open("ava_files.txt", "w") as ava_file:
    for file in ava_files:
        ava_file.write(file + "\n")
print(f'Wrote {len(ava_files)} avalanche file pathers to "ava_files.txt"')