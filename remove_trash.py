import os
import shutil

src_dir = "generated/gpt/depth16_worddim1024_bedroom_vocab1024_wo_lat_wo_textc"
dst_dir = "/mnt/disk-1/zhx24/code/ARScene/visualizations/depth16_worddim1024_bedroom_vocab1024_wo_lat_wo_textc"
count = 0
print(len(os.listdir(src_dir)))