# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import pandas as pd
import tarfile
import io
import json
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--tsv_file", type=str, required=True)
parser.add_argument("--image_dir", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--samples_per_shard", type=int, default=1000)
args = parser.parse_args()


# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# Load the TSV file
df = pd.read_csv(args.tsv_file, sep='\t')[["image_id", "caption"]]

# Group data by shard
num_shards = math.ceil(len(df) / args.samples_per_shard)
for shard_idx in tqdm(range(num_shards), desc="Creating shards"):
    # Create a new tar file for this shard
    shard_name = f"{args.output_dir}/shard_{shard_idx:05d}.tar"
    
    start_idx = shard_idx * args.samples_per_shard
    end_idx = min((shard_idx + 1) * args.samples_per_shard, len(df))
    
    with tarfile.open(shard_name, "w") as tar:
        for idx in range(start_idx, end_idx):
            row = df.iloc[idx]
            image_id = row['image_id']
            caption = row['caption']
            
            # Define the base filename using image_id
            base_name = f"{image_id:012d}"  # Format as 12-digit number
            
            # Path to the image file
            img_path = os.path.join(args.image_dir, "validation", f"COCO_val2014_{base_name}.png")
            
            # Add image to tar
            img_info = tarfile.TarInfo(f"{base_name}.png")
            img_data = open(img_path, "rb").read()
            img_info.size = len(img_data)
            tar.addfile(img_info, io.BytesIO(img_data))
            
            # Create and add txt file with the caption
            txt_info = tarfile.TarInfo(f"{base_name}.txt")
            txt_data = caption
            txt_info.size = len(txt_data.encode('utf-8'))
            tar.addfile(txt_info, io.BytesIO(txt_data.encode('utf-8')))
            
            # Create and add JSON metadata file
            metadata = {
                "id": str(image_id),
                "caption": caption,
                "filename": f"COCO_val2014_{base_name}.png"
            }
            json_info = tarfile.TarInfo(f"{base_name}.json")
            json_data = json.dumps(metadata, indent=2)
            json_info.size = len(json_data.encode('utf-8'))
            tar.addfile(json_info, io.BytesIO(json_data.encode('utf-8')))

print(f"WebDataset created with {num_shards} shards in {args.output_dir}")
