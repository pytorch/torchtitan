#!/usr/bin/env python3

# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import argparse
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument("--input-images-dir", type=str, required=True)
parser.add_argument("--input-captions-file", type=str, required=True)
parser.add_argument("--output-images-dir", type=str, required=True)
parser.add_argument("--output-tsv-file", type=str, required=True)
parser.add_argument("--num-samples", type=int, default=30000)
parser.add_argument("--seed", type=int, default=2023)
parser.add_argument("--width", type=int, default=256)
parser.add_argument("--height", type=int, default=256)
parser.add_argument("--num-workers", type=int, default=4)
parser.add_argument("--allow-duplicate-images", type=bool, default=False)

args = parser.parse_args()


def resize_image(input_image, output_image, width, height, resample=Image.Resampling.BICUBIC):
    image = Image.open(input_image)
    image = image.resize((width, height), resample=resample)
    image.save(output_image)


# Load coco annotations
with open(args.input_captions_file, "r") as f:
    captions = json.load(f)
    annotations = captions["annotations"]

# Convert to dataframe
df = pd.DataFrame(annotations)
df['caption'] = df['caption'].apply(lambda x: x.replace('\n', '').strip())

# Shuffle the dataframe
df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

# Keep a single captions per image
if not args.allow_duplicate_images:
    df = df.drop_duplicates(subset=["image_id"], keep="first")

# Take a subset
df = df[:args.num_samples]

# Sort by id
df = df.sort_values(by=["id"])

# Save the subset to a tsv file
df.to_csv(args.output_tsv_file, sep="\t", index=False)

# Create output image directory if it doesn't exist
os.makedirs(os.path.join(args.output_images_dir, "validation"), exist_ok=True)

# Save a metadata file for conversion to hf dataset
df['file_name'] = df['image_id'].apply(lambda x: f"COCO_val2014_{x:012}.png")
df[['file_name', 'caption', 'image_id']].to_csv(os.path.join(args.output_images_dir, "validation", "metadata.csv"), index=False)

# resize images with a worker pool
with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
    for i, row in df.iterrows():
        image_fname = f"COCO_val2014_{row['image_id']:012}.jpg"
        image_fname_out = f"COCO_val2014_{row['image_id']:012}.png"
        input_img = os.path.join(args.input_images_dir, image_fname)
        output_img = os.path.join(args.output_images_dir, "validation", image_fname_out)

        executor.submit(resize_image, input_img, output_img, args.width, args.height, Image.Resampling.BICUBIC)
