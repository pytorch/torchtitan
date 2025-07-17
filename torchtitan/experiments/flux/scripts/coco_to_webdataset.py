#!/usr/bin/env python3

# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import io
import json
import math
import os
import shutil
import tarfile
import tempfile
from concurrent.futures import as_completed, ProcessPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


parser = argparse.ArgumentParser(
    description="Combined script - Process COCO dataset, resize images, and create WebDataset"
)
parser.add_argument(
    "--input-images-dir",
    type=str,
    required=True,
    help="Directory containing input COCO images",
)
parser.add_argument(
    "--input-captions-file", type=str, required=True, help="COCO annotations JSON file"
)
parser.add_argument(
    "--output-dir",
    type=str,
    required=True,
    help="Output directory for WebDataset tar files",
)
parser.add_argument(
    "--output-tsv-file",
    type=str,
    required=True,
    help="Output TSV file with processed subset",
)
parser.add_argument(
    "--num-samples",
    type=int,
    default=29696,
    help="Number of samples to process",
)
parser.add_argument("--seed", type=int, default=2023, help="Random seed for shuffling")
parser.add_argument("--width", type=int, default=256, help="Target image width")
parser.add_argument("--height", type=int, default=256, help="Target image height")
parser.add_argument(
    "--samples-per-shard",
    type=int,
    default=1000,
    help="Number of samples per WebDataset shard",
)
parser.add_argument(
    "--num-workers",
    type=int,
    default=8,
    help="Number of worker processes for image processing",
)
parser.add_argument(
    "--allow-duplicate-images",
    type=bool,
    default=False,
    help="Allow multiple captions per image",
)

args = parser.parse_args()


def resize_image(
    input_image_path: str,
    output_image_path: str,
    width: int,
    height: int,
    resample=Image.Resampling.BICUBIC,
):
    """Resize image and save to disk."""
    try:
        image = Image.open(input_image_path)
        image = image.resize((width, height), resample=resample)
        image.save(output_image_path)
        return True
    except Exception as e:
        print(f"Error processing image {input_image_path}: {e}")
        return False


def main():
    print("=== PHASE 1: Processing COCO annotations and resizing images ===")

    # Load coco annotations
    print("Loading COCO annotations...")
    with open(args.input_captions_file, "r") as f:
        captions = json.load(f)
        annotations = captions["annotations"]

    # Convert to dataframe
    df = pd.DataFrame(annotations)
    df["caption"] = df["caption"].apply(lambda x: x.replace("\n", "").strip())

    print(f"Loaded {len(df)} annotations")

    # Shuffle the dataframe
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # Keep a single caption per image if not allowing duplicates
    if not args.allow_duplicate_images:
        df = df.drop_duplicates(subset=["image_id"], keep="first")
        print(f"After removing duplicates: {len(df)} samples")

    # Take a subset
    df = df[: args.num_samples]
    print(f"Processing {len(df)} samples")

    # Sort by id for consistent ordering
    df = df.sort_values(by=["id"])

    # assign a timestep to each sample
    assert len(df) % 8 == 0, "Number of samples must be divisible by 8"
    df["timestep"] = np.arange(len(df)) % 8

    # Save the subset to a TSV file
    print(f"Saving subset to {args.output_tsv_file}")
    df.to_csv(args.output_tsv_file, sep="\t", index=False)

    # Create a temporary directory for resized images
    temp_dir = tempfile.mkdtemp()
    validation_dir = os.path.join(temp_dir, "validation")
    os.makedirs(validation_dir, exist_ok=True)

    print(f"Using temporary directory: {temp_dir}")

    # Save a metadata file for conversion to hf dataset
    df_metadata = df.copy()
    df_metadata["file_name"] = df_metadata["image_id"].apply(
        lambda x: f"COCO_val2014_{x:012}.png"
    )
    df_metadata[["file_name", "caption", "image_id", "timestep"]].to_csv(
        os.path.join(validation_dir, "metadata.csv"), index=False
    )

    # Resize images using ProcessPoolExecutor for better performance
    print("Resizing images to temporary directory...")
    successful_images = []

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        # Submit all resize tasks
        future_to_row = {}
        for i, row in df.iterrows():
            image_fname = f"COCO_val2014_{row['image_id']:012}.jpg"
            image_fname_out = f"COCO_val2014_{row['image_id']:012}.png"
            input_img = os.path.join(args.input_images_dir, image_fname)
            output_img = os.path.join(validation_dir, image_fname_out)

            future = executor.submit(
                resize_image,
                input_img,
                output_img,
                args.width,
                args.height,
                Image.Resampling.BICUBIC,
            )
            future_to_row[future] = row

        # Collect results
        for future in tqdm(
            as_completed(future_to_row),
            total=len(future_to_row),
            desc="Resizing images",
        ):
            row = future_to_row[future]
            try:
                success = future.result()
                if success:
                    successful_images.append(row)
                else:
                    print(f"Failed to process image {row['image_id']}")
            except Exception as e:
                print(f"Exception processing image {row['image_id']}: {e}")

    print(f"Successfully processed {len(successful_images)} images")

    # Update dataframe to only include successfully processed images
    if len(successful_images) < len(df):
        successful_image_ids = [row["image_id"] for row in successful_images]
        df_webdataset = df[df["image_id"].isin(successful_image_ids)].copy()
    else:
        df_webdataset = df.copy()

    print("=== PHASE 2: Creating WebDataset shards ===")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Extract only the columns needed for WebDataset creation
    df_webdataset = df_webdataset[["image_id", "caption", "timestep"]].copy()

    # Group data by shard
    num_shards = math.ceil(len(df_webdataset) / args.samples_per_shard)
    print(f"Creating {num_shards} shards with {args.samples_per_shard} samples each")

    # Process each shard
    for shard_idx in tqdm(range(num_shards), desc="Creating shards"):
        shard_name = f"{args.output_dir}/shard_{shard_idx:05d}.tar"

        start_idx = shard_idx * args.samples_per_shard
        end_idx = min((shard_idx + 1) * args.samples_per_shard, len(df_webdataset))

        # Create the tar file for this shard
        with tarfile.open(shard_name, "w") as tar:
            for idx in range(start_idx, end_idx):
                row = df_webdataset.iloc[idx]
                image_id = row["image_id"]
                caption = row["caption"]
                timestep = row["timestep"]

                # Define the base filename using image_id
                base_name = f"{image_id:012d}"

                # Path to the image file
                img_path = os.path.join(validation_dir, f"COCO_val2014_{base_name}.png")

                # Check if file exists
                if not os.path.exists(img_path):
                    print(f"Skipping image {image_id} - file not found: {img_path}")
                    continue

                # Add image to tar
                img_info = tarfile.TarInfo(f"{base_name}.png")
                img_data = open(img_path, "rb").read()
                img_info.size = len(img_data)
                tar.addfile(img_info, io.BytesIO(img_data))

                # Add caption as txt file
                txt_info = tarfile.TarInfo(f"{base_name}.txt")
                txt_data = caption
                txt_info.size = len(txt_data.encode("utf-8"))
                tar.addfile(txt_info, io.BytesIO(txt_data.encode("utf-8")))

                # Add JSON metadata file
                metadata = {
                    "id": str(image_id),
                    "caption": caption,
                    "filename": f"COCO_val2014_{base_name}.png",
                    "timestep": timestep,
                }
                json_info = tarfile.TarInfo(f"{base_name}.json")
                json_data = json.dumps(metadata, indent=2)
                json_info.size = len(json_data.encode("utf-8"))
                tar.addfile(json_info, io.BytesIO(json_data.encode("utf-8")))

    print(f"WebDataset created with {num_shards} shards in {args.output_dir}")

    # Validate the first shard
    if num_shards > 0:
        first_shard = f"{args.output_dir}/shard_{0:05d}.tar"
        try:
            with tarfile.open(first_shard, "r") as tar:
                members = tar.getnames()
                print(f"First shard validation: {len(members)} files found")
                if len(members) > 0:
                    print(f"Sample files: {members[:3]}")
                    # Try to read one file to verify integrity
                    first_file = members[0]
                    try:
                        data = tar.extractfile(first_file).read()
                        print(f"Successfully read {first_file}: {len(data)} bytes")
                    except Exception as e:
                        print(f"Error reading {first_file}: {e}")
        except Exception as e:
            print(f"Warning: Could not validate first shard: {e}")

    print("=== PHASE 3: Cleaning up temporary files ===")

    # Clean up temporary directory
    print(f"Cleaning up temporary directory: {temp_dir}")
    shutil.rmtree(temp_dir)

    print("=== PROCESSING COMPLETE ===")
    print(f"Created {num_shards} WebDataset shards in {args.output_dir}")
    print(f"Processed subset saved to {args.output_tsv_file}")


if __name__ == "__main__":
    main()
