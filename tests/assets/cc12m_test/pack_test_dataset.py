# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os

import webdataset as wds


def pack_wds_dataset(tar_destination, source_folder, number_of_samples):
    """Pack cc12m dataset into a tar file using WebDataset format.
    This function is used to create the test file containing the cc12m dataset.

    Args:
        tar_destination (str): The path to the output tar file.
        source_folder (str): The path to the source folder containing the dataset.
        number_of_samples (int): The number of samples to pack.
    """

    # Create a TarWriter object to write the dataset to a tar archive
    with wds.TarWriter(tar_destination) as tar:
        # Iterate over the files in the dataset directory
        samples_cnt = 0
        for root, dirs, files in os.walk(source_folder):
            # Iterate over the files in each subdirectory
            for filename in files:
                if not filename.endswith(".jpg") or filename.startswith("."):
                    continue
                # Construct the path to the file
                img_path = os.path.join(root, filename)
                json_path = os.path.join(root, filename.replace(".jpg", ".json"))
                key = json.loads(open(json_path, "r").read())["key"]
                print(f"Saved Key to tar file: {key}")
                txt_path = os.path.join(root, filename.replace(".jpg", ".txt"))
                # Write the file and its metadata to the TarWriter
                with open(img_path, "rb") as img_file, open(
                    txt_path, "r"
                ) as txt_file, open(json_path, "r") as json_file:
                    save_dict = {
                        "__key__": key,
                        "txt": txt_file.read(),
                        "jpg": img_file.read(),
                        "json": json_file.read(),
                    }
                    tar.write(save_dict)

                samples_cnt += 1
                if samples_cnt >= number_of_samples:
                    break


if __name__ == "__main__":
    tar_destination = "tests/assets/cc12m_test/cc12m-train-0000.tar"
    source_folder = "cc12m_test"
    number_of_samples = 32
    pack_wds_dataset(tar_destination, source_folder, number_of_samples)
