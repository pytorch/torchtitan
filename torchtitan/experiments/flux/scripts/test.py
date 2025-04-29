<<<<<<< HEAD
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import tarfile
from multiprocessing import Pool
=======
import logging
import os
import tarfile
from multiprocessing import cpu_count, Pool
>>>>>>> ef9eba2 (preprocess)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger()
# Create a file handler to write logs to a file
file_handler = logging.FileHandler("tarfile_check_multiprocessing.log")
file_handler.setLevel(logging.INFO)
# Add the file handler to the logger
logger.addHandler(file_handler)


def check_tarfile_integrity(tarfile_path):
    try:
        with tarfile.open(tarfile_path, "r") as tar:
            for member in tar:
                try:
                    tar.getmember(member.name)
                except KeyError:
                    logger.error(
                        f"Corrupted header found in {tarfile_path}: {member.name}"
                    )
                    return False
        return True
    except tarfile.TarError as e:
        logger.error(f"Error opening {tarfile_path}: {e}")
        return False


def process_tarfile(filepath):
    logger.info(f"Checking {filepath}...")
    if check_tarfile_integrity(filepath):
        logger.info(f"{os.path.basename(filepath)} is OK.")
    else:
        logger.info(f"{os.path.basename(filepath)} has a corrupted header.")


def check_all_tarfiles_in_directory(directory):
    tarfiles = [
        os.path.join(directory, filename)
        for filename in os.listdir(directory)
        if filename.endswith(".tar")
    ]
    with Pool(processes=64) as pool:
        pool.map(process_tarfile, tarfiles)


# Replace 'your_directory_path' with the path to the directory containing your tar files
check_all_tarfiles_in_directory("/home/jianiw/tmp/mffuse/cc12m-wds")
