import json
import os

import webdataset as wds


# Create a TarWriter object to write the dataset to a tar archive
with wds.TarWriter("torchtitan/experiments/flux/assets/cc12m-train-0000.tar") as tar:
    # Iterate over the files in the dataset directory
    for root, dirs, files in os.walk(
        "torchtitan/experiments/flux/assets/cc12m_test/cc12m-train-0000"
    ):
        # Iterate over the files in each subdirectory
        print(root, dirs, files)
        for filename in files:
            if not filename.endswith(".jpg") or filename.startswith("."):
                continue
            # Construct the path to the file

            img_path = os.path.join(root, filename)
            json_path = os.path.join(root, filename.replace(".jpg", ".json"))
            key = json.loads(open(json_path, "r").read())["key"]
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
