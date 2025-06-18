#!/bin/bash

# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


: "${DOWNLOAD_PATH:=/datasets/coco2014}"

while [ "$1" != "" ]; do
    case $1 in
        -d | --download-path )       shift
                                     DOWNLOAD_PATH=$1
                                     ;;
    esac
    shift
done

mkdir -p ${DOWNLOAD_PATH}
cd ${DOWNLOAD_PATH}

wget -c http://images.cocodataset.org/zips/val2014.zip
wget -c http://images.cocodataset.org/annotations/annotations_trainval2014.zip

echo "fbedd73593f242db65cce6bcefde193fcedcc5c0  ./val2014.zip"                    | sha1sum -c
echo "8e0b9df54c175f1688400e98d1a97f292e726870  ./annotations_trainval2014.zip"   | sha1sum -c

unzip val2014.zip
unzip annotations_trainval2014.zip
