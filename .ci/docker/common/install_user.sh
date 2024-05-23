#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# Same as ec2-user
echo "ci-user:x:1000:1000::/var/lib/ci-user:" >> /etc/passwd
echo "ci-user:x:1000:" >> /etc/group
# Needed on Focal or newer
echo "ci-user:*:19110:0:99999:7:::" >> /etc/shadow

# Create $HOME
mkdir -p /var/lib/ci-user
chown ci-user:ci-user /var/lib/ci-user

# Allow sudo
echo 'ci-user ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/ci-user

# Test that sudo works
sudo -u ci-user sudo -v
