# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import subprocess
import glob
import time
import os
import sys
import argparse


def get_ib_bandwidth():
    """Get the infiniband bandwidth."""
    # check if IB is avaiable.
    output = os.popen('ibstat').read()
    if 'CA type: ' not in output:
        print('ib is not avaiable')
        return

    # Read the value from Infiniband counter files.
    ib_counter_file="/sys/class/infiniband/mlx5_ib*/ports/1/counters/port_*_data"
    prev_sum = 0
    while True:
        current_sum = 0
        file_count = 0
        for file_path in glob.glob(ib_counter_file):
            file_count += 1
            with open(file_path, 'r') as f:
                current_value=int(f.read().strip())
                current_sum += current_value
        if file_count == 0:
            print(f'there is no file match {ib_counter_file}, please check')
            return

        if prev_sum != 0:
            ib_bandwidth = current_sum - prev_sum
            print(f'infiniband bandwidth:{ib_bandwidth/1e9:.2f} GB')

        prev_sum = current_sum
        time.sleep(1)


def get_nvlink_bandwidth():
    """Get the nvlink bandwidth."""
    # Get the number of GPUs on this machine
    process = subprocess.Popen(['nvidia-smi', '-L'], stdout=subprocess.PIPE)
    num_gpus = 0
    for line in process.stdout:
        line_str = line.decode().strip()
        if line_str.startswith('GPU'):
            num_gpus += 1
    print(f'num_gpus:{num_gpus}')

    # Run the command and compute nvlink bandwidth.
    process = subprocess.Popen(['dcgmi', 'dmon', '-e', '1011,1012'], stdout=subprocess.PIPE)

    gpu_ids = []
    nvlink_bandwidth=0

    for line in process.stdout:
        line_str = line.decode().strip()
        if not line_str.startswith('GPU'):
            continue
        arr = line_str.split()
        gpu_id = arr[1]
        gpu_ids.append(gpu_id)

        if arr[2] != 'N/A':
            nvlink_bandwidth += int(arr[2])
        if arr[3] != 'N/A':
            nvlink_bandwidth += int(arr[3])
        if len(gpu_ids) == num_gpus:
            print(f"nvlink bandwidth:{nvlink_bandwidth/1e9:.2f} GB")
            gpu_ids.clear()
            nvlink_bandwidth = 0


def main():
    """The entry point function."""
    parser = argparse.ArgumentParser(description='stat communication')
    parser.add_argument('--ib', action='store_true', help='stat infiniband bandwidth')
    parser.add_argument('--nvlink', action='store_true', help='stat nvlink bandwidth')

    args = parser.parse_args()
    if args.ib:
        get_ib_bandwidth()
    elif args.nvlink:
        get_nvlink_bandwidth()
    else:
        print('please specify --ib or --nvlink')
        sys.exit(1)


if __name__ == "__main__":
    main()