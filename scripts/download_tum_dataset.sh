#!/bin/bash
# Download TUM fr1/desk RGB-D dataset
# Source: https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download
#
# Dataset: freiburg1/desk
# - 595 RGB frames (640×480)
# - 595 aligned depth frames (640×480, 16-bit, factor 5000)
# - Ground truth trajectory
# - Size: ~400MB

set -e

DATA_DIR="${1:-/data}"
DATASET_URL="https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz"
ASSOC_SCRIPT_URL="https://svncvpr.in.tum.de/cvpr-ros-pkg/trunk/rgbd_benchmark/rgbd_benchmark_tools/src/rgbd_benchmark_tools/associate.py"

echo "=== Downloading TUM fr1/desk dataset ==="
echo "Target: ${DATA_DIR}"

mkdir -p "${DATA_DIR}"
cd "${DATA_DIR}"

# Download dataset
if [ ! -d "rgbd_dataset_freiburg1_desk" ]; then
    echo "Downloading dataset (~400MB)..."
    wget -q --show-progress "${DATASET_URL}" -O dataset.tgz
    echo "Extracting..."
    tar xzf dataset.tgz
    rm dataset.tgz
    echo "Dataset extracted to: ${DATA_DIR}/rgbd_dataset_freiburg1_desk"
else
    echo "Dataset already exists, skipping download."
fi

cd rgbd_dataset_freiburg1_desk

# Generate associations.txt if not present
if [ ! -f "associations.txt" ]; then
    echo "Generating associations.txt..."
    # Download associate.py from TUM
    if [ ! -f "associate.py" ]; then
        wget -q "${ASSOC_SCRIPT_URL}" -O associate.py 2>/dev/null || \
        python3 -c "
import sys
# Inline associate.py (from TUM benchmark tools)
def read_file_list(filename):
    file = open(filename)
    data = file.read()
    lines = data.replace(',',' ').replace('\t',' ').split('\n')
    lst = [[v.strip() for v in line.split(' ') if v.strip()!=''] for line in lines if len(line)>0 and line[0]!='#']
    lst = [(float(l[0]),l[1:]) for l in lst if len(l)>1]
    return dict(lst)

def associate(first_list, second_list, offset=0.0, max_difference=0.02):
    first_keys = list(first_list.keys())
    second_keys = list(second_list.keys())
    potential_matches = [(abs(a-(b+offset)),a,b) for a in first_keys for b in second_keys if abs(a-(b+offset)) < max_difference]
    potential_matches.sort()
    matches = []
    first_flag = set()
    second_flag = set()
    for diff,a,b in potential_matches:
        if a not in first_flag and b not in second_flag:
            first_flag.add(a)
            second_flag.add(b)
            matches.append((a,b))
    matches.sort()
    return matches

first = read_file_list('rgb.txt')
second = read_file_list('depth.txt')
matches = associate(first, second)
for a,b in matches:
    print(f'{a:.6f} {\" \".join(first[a])} {b:.6f} {\" \".join(second[b])}')
" > associations.txt
    else
        python3 associate.py rgb.txt depth.txt > associations.txt
    fi
    echo "Generated associations.txt ($(wc -l < associations.txt) pairs)"
fi

echo ""
echo "=== Dataset Ready ==="
echo "Path: ${DATA_DIR}/rgbd_dataset_freiburg1_desk"
echo "RGB frames: $(ls rgb/*.png 2>/dev/null | wc -l)"
echo "Depth frames: $(ls depth/*.png 2>/dev/null | wc -l)"
echo "Associations: $(wc -l < associations.txt) pairs"
echo "Ground truth: $(wc -l < groundtruth.txt) poses"
