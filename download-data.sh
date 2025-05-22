#!/bin/bash

# Remove existing data directory
rm -rf data
mkdir -p data
cd data

echo "INSIDE $PWD"

# Download image zip files
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/zips/test2017.zip

# Download annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# Unzip everything
unzip train2017.zip
unzip val2017.zip
unzip test2017.zip
unzip annotations_trainval2017.zip

# Optional: Remove zip files to save space
rm train2017.zip val2017.zip test2017.zip annotations_trainval2017.zip

# Show result
echo "COCO 2017 data downloaded and extracted into: $PWD"
ls -lh
