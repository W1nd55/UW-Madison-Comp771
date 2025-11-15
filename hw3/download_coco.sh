#!/bin/bash
# script for downloading the COCO dataset

cd data

# Create coco directory if it doesn't exist
mkdir -p coco
cd coco

# Download COCO 2017 images
echo "Downloading COCO 2017 training images..."
wget http://images.cocodataset.org/zips/train2017.zip
echo "Downloading COCO 2017 validation images..."
wget http://images.cocodataset.org/zips/val2017.zip
echo "Downloading COCO 2017 test images..."
wget http://images.cocodataset.org/zips/test2017.zip

# Download COCO 2017 annotations
echo "Downloading COCO 2017 annotations..."
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# Extract all files
echo "Extracting images and annotations..."
unzip -q train2017.zip
unzip -q val2017.zip
unzip -q test2017.zip
unzip -q annotations_trainval2017.zip

# Clean up zip files
echo "Cleaning up zip files..."
rm train2017.zip val2017.zip test2017.zip annotations_trainval2017.zip

echo "COCO dataset download complete!"

cd ../..

