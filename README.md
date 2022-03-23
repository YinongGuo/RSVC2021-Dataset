# RSVC2021 Dataset
A dataset for Vehicle Counting in Remote Sensing images

This repository contains the generating codes for RSVC2021 dataset.

## Generation of RSVC2021

### Preparation
- Prerequisites
  - Python 3.x
  - numpy
  - scipy
  - opencv-python

- OS Environment  
  The code has been tested on both Windows 10 and Ubuntu 18.04 and should be able to execute on Windows and Linux.

- Data Preparation
  - RSVC2021 is originated from two public Remote Sensing datasets: DOTA and ITCVD. You should download these two datasets before running our codes.
    - [**DOTA dataset (click for link)**](https://captain-whu.github.io/DOTA/dataset.html)   
      Note that we only need the training set and the validation set because only these two parts have annotations.
    - [**ITCVD dataset (click for link)**](https://research.utwente.nl/en/datasets/itcvd-dataset)
  - You can customize the storage location of the datasets, but the internal folder tree of each dataset must be organized as follows:
    - DOTA dataset:
      ```
      +-- ... (root)
      |   +-- train
      |   |   +-- images
      |   |   |   +-- P0000.png
      |   |   |   +-- ...
      |   |   +-- labelTxt
      |   |   |   +-- P0000.txt
      |   |   |   +-- ...
      |   +-- val
      |   |   +-- images
      |   |   |   +-- P0003.png
      |   |   |   +-- ...
      |   |   +-- labelTxt
      |   |   |   +-- P0003.txt
      |   |   |   +-- ...
      ```
    - ITCVD dataset:
      ```
      +-- ... (root)
      |   +-- Training
      |   |   +-- Image
      |   |   |   +-- 00000.jpg
      |   |   |   +-- ...
      |   |   +-- GT
      |   |   |   +-- 00000.mat
      |   |   |   +-- ...
      |   +-- Testing
      |   |   +-- Image
      |   |   |   +-- 00007.jpg
      |   |   |   +-- ...
      |   |   +-- GT
      |   |   |   +-- 00007.mat
      |   |   |   +-- ...
      ```
      
  
### Running for Generation
- Image Processing and Label Generation  
  The workflows of image processing and label generation are integrated in `construct_RSVC.py`. Please run this script as follows:
  ```
  python construct_RSVC.py --DOTA_ROOT [Path to your DOTA dataset] \
      --ITCVD_ROOT [Path to your ITCVD dataset] \
      --OUTPUT_ROOT [Path where you want to store the RSVC2021 dataset]
  ```
- Density Map Generation  
  This repository will not provide the relevant codes, as this is not our original work. If you need to generate density maps based on annotations, please refer to other projects, such as [**C-3-Framework**](https://github.com/gjy3035/C-3-Framework). Note that he annotation format we provide (see below) should be similar to conventional crowd counting datasets and easy to process.

## Tips and Explanation
### Annotation Format
  ```
  Text Format：
  Line 1：GSD after downsampling
  Line 2：Number of cars
  Line 3 and after：Coordinates of center points of cars

  e.g.
  GSD:1.14514
  numCar:1919
  3216.7 1049.2
  ...
  ```

### About Discarded Images
Some images in DOTA or ITCVD datasets are discarded due to their ill-suited properties for this task, as listed below:
- DOTA dataset
  A lot of images are discarded according to the judgment process in the paper, but there are still some images will be abandoned due to their incomplete annotations: 
  ```
  P0161 P1384 P2282 P2283 P2287 P2645 P2702 P2704
  ```
- ITCVD dataset
  Images numbered 00071 and later will be discarded, because these images have oblique viewing angles and are thus not within the scope of our work.

### About Validation
The RSVC2021 dataset generated by this code only contains two parts: training set and testing set. But validation is necessary for training of deep-learning models. Therefore, in practical application or research, it is recommended to choose about 10% of the training set images as the validation set.

## Citation
TBD
