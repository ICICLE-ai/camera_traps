
# Models

## Overview
MegaDetector is an AI-powered computer vision model built using YoloV5 architecture, designed to detect animals, humans, vehicles, and false triggers in camera trap images. MegaDetector assists in automating the analysis of vast amounts of imagery collected from camera traps, aiding in wildlife research and conservation efforts. 

## Model Versions

### [MegaDetector v5b](https://github.com/ICICLE-ai/camera_traps/blob/main/models/md_v5b.0.0.pt)
- **Training Data**: 
  - Incorporates public data from:
    - [Caltech Camera Traps](https://lila.science/datasets/caltech-camera-traps)
    - [Snapshot Serengeti](https://lila.science/datasets/snapshot-serengeti)
    - [Idaho Camera Traps](https://lila.science/datasets/idaho-camera-traps/)
    - [WCS Camera Traps](https://lila.science/datasets/wcscameratraps)
    - [NACTI (North American Camera Trap Images)](https://lila.science/datasets/nacti)
    - [Island Conservation Camera Traps](https://lila.science/datasets/island-conservation-camera-traps)
    - [Orinoquía Camera Traps](https://lila.science/datasets/orinoquia-camera-traps)
    - [SWG Camera Traps](https://lila.science/datasets/swg-camera-traps)
    - [ENA24](https://lila.science/datasets/ena24)
    - [Wellington Camera Traps](https://lila.science/datasets/wellington-camera-traps)
    - [Snapshot Safari](https://lila.science/datasets/snapshot-safari)

### [MegaDetector v5a](https://github.com/ICICLE-ai/camera_traps/blob/main/models/md_v5a.0.0.pt)
- **Training Data**: 
  - Built on all the training data from MegaDetector v5b.
  - Includes new public (non-camera-trap) data from:
    - [iNaturalist Dataset 2017](https://github.com/visipedia/inat_comp)
    - [COCO (Common Objects in Context)](https://cocodataset.org/#home)

### [MegaDetector v5a fine-tuned model](https://github.com/ICICLE-ai/camera_traps/blob/main/models/md_v5a.0.0_ena.pt)
The original MegaDetector model is finetuned based on [Eastern North America](https://lila.science/datasets/ena24) to detect input images at species level. The ENA24 dataset comprises approximately 8,000 images, primarily captured in Eastern North America. The labeled dataset has 23 classes and the most commonly found species are American cow, American black bear, and Dog.
This model was trained by Sikan Li and added a few [notes](https://docs.google.com/document/d/1j5deOeSZy3slXwkTYVAF1Q6BfSBSXH-RfN-CFn3amNM/edit) here.

### [MegaDetector v5a optimized model](https://github.com/ICICLE-ai/camera_traps/blob/main/models/mdv5_optimized.pt)
[Dan Morris](emailto:agentmorris@gmail.com) developed the model for ConservationXLabs to use in their Sentinel device.

### [MegaDetector v6](https://github.com/ICICLE-ai/camera_traps/blob/main/models/MDV6b-yolov9c.pt)
The model is developed using Yolov9 architecture and is currently under beta testing for the public.

# [Ground Truth](https://github.com/ICICLE-ai/camera_traps/tree/main/ground_truth)
This repository contains three files with ground truth data. The [file](https://github.com/ICICLE-ai/camera_traps/blob/main/ground_truth/ground_truth_ena.csv) species-level ground truth information for the ENA dataset, derived from [metadata](https://storage.googleapis.com/public-datasets-lila/ena24/ena24.json) available on lila.science. Using this metadata, the following file was created for object detection [file](https://github.com/ICICLE-ai/camera_traps/blob/main/ground_truth/ground_truth_ena_megadetector.csv). The [file](https://github.com/ICICLE-ai/camera_traps/blob/main/ground_truth/ground_truth_15_images.csv) contain ground truth for 15 images that were downloaded from the google and the file was created manually by Sowbaranika.

