# AWS AI/ML Scholarship Program Fall 2024 with Udacity
**Author:** Seungmi Kim (kimsm6397@gmail.com)

**Repository:** `awsAIMLscholarship-fa24`

**Last Updated:** November 11, 2024

## Project Overview
Welcome to the awsAIMLscholarship-fa24 repository. This repository is dedicated to the projects as part of the **AI Programming with Python Nanodegree** from Udacity, awarded through the **AWS AI & ML Scholarship Program**. This Nanodegree focuses on building essential AI and Python programming skills, with an opportunity to advance into the AWS ML Fundamentals Nanodegree for selected students.

**Program Highlights**:
* Selected as part of a global initiative awarding 2,000 scholarships.
* Focus on foundational AI and Python skills with future access to specialized ML content.
* Opportunity to work on real-world projects that enhance practical understanding and programming skills in AI and machine learning.

## Project 1: City Dog Show Image Classification
The **City Dog Show Image Classification** project helps in building practical programming skills through a real-world problem where images of dogs need to be identified and classified by breed. This project demonstrates foundational image classification techniques using convolutional neural networks (CNNs), specifically focusing on three popular architectures: ResNet, AlexNet, and VGG.
### Project Goal
The main objective is to apply existing Python skills to use a pre-built image classifier function to identify dog breeds accurately. The classifier function (provided in `classifier.py`) leverages CNN architectures trained on the ImageNet dataset to classify images.
### List of Objectives
This project aims to:
* Correctly identify images of dogs (even if the breed is misclassified) and differentiate them from non-dog images.
* Classify the breed of dog for correctly identified dog images.
* Evaluate the performance of different CNN architectures (ResNet, AlexNet, VGG) to determine which one best meets the objectives.
* Compare the computation time for each algorithm to achieve a balance between accuracy and runtime.
### Project Structure and Code Outline
The following files and directories are included in this project:
* `check_images.py`: The main Python script where the majority of the project implementation occurs. It includes:  
  * Image labeling and classification
  * Timing the program
  * Using command line arguments
  * Comparing classifications
  * Calculating and printing results
* `classifier.py`: A pre-trained CNN-based classifier function. This file contains functions to classify images based on three different architectures: ResNet, AlexNet, and VGG.
* `dognames.txt`: A file that lists recognized dog breeds to help classify images as "dog" or "not dog".
* `pet_images/`: A folder containing sample images of pets used for the classification task.
### Future Improvements
Future enhancements might include fine-tuning the models specifically for the dataset in use, expanding the list of recognized dog breeds, or integrating real-time classification.
### Getting Started
#### Prerequisites
* Python 3.x: This project requires Python 3 for execution.
* Required Libraries: torch, torchvision, numpy, and other dependencies as specified in requirements.txt.
* Image Files: Ensure that the pet_images/ folder contains the images to be classified.
#### Installation
To install the required dependencies:
```bash
pip install -r requirements.txt
```
#### Usage
To execute the image classification for the dog show, run:
```bash
python check_images.py --dir pet_images/ --arch resnet  --dogfile dognames.txt > resnet_pet-images.txt
python check_images.py --dir pet_images/ --arch alexnet --dogfile dognames.txt > alexnet_pet-images.txt
python check_images.py --dir pet_images/ --arch vgg  --dogfile dognames.txt > vgg_pet-images.txt
```
or
```bash
sh run_models_batch.sh
```
#### Results
Upon execution, results including classification accuracy, breed identification accuracy, and time taken by each architecture will be printed and saved as specified.

## Project 2: Create Your Own Image Classifier
This section will contain the README documentation for the next project: Create Your Own Image Classifier. In this project, a custom image classifier will be developed using TensorFlow or PyTorch, focusing on transfer learning techniques to classify a specific set of images. Additional details will be added as the project progresses.

## Acknowledgements
Special thanks to AWS and Udacity for making this scholarship program possible and providing an incredible opportunity to develop skills in AI and machine learning.

