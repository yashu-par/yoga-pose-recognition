# Yoga Pose Recognition System

## Project Description

This project presents an advanced Yoga Pose Recognition System based on a multi-stage three-branch deep learning architecture. It combines ResNet50 and MobileNetV2 for visual feature extraction with PoseNet-based joint angle detection to improve classification accuracy.

The model classifies five yoga poses — Downdog, Goddess, Plank, Tree, and Warrior — achieving a validation accuracy of 92.02%. A Streamlit-based web application is developed to provide real-time pose prediction with skeleton overlay, joint angle visualization, and class probability analysis.

The system implements a complete pipeline including data cleaning, preprocessing, augmentation, model training, and deployment, making it a scalable and practical solution for intelligent fitness applications.

---

## Features

* Image-based yoga pose classification
* Multi-branch deep learning architecture
* Skeleton overlay with detected keypoints
* Real-time joint angle computation
* Class probability visualization
* Batch image prediction support
* High model accuracy (92.02%)
* Interactive web interface using Streamlit

---

## Model Architecture

The system uses a three-branch combined architecture:

* ResNet50 (pretrained) for deep visual feature extraction
* MobileNetV2 (pretrained) for lightweight complementary features
* Joint angle branch using PoseNet keypoints and geometric angle computation

The outputs from all branches are concatenated and passed through fully connected layers for final classification into five pose categories.

---

## Pose Classes

* Downdog
* Goddess
* Plank
* Tree
* Warrior

---

## Model Performance

| Metric              | Value  |
| ------------------- | ------ |
| Validation Accuracy | 92.02% |
| Best Epoch          | 13     |
| Validation Loss     | 0.2646 |

The model demonstrates strong performance across all classes with balanced precision, recall, and F1-scores.

---

## Dataset

* Source: Kaggle Yoga Pose Dataset
* Total Images: Approximately 1550 (cleaned to 813)

### Data Processing

* Removal of duplicate images using hashing techniques
* Handling of class imbalance using computed class weights
* Data augmentation including rotation, zoom, flip, brightness, and shift transformations
* Image preprocessing with aspect ratio preservation

---

## Technology Stack

* Python 3.12
* TensorFlow 2.18 and Keras
* ResNet50 and MobileNetV2
* PoseNet (TensorFlow Lite)
* OpenCV
* Streamlit

---

## Web Application

The project includes a Streamlit-based web interface with the following capabilities:

* Single image upload for pose prediction
* Batch processing of multiple images
* Visualization of detected skeleton keypoints
* Display of joint angles for pose analysis
* Class-wise probability outputs

---

## Project Structure

yoga-pose-recognition/
│── app.py
│── yoga_pose_recognition.ipynb
│── TRAIN/
│── best_angle_model.keras
│── movenet.tflite
│── requirements.txt
│── README.md

---

## Challenges and Solutions

| Challenge                      | Solution                                      |
| ------------------------------ | --------------------------------------------- |
| Class imbalance                | Addressed using class weighting techniques    |
| Duplicate data                 | Removed using hash-based duplicate detection  |
| Pose detection complexity      | Implemented PoseNet with custom decoding      |
| Real-world image variability   | Applied aspect ratio preserving preprocessing |
| Framework compatibility issues | Configured environment using Python 3.12      |

---

## Future Work

* Expansion to larger datasets with more pose classes
* Real-time webcam-based pose detection
* Integration of pose correction feedback mechanisms
* Deployment on cloud platforms
* Extension to video-based pose analysis

---

## References

* Kaggle Yoga Pose Dataset
* TensorFlow and Keras Documentation
* ResNet50 and MobileNetV2 Research Papers
* PoseNet by Google Research
* Streamlit Documentation

---

## Author

Yashasvee Masani
MCA Student | Aspiring Software Developer
