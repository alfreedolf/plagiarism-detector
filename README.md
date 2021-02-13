# plagiarism-detector
# Text Plagiarism Project, Machine Learning Deployment

This repository contains an implementation of a text plagiarism detector. The entire implementation is based on AWS SageMaker platform.

## Project Overview

In this project, I've built a plagiarism detector that examines a text file and performs binary classification on it labeling that file as either *plagiarized* or *not*, depending on how similar that text file is to a provided source text. Detecting plagiarism is an active area of research; the task is non-trivial and the differences between paraphrased answers and original work are often not so obvious.

This project has been broken down into three main notebooks:

**Notebook 1: Data Exploration**
* Loading in the corpus of plagiarism text data.
* Exploring the existing data features and the data distribution.

**Notebook 2: Feature Engineering**

* Cleaning and pre-process the text data.
* Defining features for comparing the similarity of an answer text and a source text, and extract similarity features.
* Selecting "good" features, by analyzing the correlations between different features.
* Creating train/test `.csv` files that hold the relevant features and class labels for train/test data points.

**Notebook 3: Train and Deploy Your Model in SageMaker**

* Uploading my train/test feature data to S3.
* Defining a binary classification model and a training script.
* Training my model and deploy it using SageMaker.
* Evaluating my deployed classifier.
* This part has been implemented with 4 distinct approaches:
    1. a Linear Learner from the SageMaker available models,
    2. an AdaBoost model from sklearn library,
    3. a custom defined Pytorch model,
    4. a custom defined Tensorflow model (work in progress)   
---

Please see the [README](https://github.com/udacity/ML_SageMaker_Studies/tree/master/README.md) in the base project from which I've implemented this detector in AWS. 
