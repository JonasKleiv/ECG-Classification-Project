# ECG Classification on the PTB-XL Dataset
This repository contains the code used in the writing of the project paper, *ECG Classification Using Traditional Machine Learning and Deep Learning Models*.

## Project Abstract
Electrocardiogram (ECG) interpretation is a complex task and subjective to the individual evaluation of medical professionals. This creates an opportunity for automated classification to improve interpretation and offload manual work. This paper covers classification of 12-lead ECG recordings from the PTB-XL dataset into five diagnostic superclasses. Traditional machine learning models are compared against deep learning models trained and evaluated using the data splits recommended by the creators of the dataset. Traditional models were trained using a combination of features extracted from PQRST waves and statistical coefficients using discrete wavelet transform (DWT). Deep learning models were trained directly on the time series data of the ECG records. The effects of class weights, feature selection and hyperparameter tuning were also explored. Overall XGBoost achieved the best results with a macro F1 score of 0.616 in the single-label and 0.707 in multi-label classification with 1D-CNN performing slightly worse. Class weighting was able to reduce the effects of class imbalance improving minority class recall. These models show potential as a clinical support tool, but further validation and more diverse testing is needed before they are ready for full clinical use.

## Repository Structure
The code used for the different experiments is organized in the following folders:
- **Traditional Machine Learning:** Data loading, preprocessing, feature extraction and selection, and experiments across different model types along with evaluation and analysis of results.
- **Deep Learning Single-Label:** Data loading, preprocessing, CNN model development and training, hyperparameter experimentation, and evaluation and analysis of results
- **Deep Learning Multi-Label:**
- **Third party/other models:** https://github.com/asmaielzia32-tech/LSTM-SVM-and-RF-ECG-PTB-XL-Classification

## Dataset 
- The project uses the PTB-XL dataset available at: https://physionet.org/content/ptb-xl/1.0.3/

## Authors
- **Jonas Isaksen Kleiv** - Traditional Machine Learning
- **Yin Hang Chan** - Deep Learning Single-Label
- **Ismail Ziai** - Third-Party/Other Models
- **Serap Yabalak** - Background, Clinical Use, and Ethics
- **Henrik Heiner** - Deep Learning Multi-Label
