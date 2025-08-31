# Sentiment Analysis with LoRA-Finetuned BERT

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![Framework: Streamlit](https://img.shields.io/badge/Framework-Streamlit-cyan.svg)](https://streamlit.io/)

## Overview

This project is a sentiment analysis system designed to identify and categorize emotions from English text. The model is fine-tuned using the LoRA (Low-Rank Adaptation) technique on a pre-trained BERT-base-uncased model. The application provides a user-friendly interface built with Streamlit for real-time sentiment prediction.
The primary goal of this project is to monitor and track user sentiment on social media platforms.

## ✨Features

- **Efficient Fine-Tuning**: Utilizes LoRA (Low-Rank Adaptation) to fine-tune the model efficiently, reducing the number of trainable parameters and computational cost.
- **Multi-class Classification**: Supports 6 distinct emotion labels: sadness, joy, love, anger, fear, and surprise.
- **Cloud-Based Training**: Integrated with `Modal` to run the training process on high-performance GPUs.
- **Interactive Web App**: A user-friendly interface created with `Streamlit` to perform real-time sentiment prediction on custom text.
- **Modular Codebase**: Well-structured and organized source code for data loading, preprocessing, model definition, training, and prediction.

## 🌟 Demo

![Application Demo](https://github.com/HuyTQ-28/viet_summarizer/issues/2#issue-3370419435)

## 🛠️ Technology Stack

- **Deep Learning**: `PyTorch`, `Transformers`
- **Efficient Fine-Tuning**: `PEFT (Parameter-Efficient Fine-Tuning)` with `LoRA`
- **Web Framework**: `Streamlit`
- **Cloud Computing**: `Modal`
- **Data Manipulation**: `Pandas`, `NumPy`
- **Evaluation**: `Scikit-learn`

## 📂 Project Structure

```
.
├── app.py                  # Streamlit web application
├── modal_train.py          # Script for cloud-based training with Modal
├── config.yaml             # Main configuration file
├── requirements.txt        # Project dependencies
├── data/                   # Datasets (raw, processed)
├── models/                 # Saved models
├── notebooks/              # Jupyter notebooks for exploration and evaluation
└── src/                    # Source code
    ├── __init__.py
    ├── config.py           # Configuration loading
    ├── data_loader.py      # Data loading logic
    ├── model.py            # Model definition and LoRA setup
    ├── predict.py          # Prediction functions
    ├── preprocess.py       # Data preprocessing scripts
    ├── train.py            # Main training script
    └── utils.py            # Utility functions
```

## Model Performance

- Dataset: The model was trained on the **Sentiment Analysis for NLP** dataset from Kaggle. This dataset contains text samples labeled with six different emotions: sadness, joy, love, anger, fear, and surprise. **Link**: https://drive.google.com/drive/folders/1BWSHm4ZjyLOskstB3Dgl0SBYOHNaKZ4p?usp=drive_link
- Performance: The model achieves an impressive accuracy of approximately **92.87%** on the test set. Here is a detailed classification report:

```
              precision    recall  f1-score   support

     sadness     0.9209    0.9106    0.9157      1241
         joy     0.9390    0.8806    0.9089      1240
        love     0.9225    0.9396    0.9309      1241
       anger     0.9307    0.9412    0.9359      1241
        fear     0.9367    0.9065    0.9214      1241
    surprise     0.9236    0.9936    0.9573      1241

    accuracy                         0.9287      7445
   macro avg     0.9289    0.9287    0.9283      7445
weighted avg     0.9289    0.9287    0.9284      7445
```

## ⚙️ Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

- Python 3.10 or higher

### Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository-url>
    cd sentiment_analysis
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Usage (Running the Application)

Once the installation is complete, you can run the Streamlit web application to interact with the model.

1.  **Ensure you have a fine-tuned model.** You can train your own (see the _Training_ section below).

2.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
    This will open a new tab in your browser with the application interface.

### Training the Model

You can train the model either locally or using Modal for cloud-based training. The trained model artifacts will be saved to the `checkpoints` directory, which you can then move to `models/final_model` to use with the app.

#### Local Training (if you have a capable GPU):

- Make sure your data is correctly set up in the `data/processed` directory.
- Run the training script:
  ```bash
  python -m src.train
  ```

#### Cloud-Based Training with Modal:

- Make sure you have Modal installed and configured (`pip install modal`, `modal setup`).
- Run the Modal training script:
  ```bash
  modal run modal_train.py
  ```

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for more details
