# Demystifying Black-box Learning Models of Rumor Detection from Social Media Posts

This repository contains code and datasets used in research to demystify black-box machine learning and deep learning models applied to rumor detection on social media. We explore both traditional machine learning algorithms and advanced deep learning models, including ensemble and hybrid approaches, to classify social media posts as rumors or non-rumors.

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Datasets](#datasets)
- [Models Used](#models-used)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [References](#references)
- [Contributing](#contributing)
- [License](#license)

## Overview

The aim of this repository is to provide transparency in rumor detection using various machine learning and deep learning approaches. Rumor detection on social media is critical in managing misinformation and ensuring the reliability of information. This project:

1. Analyzes general social media posts and COVID-19 related posts.
2. Applies both classic and advanced models to classify posts as rumors or non-rumors.
3. Compares the performance of single models, ensemble models, and hybrid approaches.

## Repository Structure

The repository contains multiple folders and files:

```plaintext
├── 5 Model COVIDPosts/                 # ML models for COVID-related posts
├── 5 Model GeneralPosts/               # ML models for general posts
├── BERT COVIDPosts/                    # BERT model for COVID-related posts
├── BERT GeneralPosts/                  # BERT model for general posts
├── Dataset for GeneralPosts/           # Datasets for general posts
├── Datasets for COVIDPosts/            # Datasets for COVID-related posts
├── Hybrid Ensemble Model COVIDPosts/   # Hybrid ensemble model for COVID posts
├── Hybrid Ensemble Model GeneralPosts/ # Hybrid ensemble model for general posts
├── LSTM COVIDPosts/                    # LSTM model for COVID-related posts
├── LSTM GeneralPosts/                  # LSTM model for general posts
├── README.md                           # This README file
```

## Datasets

Two datasets are provided in this repository:

- **GeneralPosts**: A collection of general social media posts.
- **COVIDPosts**: A collection of social media posts related to COVID-19.

Each dataset folder contains pre-processed data required for training and testing the models. You may need to obtain or preprocess additional data according to the format used in these files if you wish to expand the dataset.

## Models Used

The following models were employed to classify posts:

### Machine Learning Models

- **Support Vector Classifier (SVC)**: A supervised machine learning model known for its effectiveness in text classification tasks.
- **XGBoost**: An optimized gradient-boosting model that is efficient and performs well on structured data.
- **Random Forest**: A popular ensemble learning method for classification, using multiple decision trees.
- **Extra Trees Classifier**: A variation of Random Forest that splits nodes randomly.
- **Decision Tree Classifier**: A basic but interpretable model that divides data into classes based on feature values.

### Hybrid Ensemble Model

A custom ensemble model combining all five machine learning models in an ensemble method to achieve higher accuracy and robustness for rumor detection.

### Deep Learning Models

- **LSTM (Long Short-Term Memory)**: A recurrent neural network model suitable for text sequences, effective in capturing contextual dependencies.
- **BERT (Bidirectional Encoder Representations from Transformers)**: A transformer-based model pre-trained on large datasets, fine-tuned here for rumor detection.

## Installation

### Requirements

- Python 3.7 or higher
- Jupyter Notebook

Required libraries:

- scikit-learn
- xgboost
- pandas
- numpy
- tensorflow
- torch
- transformers

You can install the required libraries using the following command:

```bash
pip install -r requirements.txt
```

### Clone the Repository

Clone this repository using:

```bash
git clone https://github.com/namikazi25/Codes-for-Demystifying-Black-box-Learning-Models-of-Rumor-Detection-from-Social-Media-Posts.git
cd Codes-for-Demystifying-Black-box-Learning-Models-of-Rumor-Detection-from-Social-Media-Posts
```

## Usage

### Data Preprocessing

Preprocess the datasets by cleaning, tokenizing, and encoding the posts. Code for preprocessing is included in each model's Jupyter Notebook.

### Running Models

Each model type has a dedicated folder with its own Jupyter Notebook. Open the notebook of the model you want to run and follow the instructions provided.

### Evaluating Results

Each notebook provides evaluation metrics such as accuracy, F1 score, precision, and recall.

### Example

To run the LSTM model on COVID-19 posts, navigate to the `LSTM COVIDPosts` directory and open `LSTM_COVIDPosts.ipynb` in Jupyter Notebook:

```bash
cd "LSTM COVIDPosts"
jupyter notebook LSTM_COVIDPosts.ipynb
```

Follow the instructions within the notebook to train, evaluate, and test the model on the dataset.

## Results

The performance of the models is summarized in each notebook, with details on various metrics. Generally:

- The **hybrid ensemble model** performs well in both general and COVID-19 related datasets due to the combination of multiple machine learning algorithms.
- The **BERT model**, with its deep learning capabilities, yields high accuracy and recall, especially for text-heavy data.
- The **LSTM model** captures sequential dependencies effectively but may underperform compared to BERT due to limited contextual understanding.

A detailed comparison of each model's performance is provided in each model's notebook.

## References

For more context on rumor detection techniques and the machine learning and deep learning algorithms used, see the following references:

- **SVC**: [Support Vector Machines](https://en.wikipedia.org/wiki/Support-vector_machine)
- **XGBoost**: [XGBoost Documentation](https://xgboost.readthedocs.io/)
- **Random Forest & Extra Trees**: [Random Forests and Ensemble Learning](https://scikit-learn.org/stable/modules/ensemble.html)
- **BERT**: [BERT Model Paper](https://arxiv.org/abs/1810.04805)
- **LSTM**: [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

## Contributing

We welcome contributions from the community. If you wish to contribute, please:

1. Fork the repository.
2. Create a new branch for your feature (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m "Add new feature"`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.

Please ensure your code follows best practices and includes relevant tests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
