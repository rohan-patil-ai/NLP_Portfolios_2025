# 🎬 Movie Genre Classification using TF-IDF and Machine Learning

A machine learning project that classifies movie genres based on their titles using TF-IDF vectorization and multiple classification algorithms.

---

## 📋 Overview

This project implements a text classification system that predicts movie genres from titles. It uses TF-IDF (Term Frequency-Inverse Document Frequency) for feature extraction and compares the performance of several machine learning models including Random Forest, Linear SVC, Multinomial Naive Bayes, and Logistic Regression.

---

## ✨ Features

- 🧹 Data cleaning and preprocessing from text files
- 📊 TF-IDF vectorization with unigrams and bigrams
- 🔄 Multi-model comparison using cross-validation
- 🎯 Genre prediction from movie titles
- 📈 Visualization of model performance and confusion matrices
- 🔍 Chi-square feature analysis to identify genre-specific keywords

---

## 📦 Requirements

```bash
pandas
numpy
scipy
scikit-learn
matplotlib
seaborn
```

### Installation

Install dependencies:
```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn
```

---

## 🗂️ Dataset

The project expects a `train_data.txt` file with the following format:
```
id:::title:::genre:::description
```

Each line should contain movie information separated by `:::` delimiters.

---

## 🚀 Usage

### Step 1: Prepare your data
Place your `train_data.txt` file in the project directory

### Step 2: Run the notebook
Open and execute `Tf_idf_Movie-Copy1.ipynb` in Jupyter Notebook or JupyterLab

### Step 3: Predict genres
Use the trained model to predict genres for new movie titles:
```python
movie_title = "Interstellar"
predicted_genre = predict_genre_from_title(movie_title)
print(f"The predicted genre for '{movie_title}' is: {predicted_genre}")
```

---

## 🏗️ Project Structure

The notebook follows this workflow:

### 1. Data Loading & Cleaning
- Read and parse the text file
- Remove malformed entries
- Clean column contents
- Handle duplicates

### 2. Exploratory Data Analysis
- Genre distribution visualization
- Statistical analysis of dataset

### 3. Feature Extraction
- TF-IDF vectorization
- N-gram generation (unigrams and bigrams)

### 4. Feature Analysis
- Chi-square test for genre-specific terms
- Identification of most correlated words

### 5. Model Training & Evaluation
- Train-test split
- Cross-validation
- Model comparison

### 6. Prediction
- Genre prediction function
- Confusion matrix analysis

---

## 🤖 Models Evaluated

The project compares four classification algorithms:

| Model | Description |
|-------|-------------|
| **Random Forest Classifier** | Ensemble learning method with multiple decision trees |
| **Linear Support Vector Classifier (LinearSVC)** | Linear model for classification with support vectors |
| **Multinomial Naive Bayes** | Probabilistic classifier based on Bayes' theorem |
| **Logistic Regression** | Statistical model using logistic function |

Cross-validation (CV=3) is used to evaluate model performance, with results visualized using boxplots.

---

## 📊 Results

The notebook generates comprehensive visualizations and metrics:

### Visualizations
- 📊 Genre distribution bar chart
- 📦 Model accuracy comparison boxplot
- 🔥 Confusion matrix heatmap

### Metrics
- ✅ Classification report (precision, recall, F1-scores)
- 🔍 Misclassification analysis
- 📈 Cross-validation scores
- 
---
## 🙏 Acknowledgments

- Course instructors and teaching assistants
- Open source contributors of scikit-learn, pandas, and other libraries
- Dataset providers


---

**⭐ If you found this project useful, please consider giving it a star!**
