# 📰 Fake vs Real News Classification

A Machine Learning project that classifies news articles as Fake or Real using NLP techniques.
The final system is deployed as an interactive Streamlit Web App where users can paste any news text and instantly get a prediction.

## 📌 Abstract

With the rapid spread of misinformation, automated fake news detection systems have become essential.
This project builds a supervised ML classifier using TF-IDF vectorization and Logistic Regression to detect whether a news article is fake or real.
A Streamlit-based interface allows real-time classification.

## 🚀 Features

✔ Preprocesses news text using NLTK
✔ Converts text into numerical vectors using TF-IDF
✔ Trains Logistic Regression classifier
✔ Provides accuracy, classification report & confusion matrix
✔ Exports trained model using Joblib
✔ User-friendly Streamlit application for predictions

## 🛠️ Tech Stack

Python
Pandas, NumPy
Scikit-learn (TF-IDF, Logistic Regression)
NLTK (stopwords, text cleaning)
Joblib (model saving/loading)
Streamlit (deployment)

## 🔧 How to Run the Project Locally

1️⃣ Clone the Repository
 ```bash
git clone https://github.com/Praneetb2929/FakeNewsClassifier.git
cd FakeNewsClassifier
 ```

2️⃣ Create a Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate   # Windows
 ```

3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
 ```

4️⃣ Train the Model (optional)

Open the notebook:
```bash
notebook/training.ipynb
 ```

5️⃣ Run the Streamlit App
```bash
cd app
streamlit run app.py
 ```

## 📊 Model Workflow

Load dataset (Fake/Real label)
Clean text: lowercasing, stopword removal, punctuation removal
Apply TF-IDF vectorization
Train Logistic Regression classifier
Evaluate model performance
Save model + vectorizer
Deploy Streamlit app for real-time predictions

## 🧪 Example Prediction
