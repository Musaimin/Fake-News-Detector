# Fake-News-Detector
 Detecting fake news using the title and description of the news using machine learning and deep learning models. Here is the link to the detailed <a href="https://github.com/Musaimin/Fake-News-Detector/tree/main/report/FakeNewsClassifier.pdf"> report </a>.

## Overview
Fake news spreading everywhere has become a big problem in this digital age. Traditional methods for identifying fake news are often insufficient. This project aims to address this issue by employing machine learning and deep learning models to effectively detect and classify fake news articles. The goal is to Identify fake news with high accuracy and precision while maintaining high speeds suitable for real-time applications.

## Features
-  **Identifying news:** Distinguishing real news from fake news available on social networks and news portals.
-  **User Interface:** Interactive and easy to use User Interface. 
-  **Models used:** Multiple training models for fake news detection.

## Models Used
- Logistic Regression
- Support Vector Machine (SVM) with different kernels
- Multinomial Naive Bayes
- Random Forest
- Simple Recurrent Neural Network (RNN) 
- Long Short Term Memory (LSTM)

## Dataset
Here is the link to the dataset:
<a href="https://www.kaggle.com/datasets/bhavikjikadara/fake-news-detection"> Fake News Detection </a>.

## Usage
### Setup
- Desktop with RAM 32 GB, processor Intel Core i5 13500 and GPU
AMD RX 6800
- Software used: VS Code and Python3 with Jupyter Notebook
- Streamlit Python library used for developing UI

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Musaimin/Fake-News-Detector.git

2. Install the required packages:

   ```bash
    pip install -r requirements.txt

### Running the App

1. To test our application, run the following command in the project directory's terminal:

   ```bash
    streamlit run app.py

2. This will automatically navigate to a local browser page: http://localhost:8500.

3. After that by giving inputs from the csv files (true.csv, fake.csv) and choosing the required model, output will be shown in the browser.

### Limitations

    * Trained models using only one dataset.
    * Didn’t use cross-validation techniques here.
    * Have only done classification for English    language news.
    * Only available in web platform.

### Future Work

    * To do fine-tuning model parameters on our app.
    * Exploring advanced architectures/models for further improvements
    * Incorporating data in different formats (such as- images, videos).
    * Expanding the dataset to include multiple languages. 
    * To setup our application in mobile platform.




