import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import keras




def clean(text):
    """
    cleans text for ML model

    Args:
        text (string): input 

    Returns:
        string: Cleaned text
    """
    stopword = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    text = str(text).lower()
    # remove text within square brackets
    text = re.sub('\[.*?\]', '', text)
    # remove http links
    text = re.sub('https?://\S+|www\.\S+', '', text)
    # remove html tags
    text = re.sub('<.*?>+', '', text)
    # remove punctuation
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    # remove newline chars
    text = re.sub('\n', '', text)
    # remove all word containing numbers
    text = re.sub('\w*\d\w*', '', text)
    # remove stopwords
    text = [word for word in text.split(' ') if word not in stopword]
    # applies stemming to words
    text = [stemmer.stem(word) for word in text]
    text = " ".join(text)
    return text

# implement some sort of flag to indicate DL or ML model


def data_processing(input_title, input_text):
    """
    Deals with ML model data processing
    Args:
        input_title (string): title of the news  
        input_text (string): description of the news
    Returns:
        null
    """
    data = {'Title': [input_title], 'Text': [input_text]}
    df = pd.DataFrame(data)
    df.columns = df.columns.str.replace('_', ' ').str.title()
    df['Text'] = df['Text'].apply(lambda x: clean(x))
    df['Title'] = df['Title'].apply(lambda x: clean(x))

    # for ML Models (LR, NB, RF, SVM)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['Text'])
    # Load the model from the .pkl file
    with open('model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    prediction = loaded_model.predict(X)

    # uncomment after implementing flag logic
    # if(prediction >= 0.5):
    #     return 'fake'
    # else:
    #     return 'true'

    # for DL Models
    sia = SentimentIntensityAnalyzer()
    result = {}

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Sentiment Analysis"):
        text = row['Text']
        my_id = row['Title']
        result[my_id] = sia.polarity_scores(text)

    vaders = pd.DataFrame(result).T
    vaders = vaders.reset_index().rename(columns={'index': 'Title'})
    vaders = vaders.merge(df, how='left')
    df1 = vaders.copy()
    df1['content'] = df1['Title'] + ' ' + df1['Text']
    X = df1['content']
    tokenizer = Tokenizer(num_words=10000, oov_token='OOV')
    tokenizer.fit_on_texts(X)
    sequences = tokenizer.texts_to_sequences(X)
    padded = pad_sequences(sequences, maxlen=1000)
    new_model = keras.models.load_model('lstm_model.keras')
    predictions = new_model.predict(padded)

    # uncomment after implementing flag logic
    # if(prediction >= 0.5):
    #     return 'fake'
    # else:
    #     return 'true'
