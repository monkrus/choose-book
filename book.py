import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import vstack

# Load the books DataFrame
books_df = pd.read_csv(r"C:\Users\14802\Desktop\books_new.csv")

# Function to only return lower case letters and get rid of non-lower and white space
def preprocess_text(text):
    return str(text).lower().replace('[^a-z\s]', '')

# Add new 'processed_description' column
books_df['processed_description'] = books_df['description'].apply(preprocess_text)

# Converting text data into Term Frequency-Inverse Document Frequency vectors. 
# Note that common english words are exluded from analysis

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(books_df['processed_description'])
