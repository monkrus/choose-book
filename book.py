import re
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import vstack

# Load->Normalize->Add column->Analize->Set algo->Map->Recommendation

# Load the books DataFrame
# Replace "1.csv" with your csv file
books_df = pd.read_csv(r"1.csv")

# Function to only return lower case letters and get rid of non-lower and white space
def preprocess_text(text):
    return re.sub('[^a-z\s]+', '', str(text).lower())

# Add new 'processed_description' column
books_df['processed_description'] = books_df['description'].apply(preprocess_text)

# Converting text data into Term Frequency-Inverse Document Frequency vectors 
# Note that common English words are excluded from analysis.
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(books_df['processed_description'])

# Number of neighbors to return, including the input book
# Note that to include 5 neighbors, we set the number to 6 
# The reason is the input book is given as a recommendation as well
n_neighbors = 6  
model = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
model.fit(tfidf_matrix)

# Create a mapping from book titles to their index 
title_to_index = pd.Series(books_df.index, index=books_df['title'].apply(lambda x: x.lower()))

# Recommendation function
def recommend_books(title, model=model, books_df=books_df, title_to_index=title_to_index):
    idx = title_to_index[title.lower()]
    distances, indices = model.kneighbors(tfidf_matrix[idx])
    indices = indices[0][1:]  # Remove the input book from the recommendations
    return books_df['title'].iloc[indices].values.tolist()

def main():
    while True:
        book_title = input("Enter the book title (Type 'end' to end the program):")
        if book_title.lower() == 'end':
            break
        if book_title.lower() not in title_to_index:
            print("Sorry! Book not found in the database. Try another book!")
            continue

        recommendations = recommend_books(book_title)
        print(f"\nThe following are the best recommendations for '{book_title}':\n")
        print(recommendations)
        print("\n")

if __name__ == "__main__":
    main()
