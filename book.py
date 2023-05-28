import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import vstack

# Load the books DataFrame
books_df = pd.read_csv(r"C:\Users\14802\Desktop\books_new.csv")
