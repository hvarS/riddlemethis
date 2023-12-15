import pandas as pd
from difflib import SequenceMatcher

# Sample DataFrame (replace this with your actual DataFrame)

df = pd.read_csv('generations/gpt-turbo-instruct-simile.csv')

# Function to calculate the similarity ratio using SequenceMatcher
def similarity_ratio(text1, text2):
    return SequenceMatcher(None, text1, text2).ratio()

# Calculate statistics
df['TextLength_GenRiddle'] = df['GenRiddle'].apply(len)
df['TextLength_mFLAG'] = df['mFLAG'].apply(len)

df['WordCount_GenRiddle'] = df['GenRiddle'].apply(lambda x: len(x.split()))
df['WordCount_mFLAG'] = df['mFLAG'].apply(lambda x: len(x.split()))

df['CharDiff'] = df.apply(lambda row: abs(row['TextLength_GenRiddle'] - row['TextLength_mFLAG']), axis=1)
df['WordDiff'] = df.apply(lambda row: abs(row['WordCount_GenRiddle'] - row['WordCount_mFLAG']), axis=1)

df['SimilarityRatio'] = df.apply(lambda row: similarity_ratio(row['GenRiddle'], row['mFLAG']), axis=1)

# Display statistics
statistics = df.describe()
print(statistics)
