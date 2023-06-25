import numpy as np
import json
import pickle
from gensim.models import KeyedVectors

# Load the Word2Vec model
print("a")
model = KeyedVectors.load_word2vec_format('wikipedia-320.txt', binary=False)
print("b")
# Define the list of valid transcription file numbers
valid_list = [2, 3, 4, 5, 7, 10, 11, 17, 22, 23, 27, 34, 35]

# Initialize an empty matrix
# Initialize dictionaries instead of arrays
vector_sums = {}
vector_counts = {}

for transcript_id in valid_list:
    # Initialize arrays for each transcript
    vector_sums[transcript_id] = np.zeros((9900, 320))
    vector_counts[transcript_id] = np.zeros((9900,))

    # Load the corresponding file
    with open(f'Transcription{transcript_id}.txt', 'r') as f:
        data = json.load(f)

    # Process the data
    for entry in data["segments"]:
        for word_info in entry["words"]:
            word = word_info["text"]
            if word in model:
                vector = model[word]
                start = int(word_info["start"])
                end = int(word_info["end"])
                for i in range(start, end):
                    if end < 9900:
                        vector_sums[transcript_id][i] += vector
                        vector_counts[transcript_id][i] += 1

# Calculate the average vectors
average_vectors = {}

for transcript_id in vector_sums:
    # Avoid division by zero by replacing zero counts with one
    counts = np.maximum(vector_counts[transcript_id], 1)[:, None]

    # Calculate the average vectors
    average_vectors[transcript_id] = np.divide(vector_sums[transcript_id], counts)

# Save the matrix to a pickle file
with open('average_vectors.pkl', 'wb') as f:
    pickle.dump(average_vectors, f)