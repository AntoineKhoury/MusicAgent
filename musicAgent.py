"""
This is the script to run the bot in your terminal.

After having created the database in elastic search and running it on your machine, you can launch similarity search on
a song locally. The features will be computed as it was on the FMA dataset. This part of the code is credited to them.
After these features are extracted, a cosine similarity search is performed with the dataset and the list of the n's closest
match is returned. You can modify that size with the parameters NUMBER_SIMILAR_SONGS_TO_RETURN = 1. Note that increasing that parameter
greatly affects the code's performance.


Usage:
- To run this script, execute it using the Python interpreter, passing it the name of the song to analyze as a .wav format:
    python3 musicAgent.py "Song Name"
- Also make sure you have run the creatingElasticDatabase.py script before, and that ElasticSearch is running on your machine.
- Make sure you have the necessary dependencies installed:
    - pandas                  1.2.4
    - numpy                   1.22.4
    - elasticsearch           7.13.0
    - librosa                 0.10.1
    - scipy                   1.6.3
    - transformers            4.17.0
    - torch                   1.9.0
    - their necessary dependencies

"""
#%%
import sys
import warnings
import numpy as np
from scipy import stats
import pandas as pd
import librosa
from elasticsearch import Elasticsearch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

NUMBER_SIMILAR_SONGS_TO_RETURN = 1

# Processing functions to use the LLM, this code if form Hugging Face's model description "af1tang/personaGPT"
def to_data(x):
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()

def to_var(x):
    if not torch.is_tensor(x):
        x = torch.Tensor(x)
    if torch.cuda.is_available():
        x = x.cuda()
    return x

# Utility function
flatten = lambda l: [item for sublist in l for item in sublist]

# Function to normalize the computed feature vector
def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
        return v
    return (v / norm).tolist()

#%%##############################################################################################
# This is the part of the code that is credited solely to FMA, that's their feature extraction, 
#   for more information go visit: https://github.com/mdeff/fma
def columns():
    feature_sizes = dict(chroma_stft=12, chroma_cqt=12, chroma_cens=12,
                         tonnetz=6, mfcc=20, rms=1, zcr=1,
                         spectral_centroid=1, spectral_bandwidth=1,
                         spectral_contrast=7, spectral_rolloff=1)
    moments = ('mean', 'std', 'skew', 'kurtosis', 'median', 'min', 'max')

    columns = []
    for name, size in feature_sizes.items():
        for moment in moments:
            it = ((name, moment, '{:02d}'.format(i+1)) for i in range(size))
            columns.extend(it)

    names = ('feature', 'statistics', 'number')
    columns = pd.MultiIndex.from_tuples(columns, names=names)

    # More efficient to slice if indexes are sorted.
    return columns.sort_values()


def compute_features(tid):

    features = pd.Series(index=columns(), dtype=np.float32, name=tid)

    # Catch warnings as exceptions (audioread leaks file descriptors).
    warnings.filterwarnings('error', module='librosa')
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    def feature_stats(name, values):
        features[name, 'mean'] = np.mean(values, axis=1)
        features[name, 'std'] = np.std(values, axis=1)
        features[name, 'skew'] = stats.skew(values, axis=1)
        features[name, 'kurtosis'] = stats.kurtosis(values, axis=1)
        features[name, 'median'] = np.median(values, axis=1)
        features[name, 'min'] = np.min(values, axis=1)
        features[name, 'max'] = np.max(values, axis=1)

    try:
        
        x, sr = librosa.load('Serious.wav', sr=None, mono=True)
        f = librosa.feature.zero_crossing_rate(x, frame_length=2048, hop_length=512)
        feature_stats('zcr', f)
        

        cqt = np.abs(librosa.cqt(x, sr=sr, hop_length=512, bins_per_octave=12,
                                 n_bins=7*12, tuning=None))
        assert cqt.shape[0] == 7 * 12
        assert np.ceil(len(x)/512) <= cqt.shape[1] <= np.ceil(len(x)/512)+1

        f = librosa.feature.chroma_cqt(C=cqt, n_chroma=12, n_octaves=7)
        feature_stats('chroma_cqt', f)
        f = librosa.feature.chroma_cens(C=cqt, n_chroma=12, n_octaves=7)
        feature_stats('chroma_cens', f)
        f = librosa.feature.tonnetz(chroma=f)
        feature_stats('tonnetz', f)

        del cqt
        stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
        assert stft.shape[0] == 1 + 2048 // 2
        assert np.ceil(len(x)/512) <= stft.shape[1] <= np.ceil(len(x)/512)+1
        del x

        f = librosa.feature.chroma_stft(S=stft**2, n_chroma=12)
        feature_stats('chroma_stft', f)

        f = librosa.feature.rms(S=stft)
        feature_stats('rms', f)

        f = librosa.feature.spectral_centroid(S=stft)
        feature_stats('spectral_centroid', f)
        f = librosa.feature.spectral_bandwidth(S=stft)
        feature_stats('spectral_bandwidth', f)
        f = librosa.feature.spectral_contrast(S=stft, n_bands=6)
        feature_stats('spectral_contrast', f)
        f = librosa.feature.spectral_rolloff(S=stft)
        feature_stats('spectral_rolloff', f)

        mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
        del stft
        f = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
        feature_stats('mfcc', f)

    except Exception as e:
        print('{}: {}'.format(tid, repr(e)))

    return features
#%%##############################################################################################

def main():
    if len(sys.argv) != 2:
        print('Usage: python script.py "your_song_name"')
        return

    location_song = sys.argv[1]
    print("The provided song is:", location_song)
    filename = location_song
    features = compute_features(filename)

    # Apply the normalization function to the computed feature vector
    features_normalized = normalize_vector(features)
    
    # Search for closest similarity search
    # Connect to the elastic search client
    es = Elasticsearch("http://localhost:9200")  

    # Specify the index and vector field name
    index_name = "track_index"
    vector_field_name = "normalized_vector"  

    # Formulate the query
    query = {
        "size": NUMBER_SIMILAR_SONGS_TO_RETURN,  # Return the top n results
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'normalized_vector') + 1.0",
                    "params": {
                        "query_vector": features_normalized
                    }
                }
            }
        },
        "sort": [
            {"_score": {"order": "desc"}}  # Sort by the score in descending order
        ]
    }
    # Perform the search
    response = es.search(index=index_name, body=query)

    # Formulate the response string
    closest_match = 1
    recommendation_string = ''
    for hit in response['hits']['hits']:
        
        # Accessing the entire document source
        doc_source = hit['_source']

        # Accessing a specific field, e.g., 'album date_released'
        album_title = doc_source.get('album title', 'Default Value')
        artist_location = doc_source.get('artist location', 'Default Value')
        artist_name = doc_source.get('artist name', 'Default Value')
        track_date_created = doc_source.get('track date_created', 'Default Value')
        track_genre_top = doc_source.get('track genre_top', 'Default Value')
        track_title = doc_source.get('track title', 'Default Value')
        information_string = 'The song "' +track_title+ '" is ranked '+str(closest_match)+" in similarity. It was composed by '"+artist_name+"', an artist in "+artist_location+". It appears on the album '"+album_title+"'. The genre of the track is "+track_genre_top+". "

        closest_match +=1
        
        recommendation_string += information_string

    # Create the context part of the prompt
    prompt = "You are a supporting assistant for music producer. You've just heard their song and based on the following song details, provide recommendations and insights for other songs similar: "
    prompt += recommendation_string
    prompt += ". If some information is null, NaN or 0, you don't have to mention it. You don't have to mention all the information, just some of them. You always need to mention the artist name and song names."
    
    tokenizer = GPT2Tokenizer.from_pretrained("af1tang/personaGPT")
    model = GPT2LMHeadModel.from_pretrained("af1tang/personaGPT")

    # Create functions used in with the bot
    def display_dialog_history(dialog_hx):
        for j, line in enumerate(dialog_hx):
            msg = tokenizer.decode(line)
            if j %2 == 0:
                print(">> User: "+ msg)
            else:
                print("Bot: "+msg)
                print()

    # Function to generate the next message from the bot
    def generate_next(bot_input_ids, do_sample=True, top_k=10, top_p=.92,max_length=1000, pad_token=tokenizer.eos_token_id):
        full_msg = model.generate(bot_input_ids, do_sample=True,top_k=top_k, top_p=top_p, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
        msg = to_data(full_msg.detach()[0])[bot_input_ids.shape[-1]:]
        return msg
    
    if torch.cuda.is_available():
        model = model.cuda()


    # Create the personnality of the bot with facts
    personas = []
    fact_1 = ">> Fact 1: You love music"+tokenizer.eos_token
    fact_2 = ">> Fact 2: You have just listened to a song I composed"+tokenizer.eos_token
    fact_3 = ">> Fact 3: You want to recommend me these musics that are similar to mine:"+recommendation_string+tokenizer.eos_token
    personas.append(fact_1)
    personas.append(fact_2)
    personas.append(fact_3)
    personas = tokenizer.encode(''.join(['<|p2|>'] + personas + ['<|sep|>'] + ['<|start|>']))

    # Create and start the dialog chat
    dialog_hx = []
    startingChat = True
    while True:
        if startingChat:
            user_inp = tokenizer.encode(">> User: Did you like my music?" + tokenizer.eos_token)
            print(">> User: Did you like  my music?")
            startingChat = False
        else:
            user_input = input(">> User: ")
            # Quit the chat if the user inputs 'quit'   
            if user_input.lower() == "quit":
                break
            user_inp = tokenizer.encode(">> User: " + user_input + tokenizer.eos_token)

        

        # Append the current chat to the log history
        dialog_hx.append(user_inp)
                
        # Generate the previous chat + personas details to pass to the model
        bot_input_ids = to_var([personas + flatten(dialog_hx)]).long()

        msg = generate_next(bot_input_ids)
        dialog_hx.append(msg)   
        print("Bot: {}".format(tokenizer.decode(msg, skip_special_tokens=True)))


if __name__ == "__main__":
    main()

