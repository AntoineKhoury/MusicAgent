"""
The file creatingElastcDatabase.py is to load the dataset provided by FMA, process it and pass it to an elastic search database.

You can provide more detailed information about the script here.
Include any important information such as the author's name, creation date,
modification history, and usage instructions.

Usage:
- Run the elasticsearch executable on your local machine. You will also need to change the security parameters of elastic search:
    - In elasticsearch.yml: set xpack.security.enabled: false
                            set xpack.security.http.ssl:
                                    enabled: false
        This is just to run your elastic search locally, as this is a simple trying out project.
        For deployement online, make sure you check the importance of those parameters

- Make sure you have the necessary dependencies installed:
    - pandas                  1.2.4
    - numpy                   1.22.4
    - elasticsearch           7.13.0
    - their necessary dependencies
"""
#%% Library imports
import pandas as pd
import numpy as np
import re
import ast
from elasticsearch import Elasticsearch,helpers

# Specify the size of the database of songs to create, to not use too much local disk space
SIZE_OF_DATABASE = 10000

# Path to the tracks features csv file
file_path = 'fma_metadata/features.csv'

# Read the CSV file
features_df = pd.read_csv(file_path, header=0)
features_df.set_index('feature', inplace=True)

# Remove the first three lines, that are just columns
features_df = features_df.iloc[3:]
features_df = features_df.rename_axis('track_id')

#%% Create a new field with all the vector elements combined
features_df['vector'] = features_df[features_df.columns.tolist()].values.tolist()

#%% Some vector elements are stored as strings representing floats, convert them
def to_float_list(value):
    if isinstance(value, str):
        # Convert string representation of list to an actual list
        value = eval(value)
    return [float(i) for i in value]

# Apply the function to the vector column
features_df['vector'] = features_df['vector'].apply(to_float_list)

#%% Normalize the vectors
# Function to normalize a vector
def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
        return v
    return (v / norm).tolist()

# Apply the normalization function to each vector
features_df['normalized_vector'] = features_df['vector'].apply(normalize_vector)

#%% Import the metadata part of the dataset, and process it

# Path to the genres.csv file
file_path = 'fma_metadata/tracks.csv'
df = pd.read_csv(file_path)

def clean_column_names(column):
    """ Function to remove ints,'.' from columns names.
    """
    pattern = r'\d+\.\d*|\d+|\.'
    return re.sub(pattern, '', column)

df.iloc[0] = df.iloc[0].astype(str)
df.iloc[1] = df.iloc[1].astype(str)
# Combine the first and second rows into column names
custom_header = df.columns + " " + df.iloc[0]
df.columns = custom_header.apply(clean_column_names)
# Drop the first two rows as they are now part of the column names
df = df.drop([0, 1]).reset_index(drop=True)
df.set_index('Unnamed:  nan',inplace=True)
df = df.rename_axis('track_id')
#%% Check the length of my vectors
# My vectors have a length of 518
len_vectors = len(features_df['normalized_vector'].values[0])
print("The length of my vectors is: " +str(len_vectors))

#%% Ensure indexes are ints, if this isn't done some indices are saved as strings and you won't be able to combine the two datasets
df.index = df.index.astype(int)
features_df.index = features_df.index.astype(int)

#%% Add the vectors to the metadata df
df = df.join(features_df['normalized_vector'])

#%% Replace NaN by null which is the correct way to use in JSON, when passing it to elasticsearch
df = df.fillna(value="null")

#%% Drop useless columns, this can be changed based on expected behavior by the user
columns_to_drop = ['album comments','album date_created','album engineer','album id','album type','artist date_created','artist id','artist latitude','artist longitude','set split','set subset','track bit_rate']
lighter_df = df.drop(columns=columns_to_drop)

#%% Remove html tags in the dataframe
# Define a function to remove HTML tags using regular expressions
def remove_html_tags(text):
    if isinstance(text, str):
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)
    else:
        return text
lighter_df = lighter_df.applymap(remove_html_tags)

#%% The genre field for now contains ints representing the genre, but it's better to have names instead, so convert that
# Convert the genres numbers to actual genres
genre_data = """1,8693,38,Avant-Garde,38
2,5271,0,International,2
3,1752,0,Blues,3
4,4126,0,Jazz,4
5,4106,0,Classical,5
6,914,38,Novelty,38
7,217,20,Comedy,20
8,868,0,Old-Time / Historic,8
9,1987,0,Country,9
10,13845,0,Pop,10
11,367,14,Disco,14
12,32923,0,Rock,12
13,730,0,Easy Listening,13
14,1499,0,Soul-RnB,14
15,34413,0,Electronic,15
16,304,6,Sound Effects,38
17,12706,0,Folk,17
18,5913,1235,Soundtrack,1235
19,773,14,Funk,14
20,1876,0,Spoken,20
21,8389,0,Hip-Hop,21
22,774,38,Audio Collage,38
25,9261,12,Punk,12
26,1952,12,Post-Rock,12
27,6041,12,Lo-Fi,12
30,3237,38,Field Recordings,38
31,1498,12,Metal,12
32,7268,38,Noise,38
33,2267,17,Psych-Folk,17
36,688,12,Krautrock,12
37,97,4,Jazz: Vocal,4
38,38154,0,Experimental,38
41,6110,38,Electroacoustic,38
42,5723,15,Ambient Electronic,15
43,210,65,Radio Art,20
45,2469,12,Loud-Rock,12
46,573,2,Latin America,2
47,2546,38,Drone,38
49,753,17,Free-Folk,17
53,2071,45,Noise-Rock,12
58,2502,12,Psych-Rock,12
63,178,9,Bluegrass,9
64,563,25,Electro-Punk,12
65,518,20,Radio,20
66,5432,12,Indie-Rock,12
70,2230,12,Industrial,12
71,600,25,No Wave,12
74,1531,4,Free-Jazz,4
76,7144,10,Experimental Pop,10
77,387,2,French,2
79,880,2,Reggae - Dub,2
81,110,92,Afrobeat,2
83,84,21,Nerdcore,21
85,3548,12,Garage,12
86,216,2,Indian,2
88,517,12,New Wave,12
89,1858,25,Post-Punk,12
90,251,53,Sludge,12
92,329,2,African,2
94,1289,17,Freak-Folk,17
97,291,4,Jazz: Out,4
98,795,12,Progressive,12
100,740,21,Alternative Hip-Hop,21
101,196,31,Death-Metal,12
102,176,2,Middle East,2
103,4162,17,Singer-Songwriter,17
107,7206,1235,Ambient,1235
109,1419,25,Hardcore,12
111,1003,25,Power-Pop,12
113,484,26,Space-Rock,12
117,62,2,Polka,2
118,608,2,Balkan,2
125,1511,38,Unclassifiable,38
130,727,2,Europe,2
137,994,9,Americana,9
138,490,20,Spoken Weird,20
166,122,65,Interview,20
167,152,31,Black-Metal,12
169,128,9,Rockabilly,9
170,47,13,Easy Listening: Vocal,13
171,234,2,Brazilian,2
172,118,2,Asia-Far East,2
173,4,86,N. Indian Traditional,2
174,17,86,South Indian Traditional,2
175,0,86,Bollywood,2
176,23,2,Pacific,2
177,72,2,Celtic,2
178,0,4,Be-Bop,4
179,104,4,Big Band/Swing,4
180,161,17,British Folk,17
181,2140,15,Techno,15
182,1482,15,House,15
183,2809,15,Glitch,15
184,1013,15,Minimal Electronic,15
185,511,15,Breakcore - Hard,15
186,682,38,Sound Poetry,38
187,292,5,20th Century Classical,5
188,301,20,Poetry,20
189,26,65,Talk Radio,20
214,40,92,North African,2
224,1916,38,Sound Collage,38
232,47,2,Flamenco,2
236,3472,15,IDM,15
240,1231,297,Chiptune,15
247,2957,38,Musique Concrete,38
250,4261,38,Improv,38
267,360,1235,New Age,1235
286,1751,15,Trip-Hop,15
296,1414,15,Dance,15
297,2208,15,Chip Music,15
311,440,13,Lounge,13
314,482,12,Goth,12
322,630,5,Composed Music,5
337,500,15,Drum & Bass,15
359,762,12,Shoegaze,12
360,201,6,Kid-Friendly,38
361,175,109,Thrash,12
362,1835,10,Synth Pop,10
374,9,20,Banter,20
377,1,19,Deep Funk,14
378,177,20,Spoken Word,20
400,823,182,Chill-out,15
401,189,181,Bigbeat,15
404,265,85,Surf,12
428,73,20,Radio Theater,20
439,314,31,Grindcore,12
440,116,12,Rock Opera,12
441,161,5,Opera,5
442,170,5,Chamber Music,5
443,216,5,Choral Music,5
444,25,5,Symphony,5
456,1392,38,Minimalism,38
465,18,20,Musical Theater,20
468,1144,15,Dubstep,15
491,78,468,Skweee,15
493,4,651,Western Swing,9
495,2061,15,Downtempo,15
502,67,46,Cumbia,2
504,114,2,Latin,2
514,1414,38,Sound Art,38
524,112,130,Romany (Gypsy),2
538,338,18,Compilation,1235
539,638,21,Rap,21
542,735,21,Breakbeat,21
567,66,3,Gospel,3
580,202,21,Abstract Hip-Hop,21
602,94,79,Reggae - Dancehall,2
619,115,130,Spanish,2
651,79,9,Country & Western,9
659,1239,5,Contemporary Classical,5
693,56,21,Wonky,21
695,258,15,Jungle,15
741,57,130,Klezmer,2
763,268,16,Holiday,38
808,12,46,Salsa,2
810,120,13,Nu-Jazz,13
811,1192,21,Hip-Hop Beats,21
906,107,4,Modern Jazz,4
1032,60,102,Turkish,2
1060,30,46,Tango,2
1156,26,130,Fado,2
1193,72,763,Christmas,38
1235,14938,0,Instrumental,1235"""
lines = genre_data.split('\n')

# Create a list of sublists containing genre ID and genre name
genre_list = [[int(line.split(',')[0]), line.split(',')[3]] for line in lines]

# Convert the list of lists into a dictionary
genre_id_to_name_dict = {genre[0]: genre[1] for genre in genre_list}

# Define a function to map IDs to names in a list, using the list created before
def map_ids_to_names(id_list):
    id_list = ast.literal_eval(id_list)
    if isinstance(id_list, list):
        # Check if the input is a list
        if len(id_list) == 1:
            # Handle single-element lists like [21]
            return [genre_id_to_name_dict[id_list[0]]]
        else:
            # Map genre IDs to names for lists with multiple elements
            return [genre_id_to_name_dict[genre_id] for genre_id in id_list]
    else:
        # Handle the case where the input is not a list (e.g., 21)
        return [genre_id_to_name_dict[id_list]]
# Apply the mapping function to 'track genre' and 'track genres_all' columns
lighter_df['track genres'] = lighter_df['track genres'].apply(map_ids_to_names)
lighter_df['track genres_all'] = lighter_df['track genres_all'].apply(map_ids_to_names)

#%% Force int columns to have the type int, and not string
columns_to_convert = ['album favorites','album listens','album tracks','artist favorites','track duration','track favorites','track interest','track listens','track number']
for column in columns_to_convert:
    lighter_df[column] = lighter_df[column].astype(int)


#%% Create the track_index with all it's fields in the ElasticSearch database
    
# Connect to the local Elasticsearch instance
es = Elasticsearch("http://localhost:9200")
index_body = {
    "mappings": {
        "properties": {
            "album_released": {
                "type": "date",
                "format": "yyyy-MM-dd HH:mm:ss",
                "null_value": None
            },
            "album_favorites": {
                "type": "integer",
                "null_value": None
            },
            "album_information": {
                "type": "text",
                "null_value": None
            },
            "album_listens": {
                "type": "integer",
                "null_value": None
            },
            "album_producer": {
                "type": "text",
                "null_value": None
            },
            "album_tags": {
                "type": "text",
                "null_value": None
            },
            "album_title": {
                "type": "text",
                "null_value": None
            },
            "album_tracks": {
                "type": "integer",
                "null_value": None
            },
            "artist_begin": {
                "type": "date",
                "format": "yyyy-MM-dd HH:mm:ss",
                "null_value": None
            },
            "artist_end": {
                "type": "date",
                "format": "yyyy-MM-dd HH:mm:ss",
                "null_value": None
            },
            "artist_associated_labels": {
                "type": "text",
                "null_value": None
            },
            "artist_bio": {
                "type": "text",
                "null_value": None
            },
            "artist_comments": {
                "type": "text",
                "null_value": None
            },
            "artist_favorites": {
                "type": "integer",
                "null_value": None
            },
            "artist_location": {
                "type": "text",
                "null_value": None
            },
            "artist_members": {
                "type": "text",
                "null_value": None
            },
            "artist_name": {
                "type": "text",
                "null_value": None
            },
            "artist_related_projects": {
                "type": "text",
                "null_value": None
            },
            "artist_tags": {
                "type": "text",
                "null_value": None
            },
            "artist_website": {
                "type": "text",
                "null_value": None
            },
            "artist_wikipedia": {
                "type": "text",
                "null_value": None
            },
            "track_comments": {
                "type": "text",
                "null_value": None
            },
            "track_composer": {
                "type": "text",
                "null_value": None
            },
            "track_date_created": {
                "type": "date",
                "format": "yyyy-MM-dd HH:mm:ss",
                "null_value": None
            },
            "track_date_recorded": {
                "type": "date",
                "format": "yyyy-MM-dd HH:mm:ss",
                "null_value": None
            },
            "track_duration": {
                "type": "integer",
                "null_value": None
            },
            "track_favorites": {
                "type": "integer",
                "null_value": None
            },
            "track_genre_top": {
                "type": "text",
                "null_value": None
            },
            "track_genres": {
                "type": "nested",
                "properties": {
                    "genre_name": {"type": "text"}
                },
                "null_value": None
            },
            "track_genres_all": {
                "type": "nested",
                "properties": {
                    "genre_name": {"type": "text"}
                },
                "null_value": None
            },
            "track_information": {
                "type": "text",
                "null_value": None
            },
            "track_interest": {
                "type": "integer",
                "null_value": None
            },
            "track_language_code": {
                "type": "text",
                "null_value": None
            },
            "track_licence": {
                "type": "text",
                "null_value": None
            },
            "track_listens": {
                "type": "integer",
                "null_value": None
            },
            "track_lyricist": {
                "type": "text",
                "null_value": None
            },
            "track_number": {
                "type": "integer",
                "null_value": None
            },
            "track_publisher": {
                "type": "text",
                "null_value": None
            },
            "track_tags": {
                "type": "text",
                "null_value": None
            },
            "track_title": {
                "type": "text",
                "null_value": None
            },
            "normalized_vector": {
                "type": "dense_vector",
                "dims": len_vectors  
            }
        }
    }
}
# Index name in Elasticsearch
index_name = "track_index"
es.indices.create(index=index_name, body=index_body, ignore=400)

#%% Code to clear the database, in case you didn't send the appropriate data
"""
es.delete_by_query(
    index=index_name,
    body={
        "query": {
            "match_all": {}
        }
    },
    refresh=True  # Ensures that the operation is written and made visible immediately
)"""

#%% Chose a subset of the dataframe to pass to elastic search: the goal of this project isn't to pass too much data and use a lot of local disk
subset = lighter_df.sample(SIZE_OF_DATABASE)

#%% Create the document generating function and pass the data to the database

def generate_document(df, index_name):
    for index,rows in df.iterrows():
        yield {
            "_index" : index_name,
            "id" : index,
            "_source" : rows.to_dict()
        }

documents = generate_document(subset,index_name=index_name)

try:
    # Here set a timeout higher to be able to pass all the elements
    helpers.bulk(es, documents, request_timeout=60)

except helpers.BulkIndexError as e:
    print(e.errors)

#%% Count how many documents are in the database
doc_count = es.count(index=index_name)['count']
print(f"Number of documents in the index '{index_name}': {doc_count}")