import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    'I love my dog',
    'I love my cat'
    'You love my dog!',
    'Do you think my dog is amazing?'
]

tokenizer = Tokenizer(num_words = 100, oov_token="<OOV>") # OOV = Out of Vocabulary

tokenizer.fit_on_texts(sentences) # SET OF SENTENCES TO TRAIN NEURAL NETWORK

word_index = tokenizer.word_index 

sequences = tokenizer.texts_to_sequences(sentences) # TOKENIZER GETS WORD INDEX AND CREATES SEQUENCES

print(word_index, '\n') 
print(sequences)

with open("sarcasm.json", 'r') as f:
    datastore = json.load(f)

sentences = []
labels = []
urls = []

for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])
