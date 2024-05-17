import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer

sentences = [
    'I love my dog',
    'I love my cat'
    'You love my dog!',
    'Do you think my dog is amazing?'
]

tokenizer = Tokenizer(num_words = 100)

tokenizer.fit_on_texts(sentences) # SET OF SENTENCES TO TRAIN NEURAL NETWORK

word_index = tokenizer.word_index 

sequences = tokenizer.texts_to_sequences(sentences) # TOKENIZER GETS WORD INDEX AND CREATES SEQUENCES

print(word_index, '\n') 
print(sequences)