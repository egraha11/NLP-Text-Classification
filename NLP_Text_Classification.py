import keras_nlp.layers.position_embedding
import pandas as pd
import numpy as np
import keras
from sklearn import preprocessing
import nltk
from nltk.corpus import stopwords
from keras import layers
import keras_nlp
import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


data = pd.read_csv("ecommerceDataset.csv", names=["Text Label", "Desc"], header=None)

encoder = preprocessing.LabelEncoder()

data['Label']= encoder.fit_transform(data['Text Label'])

data["Desc"] = data["Desc"].str.lower()


#replace/delete characters that dont provide vital info (remove commas and parenthesis)
def remove_words(df):
    remove_list = ["/", "'", ")", "(", "[", "]", "|"]

    for char in remove_list:
        df["Desc"] = df["Desc"].str.replace(char, "")

data["Desc"] = remove_words(data)

#remove stop words
def remove_stop_words(df):
    nltk.download('stopwords')
    stop = stopwords.words('english')

    df["Desc"] = df["Desc"].apply(lambda x: ' '.join([word for word in str(x).split() if word not in (stop)]))

data["Desc"] = remove_stop_words(data)

X_train, X_test, y_train, y_test = train_test_split(data["Desc"], data["Label"], test_size=0.33)

corpus = X_train.to_numpy().flatten().tolist()


corpus = ' '.join(corpus)

pad_length = 15000

#text vectorize
vectorizer = layers.TextVectorization(max_tokens=40000, output_sequence_length=pad_length)
#text_ds = tf_data.Dataset.from_tensor_slices(train_samples).batch(128)
vectorizer.adapt(corpus)

print("\n\n")
voc = vectorizer.get_vocabulary()
word_index = dict(zip(voc, range(len(voc))))




#download GLoVE word embedding
embeddings_index = {}
with open("glove.6B.50d.txt", encoding="utf-8") as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

embedding_matrix = np.zeros((len(voc), 50))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    embedding_matrix[i] = embedding_vector


embedding_layer = layers.Embedding(
    len(voc),
    50,
    #trainable=False,
)
embedding_layer.build((1,))
embedding_layer.set_weights([embedding_matrix])



int_sequences_input = keras.Input(shape=(None,), dtype="float32")


x = vectorizer(int_sequences_input)

x = embedding_layer(x)


#positional encoding 
position_embeddings = keras_nlp.layers.PositionEmbedding(max_length=pad_length)


x = position_embeddings(x)


#Multi Head Attention Layer
multi_head_layer = layers.MultiHeadAttention(num_heads=5, key_dim=2)

target = keras.Input(shape=[15000, 1], dtype="float32")
source = keras.Input(shape=[15000, 1], dtype="float32")
output_tensor, weights = multi_head_layer(target, source,return_attention_scores=True)

x = multi_head_layer(x)

#Normalize
x = layers.BatchNormalization()(x)


#Flatten
x = layers.Flatten()(x)

x = layers.Dense(15000, input_shape = (15000,)) 

x = layers.Dropout(0.2)

output = layers.Dense(2, input_shape = (15000,4), activation="softmax") 

model = keras.Model(int_sequences_input, output)

model.compile(loss="categorical_crossentropy", optimizer="adam")

model.fit(X_train, y_train, epochs=10, batch_size=10)



"""
#for layer in model.layers:
    #print(layer.output_shape)

model.summary()
"""

#evaluate model 
y_pred = model.predict(data["Desc"])

print(accuracy_score(y_test, y_pred))

#save weights
cwd = os.getcwd()
model.save(cwd+".keras")
model.save_weights(cwd+".h5", overwrite=True)


#reconstructed_model = keras.models.load_model("my_model.keras")

