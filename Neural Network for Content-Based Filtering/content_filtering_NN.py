import numpy as np
import numpy.ma as ma
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import tabulate
from recsysNN_utils import *
pd.set_option("display.precision", 1)

# Load data from CSV files
top10_df = pd.read_csv("./data/content_top10_df.csv")
bygenre_df = pd.read_csv("./data/content_bygenre_df.csv")
top10_df
bygenre_df

# Load Data, set configuration variables
item_train, user_train, y_train, item_features, user_features, item_vecs, movie_dict, user_to_genre = load_data()

num_user_features = user_train.shape[1] - 3  # remove userid, rating count and ave rating during training
num_item_features = item_train.shape[1] - 1  # remove movie id at train time
uvs = 3  # user genre vector start
ivs = 3  # item genre vector start
u_s = 3  # start of columns to use in training, user
i_s = 1  # start of columns to use in training, items
print(f"Number of training vectors: {len(item_train)}")

# Print training data
pprint_train(user_train, user_features, uvs, u_s, maxcount=5)
pprint_train(item_train, item_features, ivs, i_s, maxcount=5, user=False)
print(f"y_train[:5]: {y_train[:5]}")

# Scale training data
item_train_unscaled = item_train
user_train_unscaled = user_train
y_train_unscaled = y_train

scalerItem = StandardScaler()
scalerItem.fit(item_train)
item_train = scalerItem.transform(item_train)

scalerUser = StandardScaler()
scalerUser.fit(user_train)
user_train = scalerUser.transform(user_train)

scalerTarget = MinMaxScaler((-1, 1))
scalerTarget.fit(y_train.reshape(-1, 1))
y_train = scalerTarget.transform(y_train.reshape(-1, 1))

# Check if transformations are correct
print(np.allclose(item_train_unscaled, scalerItem.inverse_transform(item_train)))
print(np.allclose(user_train_unscaled, scalerUser.inverse_transform(user_train)))

item_train, item_test = train_test_split(item_train, train_size=0.80, shuffle=True, random_state=1)
user_train, user_test = train_test_split(user_train, train_size=0.80, shuffle=True, random_state=1)

# Define NN models, loss, and accuracy
def item_NN(input_dim):
    inputs = keras.layers.Input(shape=(input_dim,))
    hidden1 = keras.layers.Dense(64, activation='relu')(inputs)
    hidden2 = keras.layers.Dense(64, activation='relu')(hidden1)
    outputs = keras.layers.Dense(32)(hidden2)
    return keras.Model(inputs, outputs)

input_item = tf.keras.layers.Input(shape=(num_item_features)) 
vm = item_NN(input_item)  
vm = tf.linalg.l2_normalize(vm, axis=1)  
model = tf.keras.Model(input_item, vm)
model.summary()

# Predict for testing
item_vecs_scaled = scalerItem.transform(item_vecs[:, i_s:])
vms = model.predict(item_vecs_scaled)
print(f"Predicted movie feature vectors: {vms.shape}")

# Calculate squared distances
def sq_dist(a, b):
    d = np.linalg.norm(a-b)
    return d**2

a1 = np.array([1.0, 2.0, 3.0]); b1 = np.array([1.0, 2.0, 3.0])
a2 = np.array([1.1, 2.1, 3.1]); b2 = np.array([1.0, 2.0, 3.0])
print(f"squared distance between a1 and b1: {sq_dist(a1, b1):0.3f}")

# Movie similarity comparison
count = 50  # number of movies to display
dim = len(vms)
dist = np.zeros((dim,dim))

for i in range(dim):
    for j in range(dim):
        dist[i,j] = sq_dist(vms[i, :], vms[j, :])

m_dist = ma.masked_array(dist, mask=np.identity(dist.shape[0]))  # mask the diagonal
disp = [["movie1", "genres", "movie2", "genres"]]

for i in range(count):
    min_idx = np.argmin(m_dist[i])
    movie1_id = int(item_vecs[i, 0])
    movie2_id = int(item_vecs[min_idx, 0])
    disp.append([movie_dict[movie1_id]['title'], movie_dict[movie1_id]['genres'], movie_dict[movie2_id]['title'], movie_dict[movie2_id]['genres']])

table = tabulate.tabulate(disp, tablefmt='html', headers="firstrow")
table
