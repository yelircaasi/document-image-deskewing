import numpy as np
import tensorflow as tf


input_shape = (4, 2339, 1)
batch_size = 4
n_pixel_rows = 2339


inp = tf.random.normal(input_shape)
#l1 = tf.keras.layers.Conv1D(16, 50, strides=5, activation="relu", input_shape=input_shape[1:])
l1 = tf.keras.layers.Conv1D(16, 50, strides=5, activation="relu", input_shape=(batch_size, n_pixel_rows, 1))
l2 = tf.keras.layers.MaxPooling1D(4)
l3 = tf.keras.layers.Conv1D(8, 5, strides=3, activation="relu")#, input_shape=(114, 16))
l4 = tf.keras.layers.MaxPooling1D(4)
l5 = tf.keras.layers.Flatten()
l6 = tf.keras.layers.Dense(1)
output = l1(inp)
print(output.shape)
output = l2(output)
print(output.shape)
output = l3(output)
print(output.shape)
output = l4(output)
print(output.shape)
output = l5(output)
print(output.shape)
output = l6(output)
print(output.shape)

model = tf.keras.Sequential()
model.add(l1) # -> [4, ]
model.add(l2) # -> [4, ]
model.add(l3) # -> [4, ]
model.add(l4) # -> [4, ]
model.add(l5) # -> [4, ]
model.add(l6) # -> [4, ]
model.build(input_shape=(batch_size, n_pixel_rows, 1))
model.summary()

x = tf.constant([[[2],[3],[5],[1],[-6.6],[7],[9],[2],[4], [2.3], [-2.5], [0.9], [-1.9], 
                  [0], [3.4], [-5], [-3.2], [-0.5]]], dtype=tf.float32)
c1 = tf.keras.layers.Conv1D(2, 4, strides=3, activation='relu')
o = c1(x)
print(x.shape)
print(c1.get_weights()[0].shape)
print(o.shape)
w1 = c1.get_weights()[0]
print(w1)
print(o)

onp = o.numpy()
xnp = x.numpy()
abs(onp[0, 0, 0] - tf.nn.relu(xnp[0, 0:4, 0].dot(w1[:, :, 0])).numpy()[0]) < 0.0001
abs(onp[0, 1, 0] - tf.nn.relu(xnp[0, 3:7, 0].dot(w1[:, :, 0])).numpy()[0]) < 0.0001
abs(onp[0, 2, 0] - tf.nn.relu(xnp[0, 6:10, 0].dot(w1[:, :, 0])).numpy()[0]) < 0.0001
abs(onp[0, 3, 0] - tf.nn.relu(xnp[0, 9:13, 0].dot(w1[:, :, 0])).numpy()[0]) < 0.0001
abs(onp[0, 4, 0] - tf.nn.relu(xnp[0, 12:16, 0].dot(w1[:, :, 0])).numpy()[0]) < 0.0001

abs(onp[0, 0, 1] - tf.nn.relu(xnp[0, 0:4, 0].dot(w1[:, :, 1])).numpy()[0]) < 0.0001
abs(onp[0, 1, 1] - tf.nn.relu(xnp[0, 3:7, 0].dot(w1[:, :, 1])).numpy()[0]) < 0.0001
abs(onp[0, 2, 1] - tf.nn.relu(xnp[0, 6:10, 0].dot(w1[:, :, 1])).numpy()[0]) < 0.0001
abs(onp[0, 3, 1] - tf.nn.relu(xnp[0, 9:13, 0].dot(w1[:, :, 1])).numpy()[0]) < 0.0001
abs(onp[0, 4, 1] - tf.nn.relu(xnp[0, 12:16, 0].dot(w1[:, :, 1])).numpy()[0]) < 0.0001

c2 = tf.keras.layers.Conv1D(7, 3, strides=2)#, activation='relu')
o2 = c2(o)
print(o.shape)
print(o2.shape)
print(c2.get_weights()[0].shape)

onp2 = o2.numpy()
w2 = c2.get_weights()[0]

abs(onp2[0, 0, 0] - np.sum(onp[:, 0:3, :] * w2[:, :, 0])) < 0.0001
abs(onp2[0, 1, 0] - np.sum(onp[:, 2:5, :] * w2[:, :, 0])) < 0.0001
abs(onp2[0, 0, 1] - np.sum(onp[:, 0:3, :] * w2[:, :, 1])) < 0.0001
abs(onp2[0, 1, 1] - np.sum(onp[:, 2:5, :] * w2[:, :, 1])) < 0.0001

