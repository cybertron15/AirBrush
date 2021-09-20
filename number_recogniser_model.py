import tensorflow as tf

mnist = tf.keras.datasets.mnist

(img_train,label_train), (img_test,label_test) = mnist.load_data()
#load data method will return two tuples containing training and testing data

#normalizing data
img_train = tf.keras.utils.normalize(img_train,axis=1)
img_test = tf.keras.utils.normalize(img_test,axis=1)

#creating model
model = tf.keras.models.Sequential()

#adding layers
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))

#compiling the model        
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#training the model
model.fit(img_train,label_train, epochs=6)

#saving the model
model.save(r'Number_recognising_model')

#testing the model
#if we want to load the model from the saved file we can use this code
#model = tf.keras.models.load_model(r'Number_recognising_model')
loss, accuracy = model.evaluate(img_test,label_test)
print(loss,accuracy)
