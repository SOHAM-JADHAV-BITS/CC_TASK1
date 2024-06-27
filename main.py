import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import PIL
import IPython.display as display
plt.ion()
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train/255.0
x_test = x_test/255.0

print('size of training images = ', x_train.shape)
print('size of testing images = ', x_test.shape)

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

r = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)

plt.figure()
plt.plot(r.history['loss'], label='loss function on training data')
plt.plot(r.history['val_loss'], label='loss function on testing data')
plt.xlabel('epoch')
plt.ylabel('loss function')
plt.legend()
plt.show(block=True)  # Show the plot

plt.figure()
plt.plot(r.history['accuracy'], label='accuracy on training data')
plt.plot(r.history['val_accuracy'], label='accuracy on testing data')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.show(block=True)  # Show the plot

y_pred = model.predict(x_test)
y_pred = tf.math.argmax(y_pred, axis=1)
y_pred = y_pred.numpy()
print(y_pred[0:11])

cm = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(cm).plot()
plt.show(block=True)  # Show the plot

# Load an image and predict the digit in it   " I'm using a .png file here which is automatically resized and converted to grayscale "
img = PIL.Image.open("digit_image.png").convert('L').resize((28, 28))
img = PIL.ImageOps.invert(img)

# using matplotlib to display it
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show(block=True)

img = np.array(img)
img = img / 255.0
img = img.reshape(1, 28, 28)
y_pred_test_img = model.predict(img)
print(y_pred_test_img)
y_pred_test_img = tf.math.argmax(y_pred_test_img, axis=1)
y_pred_test_img = y_pred_test_img.numpy()
print(y_pred_test_img)
