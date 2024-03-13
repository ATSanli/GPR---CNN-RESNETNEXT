import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.utils import to_categorical # convert to one-hot-encoding
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, GlobalAveragePooling2D
from keras.applications import ResNet50
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

# Load and preprocess your data as needed
dataType = "Dataset_All"

def getDataWithLabel():
    data = pd.read_pickle(dataType + ".pkl") 
    data = data.sample(frac=1).reset_index(drop=True)
    labels = data['Category']
    data = data.drop(labels='Category', axis=1)
    return data, labels

def labelEncode(label):
    if label == 'pos':
        return 0
    elif label == 'neg':
        return 1

def fitLabelEncoder(labels):
    return np.array([labelEncode(label) for label in labels])

def createTestAndTrain():
    X_train, y_train = getDataWithLabel()
    X_train = X_train / 255.0
    X_train = X_train.values.reshape(-1, 48, 48, 1)
    y_train = fitLabelEncoder(y_train)
    y_train = to_categorical(y_train, num_classes=2)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    return X_train, X_test, y_train, y_test

def create_resnet_model(input_shape=(48, 48, 3)):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(2, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=x)
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

X_train, X_test, y_train, y_test = createTestAndTrain()

# Convert grayscale images to RGB
def convert_grayscale_to_rgb(images):
    rgb_images = np.zeros((images.shape[0], images.shape[1], images.shape[2], 3))
    rgb_images[:, :, :, 0] = images[:, :, :, 0]
    rgb_images[:, :, :, 1] = images[:, :, :, 0]
    rgb_images[:, :, :, 2] = images[:, :, :, 0]
    return rgb_images

X_train_rgb = convert_grayscale_to_rgb(X_train)
X_test_rgb = convert_grayscale_to_rgb(X_test)

model = create_resnet_model(input_shape=(48, 48, 3))

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

# Train the model
history = model.fit(X_train_rgb, y_train, epochs=5, batch_size=32, validation_split=0.1, callbacks=[early_stopping])

# Evaluate the model
score = model.evaluate(X_test_rgb, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Make predictions
y_pred = model.predict(X_test_rgb)
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate accuracy and other metrics
acc = accuracy_score(np.argmax(y_test, axis=1), y_pred_classes)
print('Accuracy:', acc)

# Confusion matrix
confusion_mtx = confusion_matrix(np.argmax(y_test, axis=1), y_pred_classes)
sns.heatmap(confusion_mtx, annot=True, fmt='d')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()
