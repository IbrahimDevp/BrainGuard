import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint

# Define the paths to your training and testing data directories
train_data_dir = 'dataset/TRAIN'
test_data_dir = 'dataset/TEST'

# Preprocessing steps
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=25,
    zoom_range=0.05,
    horizontal_flip=True,
    vertical_flip=False,
    validation_split=0.8
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

# Set the batch size and image dimensions
batch_size = 32
image_size = (256, 256)

# Flow training images from the directory
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'
)

# Flow validation images from the directory
validation_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'
)

# Define the CNN model
model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(256, 256, 3), padding="same"))
model.add(Conv2D(32,3,activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding="same"))
#model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding="same"))
#model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=SGD(), loss='binary_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)
# Train the model
model.fit(
    train_generator,
    #steps_per_epoch=train_generator.n // batch_size,
    epochs=50,
    #validation_split=0.1 #remove 0.1 from the training to make it validaiton set
    validation_data=validation_generator,
    callbacks=[checkpoint]
    #validation_steps=validation_generator.n // batch_size
)

# Load the best model weights
model.load_weights('best_model.h5')
# Evaluate the model
_, accuracy = model.evaluate(validation_generator)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
# Save the model
model.save('best_model.h5')
print("Model saved successfully.")