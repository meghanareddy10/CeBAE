import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Initialize the CNN
my_model = models.Sequential()

# Step 1 - Convolution
my_model.add(layers.Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
my_model.add(layers.Dropout(0.2))  # Adding dropout for regularization

# Step 2 - Pooling
my_model.add(layers.MaxPooling2D(pool_size=(2, 2)))

# Adding a second convolutional layer
my_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
my_model.add(layers.Dropout(0.3))  # Adding dropout for regularization
my_model.add(layers.MaxPooling2D(pool_size=(2, 2)))

# Adding a third convolutional layer
my_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
my_model.add(layers.Dropout(0.3))  # Adding dropout for regularization
my_model.add(layers.MaxPooling2D(pool_size=(2, 2)))

# Step 3 - Flattening
my_model.add(layers.Flatten())

# Step 4 - Full connection
my_model.add(layers.Dense(128, activation='relu'))
my_model.add(layers.Dropout(0.5))  # Adding dropout for regularization
my_model.add(layers.Dense(1, activation='sigmoid'))

my_model.summary()

# Learning Rate Scheduler
lr_schedule = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

# Compile the CNN
my_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Set up data generators with data augmentation
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   brightness_range=[0.8, 1.2])

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('F:/Meg docs/my prj/ML/classification - celestial/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('F:/Meg docs/my prj/ML/classification - celestial/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

# Fit the model with callbacks
Analysis = my_model.fit(training_set,
                        steps_per_epoch=790 // 32,
                        epochs=50,
                        validation_data=test_set,
                        validation_steps=273 // 32,
                        callbacks=[lr_schedule])

# Save the model
my_model.save('Finalmodel.h5')

# Plot the Loss
plt.plot(Analysis.history['loss'], label='loss')
plt.plot(Analysis.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# Plot the Accuracy
plt.plot(Analysis.history['accuracy'], label='acc')
plt.plot(Analysis.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()
