"""
Michael Patel
March 2021

Project description:
    Use TensorFlow Lite to create an object detection model than runs on an Android device

"""
################################################################################
# Imports
import os
import matplotlib.pyplot as plt

import tensorflow as tf


################################################################################
# Main
if __name__ == "__main__":
    # TF version
    print(tf.__version__)

    # ----- ETL ----- #
    # download dataset
    _URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    zip_file = tf.keras.utils.get_file(origin=_URL,
                                       fname="flower_photos.tgz",
                                       extract=True)
    base_dir = os.path.join(os.path.dirname(zip_file), 'flower_photos')

    # rescale images
    IMAGE_SIZE = 224
    BATCH_SIZE = 32

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2)

    train_generator = datagen.flow_from_directory(
        base_dir,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        subset='training')

    val_generator = datagen.flow_from_directory(
        base_dir,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        subset='validation')

    print(train_generator.class_indices)

    labels = '\n'.join(sorted(train_generator.class_indices.keys()))

    with open('labels.txt', 'w') as f:
        f.write(labels)

    # ----- MODEL ----- #
    IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

    # Create the base model from the pre-trained model MobileNet V2
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SHAPE,
        weights='imagenet',
        include_top=False
    )

    base_model.trainable = False

    # add classification head
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(5, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    print('Number of trainable variables = {}'.format(len(model.trainable_variables)))

    # ----- TRAIN ----- #
    epochs = 10

    history = model.fit(train_generator,
                        steps_per_epoch=len(train_generator),
                        epochs=epochs,
                        validation_data=val_generator,
                        validation_steps=len(val_generator))

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    #plt.show()
    plt.savefig(os.path.join(os.getcwd(), "plots"))

    # ----- DEPLOY ---- #
    """
    # convert model to TF Lite
    saved_model_dir = 'save'
    tf.saved_model.save(model, saved_model_dir)

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()

    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)
    """
