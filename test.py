import tensorflow as tf

# Set parameters
batch_size = 32
img_height = 180
img_width = 180

# Load test dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    "D:/Python/dataset/test-images",
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Normalize test dataset
normalization_layer = tf.keras.layers.Rescaling(1./255)
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

# Prefetch to improve performance
test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# Load the trained model
model = tf.keras.models.load_model('D:/Python/my_model.h5')  # Fixed the path

# Evaluate the model
test_loss, test_acc = model.evaluate(test_ds, verbose=2)
print(f"Test accuracy: {test_acc}")
