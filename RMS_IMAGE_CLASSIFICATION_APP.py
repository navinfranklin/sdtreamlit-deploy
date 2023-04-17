import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Load the saved model
modelDense = tf.keras.models.load_model('D:\\Data science 2023\\models\\DenseNet121_loss_non_loss.h5')


# Define the test data generator
test_datagen = ImageDataGenerator(rescale=1./255)

# Load the test data from a directory
test_generator = test_datagen.flow_from_directory(
        'D:\\Data science 2023\\201000\\validation',
        target_size=(256, 256),
        batch_size=32,
        class_mode='categorical',
        shuffle=False)

# Define a function to make predictions on user uploaded images
def predict(image_path, model):
    # Load the image and resize it to (224, 224)
    image = load_img(image_path, target_size=(256, 256))

    # Convert the image to a NumPy array and scale the pixel values to be between 0 and 1
    image = img_to_array(image) / 255.0

    # Add a batch dimension to the image
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

    # Use the model to generate a prediction for the image
    pred = model.predict(image)

    # Get the predicted class index
    predicted_class_index = pred.argmax(axis=-1)[0]

    # Get the predicted class name
    class_indices = test_generator.class_indices
    class_names = {v: k for k, v in class_indices.items()}
    predicted_class_name = class_names[predicted_class_index]

    # Return the predicted class name
    return predicted_class_name

# Create a Streamlit app
def app():
    st.title('Image Classification App')
    
    # Allow the user to upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = plt.imread(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Make a prediction on the uploaded image
        predicted_class_name = predict(uploaded_file, modelDense)
        st.write("Prediction:", predicted_class_name)
        
# Run the Streamlit app
if __name__ == '__main__':
    app()
