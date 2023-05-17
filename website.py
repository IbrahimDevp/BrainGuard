import streamlit as st
import numpy as np
from keras.models import load_model
from keras.utils import load_img, img_to_array
# Load the trained model
model = load_model('brain_tumor_classifier.h5')

# Define the class labels
class_labels = ['Non-Tumor', 'Tumor']

# Create a function to preprocess the image
def preprocess_image(image):
    # Resize the image to 256x256 pixels
    image = image.resize((256, 256))
    # Convert the image to an array
    image = img_to_array(image)
    # Normalize the image
    image = image / 255.0
    # Expand the dimensions to match the model input shape
    image = np.expand_dims(image, axis=0)
    return image

# Create the Streamlit web application
def main():
    st.title("Brain Tumor Classification")
    st.text("Upload an MRI image for tumor classification")

    # Upload the image file
    uploaded_file = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image file
        image = load_img(uploaded_file, grayscale=False, target_size=(256, 256, 3), color_mode="rgb")
        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        # Perform prediction
        prediction = model.predict(preprocessed_image)
        # Get the predicted class label
        predicted_label = class_labels[1 if prediction[0][0] > 0.5 else 0]
        # Display the image and prediction result
        #print(round(prediction[0][0],2))
        st.image(image, width=100)
        if prediction[0][0] > 0.5:
            st.error(f"Prediction: {predicted_label} ({100*prediction[0][0]:.2f}%)")
        else:
            st.success(f"Prediction: {predicted_label} ({100-100*prediction[0][0]:.2f}%)")

# Run the web application
if __name__ == "__main__":
    main()