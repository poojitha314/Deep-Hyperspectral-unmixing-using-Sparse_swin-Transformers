import streamlit as st
import numpy as np
import torch
from PIL import Image
from transformer import SwinAutoEncoder
import utils
import scipy.io as sio

# Title of the app
st.title("Hyperspectral Unmixing Using Swin Transformer")

# Section for uploading the hyperspectral image
uploaded_file = st.file_uploader("Upload Hyperspectral Image", type=["mat", "tiff", "png", "jpg"])

# Helper function to preprocess image data
def preprocess_image(image_array):
    """
    Converts the image array into a PyTorch tensor, resizes, and normalizes.
    Assuming the image_array is a 3D array (height, width, channels).
    """
    image_tensor = torch.tensor(image_array).float()  # Convert to tensor
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # Reorder to (1, channels, height, width)
    return image_tensor

# Function to run the unmixing model
def run_unmixing(image_array, model, num_epochs, learning_rate):
    """
    This function takes the uploaded image, processes it through the model,
    and returns the abundance maps, reconstruction error, and SAD (spectral angle distance).
    """
    # Preprocess the image for model input
    processed_image = preprocess_image(image_array)

    # Run the image through the model
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        abundance_map, reconstruction = model(processed_image)

    # Calculate performance metrics (Reconstruction Error, Spectral Angle Distance)
    re_error = utils.compute_rmse(reconstruction, processed_image)
    sad = utils.compute_sad(abundance_map, processed_image)  # Placeholder for ground truth

    # Return the results as a dictionary
    return {'abundance_maps': abundance_map, 're_error': re_error, 'sad': sad}

# If a file is uploaded, continue
if uploaded_file is not None:
    # Load the image (assuming it's a hyperspectral image in some format)
    image = Image.open(uploaded_file)
    image_array = np.array(image)  # Convert image to numpy array

    # Show the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Model selection
    model_option = st.selectbox("Choose Model", ["SwinAutoEncoder", "CNN", "Linear"])

    # Hyperparameter sliders
    num_epochs = st.slider("Number of Epochs", 10, 1000, 200)
    learning_rate = st.slider("Learning Rate", 1e-5, 1e-2, 1e-3)

    if st.button("Run Unmixing"):
        # Initialize model (using the selected model option)
        if model_option == "SwinAutoEncoder":
            model = SwinAutoEncoder(P=3, L=156, size=95, patch=5, dim=200).to("cuda" if torch.cuda.is_available() else "cpu")
        
        # Run the unmixing process
        result = run_unmixing(image_array, model, num_epochs, learning_rate)

        # Show results
        st.write("Abundance Maps:")
        st.image(result['abundance_maps'], caption="Abundance Maps")  # Display abundance maps
        
        st.write("Reconstruction Error (RE):", result['re_error'])  # Display RE
        st.write("Spectral Angle Distance (SAD):", result['sad'])  # Display SAD

        # Option to download results
        st.download_button("Download Results", data=result['abundance_maps'], file_name="abundance_maps.png")
