import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

model = load_model("skin_cancer_model.h5")


def predict_skin_cancer(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224))  # Load Image
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make Prediction
    prediction = model.predict(img_array)
    class_label = "Malignant" if prediction > 0.5 else "Benign"

    # Show Image with Prediction
    plt.imshow(img)
    plt.title(f"Predicted: {class_label}")
    plt.axis("off")
    plt.show()

    return class_label

#UI APP
st.title("Skin cancer detection")
st.markdown("""
This is skin cancer detection application. You can upload an image of affected part and get results.""")
uploaded_img = st.file_uploader("Choose an image",type=['jpg','jpeg','png'])

if uploaded_img is not None:
    class_label = predict_skin_cancer(uploaded_img,model)

    st.image(uploaded_img,caption='Uploaded image',width=500) # for width equal to img-> true
    st.write(f'**Prediction:{class_label}**')

st.markdown("""
### About the model:
This model uses CNN architecture for predicting whether a skin lesion is **Benign** or
**Malignant**.

### Features:
- **Input**: Skin lesion image
- **Output**:**Benign** or **Malignant** classification

### How to use:
1. Upload an image of a skin lesion.
2. The model will predict if it's **Benign** or **Malignant**.
""")