import streamlit as st

# Set page config
st.set_page_config(page_title="Smoker Detection", page_icon="🚬", layout="wide")

# Add title and subtitle to the app
st.title("🚭 Smoker Detection App 🚭")
st.markdown("*by Pranya Jain*")
st.subheader("Please upload an image of a person to detect if they are smoking or not")

# Create file uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Perform detection if image is uploaded
if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Detect smoker
    if True: # Replace with your detection algorithm
        st.error("🚬 **Smoking Detected!** 🚬")
    else:
        st.success("😃 **No Smoking Detected!** 😃")
