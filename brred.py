import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from pymongo import MongoClient
import io
import base64

# -----------------------------
# MongoDB setup
# -----------------------------
client = MongoClient("mongodb://localhost:27017/")
db = client["breed_images"]
collection = db["BreeddlImagess"]
users_collection = db["users"]  # New collection for users

# -----------------------------
# Load the TFLite model safely
# -----------------------------
try:
    interpreter = tf.lite.Interpreter(model_path="buffalo_cnn_model (1).tflite")
    interpreter.allocate_tensors()
except Exception as e:
    st.error(f"Error loading TFLite model: {e}")
    st.stop()

# -----------------------------
# Class names
# -----------------------------
class_names = [
    'Banni','Guernsey','Jersey','Ongole',
    'Umblachery'
]

# -----------------------------
# Authentication UI
# -----------------------------
def signup():
    st.subheader("Sign Up")
    authority_name = st.text_input("Authority Name")
    designation = st.text_input("Designation")
    user_id = st.text_input("Authority ID")
    password = st.text_input("Set Password", type="password")
    if st.button("Create Account"):
        if authority_name and designation and user_id and password:
            if users_collection.find_one({"user_id": user_id}):
                st.error("User ID already exists.")
            else:
                users_collection.insert_one({
                    "authority_name": authority_name,
                    "designation": designation,
                    "user_id": user_id,
                    "password": password
                })
                st.success("Account created! Please sign in.")
                st.session_state.page = "login"
        else:
            st.error("Please fill all fields.")
    if st.button("Sign In"):
        st.session_state.page = "login"

def login():
    st.subheader("Sign In")
    user_id = st.text_input("Authority ID")
    password = st.text_input("Password", type="password")
    if st.button("Sign In"):
        user = users_collection.find_one({"user_id": user_id, "password": password})
        if user:
            st.session_state.logged_in = True
            st.session_state.user = user
            st.session_state.page = "main"
        else:
            st.error("Invalid ID or password.")

def main_menu():
    st.subheader(f"Welcome, {st.session_state.user['authority_name']} ({st.session_state.user['designation']})")
    if st.button("Breed Checker"):
        st.session_state.page = "breed_checker"

# -----------------------------
# Main App Logic
# -----------------------------
if "page" not in st.session_state:
    st.session_state.page = "login"
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if st.session_state.page == "signup":
    signup()
elif st.session_state.page == "login":
    login()
    st.write("Don't have an account?")
    if st.button("Go to Sign Up"):
        st.session_state.page = "signup"
elif st.session_state.page == "main":
    main_menu()
elif st.session_state.page == "breed_checker":
    st.title("🐄 Indian Bovine Breed Classifier")
    st.write("Upload an image of a cow/buffalo to predict its breed.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded Image', width=600)
            img = image.resize((128,128))  # <-- changed to match model input
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # TFLite prediction
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            preds = interpreter.get_tensor(output_details[0]['index'])[0]
            pred_idx = int(np.argmax(preds))
            confidence = float(preds[pred_idx])
            class_name = class_names[pred_idx]

            st.subheader(f"Predicted Breed: {class_name}")
            st.write(f"Confidence: {confidence*100:.2f}%")
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            doc = {
                "filename": uploaded_file.name,
                "image_data": img_base64,
                "predicted_breed": class_name,
                "confidence": confidence,
                "uploaded_by": st.session_state.user['user_id']
            }
            collection.insert_one(doc)
            st.success("Image and prediction stored in MongoDB!")
        except Exception as e:
            st.error(f"Failed to process the image: {e}")

    st.header("Last Uploaded Image from MongoDB")
    last_doc = collection.find().sort([('_id', -1)]).limit(1)
    for doc in last_doc:
        st.write(f"Filename: {doc['filename']}")
        st.write(f"Predicted Breed: {doc['predicted_breed']}")
        st.write(f"Confidence: {doc['confidence']*100:.2f}%")
        img_bytes = base64.b64decode(doc['image_data'])
        st.image(Image.open(io.BytesIO(img_bytes)), caption='Image from MongoDB', width=400)

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.page = "login"