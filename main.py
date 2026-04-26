import streamlit as st
import cv2
import authenticate
import age_gender
import os

st.title("🔐 Face Authentication System")

REGISTER_PATH = "registered_face.jpg"
VERIFY_PATH = "verify_face.jpg"


st.header("Register Face")

register_photo = st.camera_input("Take your face photo")

if register_photo is not None:

    with open(REGISTER_PATH,"wb") as f:
        f.write(register_photo.getbuffer())

    st.success("Face Registered Successfully")


st.header("Verify Face")

verify_photo = st.camera_input("Take photo for verification")

if verify_photo is not None:

    with open(VERIFY_PATH,"wb") as f:
        f.write(verify_photo.getbuffer())

    if not os.path.exists(REGISTER_PATH):

        st.warning("Please register your face first.")

    else:

        face1 = authenticate.extract_face(REGISTER_PATH)
        face2 = authenticate.extract_face(VERIFY_PATH)

        if face1 is None or face2 is None:

            st.error("Face not detected properly. Look straight at the camera.")

        else:

            score = authenticate.compare_faces(face1,face2)

            st.write("Similarity Score:",score)

            if score < 2000:

                st.success("✅ Access Granted")

                img = cv2.imread(VERIFY_PATH)
                img = cv2.resize(img,(227,227))

                gender,age = age_gender.predict_age_gender(img)

                st.write("Gender:",gender)
                st.write("Estimated Age:",age)

            else:

                st.error("❌ Access Denied")