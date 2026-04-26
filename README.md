Face Authentication System with Age & Gender Detection

This project is a real-time face authentication system built using OpenCV and Streamlit, with additional features to predict:
Face Matching (Authentication)
Gender Detection
Age Group Estimation

How It Works
1. Face Registration
User captures image via webcam
Image saved as registered_face.jpg
2. Face Verification
User captures another image
System:
Detects face using Haar Cascade
Extracts and resizes face
Compares with registered face
3. Face Matching Logic
Uses pixel-wise difference:
difference = sum((face1 - face2)^2)
score = difference / (image_size)
If:
Score < 2000 → Match (Access Granted)
Score ≥ 2000 → No Match
4. Age & Gender Prediction
Uses pre-trained Caffe models
Input image converted into blob
Predictions:
Gender → Male / Female
Age → Age bucket (e.g., 25–32)

How to Run = 
Step-1= Install dependencies
Step-2= Add model files
Make sure these files are in your project folder:
age_net.caffemodel
age_deploy.prototxt
gender_net.caffemodel
gender_deploy.prototxt
Step-3 = Run the app
Step-4 = Open the browser

Limitations
Face matching is very basic (pixel comparison)
Sensitive to:
Lighting
Face angle
Distance from camera
Not as accurate as deep learning face recognition (FaceNet, etc.)
Only detects one face
