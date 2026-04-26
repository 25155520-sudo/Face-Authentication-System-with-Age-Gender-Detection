import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def extract_face(img_path):

    img = cv2.imread(img_path)

    if img is None:
        return None

    img = cv2.resize(img,(640,480))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4
    )

    if len(faces) == 0:
        return None

    x,y,w,h = faces[0]

    face = gray[y:y+h, x:x+w]

    face = cv2.resize(face,(200,200))

    return face


def compare_faces(face1,face2):

    diff = np.sum((face1.astype("float") - face2.astype("float"))**2)

    score = diff/(200*200)

    return score