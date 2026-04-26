import cv2

age_model = cv2.dnn.readNet(
    "age_net.caffemodel",
    "age_deploy.prototxt"
)

gender_model = cv2.dnn.readNet(
    "gender_net.caffemodel",
    "gender_deploy.prototxt"
)

AGE_BUCKETS = [
    '(0-2)','(4-6)','(8-12)','(15-20)',
    '(25-32)','(38-43)','(48-53)','(60-100)'
]

GENDER_LIST = ['Male','Female']


def predict_age_gender(face_img):

    blob = cv2.dnn.blobFromImage(
        face_img,
        1.0,
        (227,227),
        (78.4263377603,87.7689143744,114.895847746),
        swapRB=False
    )

    gender_model.setInput(blob)
    gender_preds = gender_model.forward()
    gender = GENDER_LIST[gender_preds[0].argmax()]

    age_model.setInput(blob)
    age_preds = age_model.forward()
    age = AGE_BUCKETS[age_preds[0].argmax()]

    return gender,age