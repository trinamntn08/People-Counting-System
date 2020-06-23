import numpy as np
from .model_face_age_gender_detector import *

AGE_RANGE = ["(0-3)", "(4-7)", "(8-14)", "(15-23)", "(24-35)",
		    "(36-45)", "(46-59)", "(60-100)"]

GENDER_RANGE=["MALE","FEMALE"]


class frame_size:
    def __init__(self, width=500, height=500):
        self.m_width = width
        self.m_height = height


class detect_face_and_predict_age:
    def __init__(self,frame_size,model_face_detector,model_age_detector,model_gender_detector,confidence=0.2):
        self.m_frame_size = frame_size
        self.m_model_face_detector =  model_face_detector
        self.m_model_age_detector = model_age_detector
        self.m_model_gender_detector = model_gender_detector
        self.m_confidence=confidence

    def set_size_frame(self,H_frame, W_frame):
        self.m_frame_size.m_width= W_frame
        self.m_frame_size.m_height = H_frame

    def detect_face(self,frame):
        list_bboxes = []
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                     (104.0, 177.0, 123.0))
        self.m_model_face_detector.setInput(blob)
        detections_bboxes = self.m_model_face_detector.forward()
        for i in range(0, detections_bboxes.shape[2]):
            confidence= detections_bboxes[0,0,i,2]
            if confidence > self.m_confidence:
                b_box= detections_bboxes[0,0,i,3:7] * np.array([self.m_frame_size.m_width,self.m_frame_size.m_height,self.m_frame_size.m_width,self.m_frame_size.m_height])
                (startX,startY, endX, endY)  = b_box.astype('int')
                list_bboxes.append(frame[startY:endY, startX:endX])
        return list_bboxes

    def predict_gender(self,face):
        self.m_model_gender_detector.setInput(face)
        gender_predicts = self.m_model_gender_detector.forward()
        gender = AGE_RANGE[gender_predicts[0].argmax()]
        #print("Gender: {}".format(gender))
        return gender

    def predict_age(self,face):
        self.m_model_age_detector.setInput(face)
        age_predicts = self.m_model_age_detector.forward()
        age = GENDER_RANGE[age_predicts[0].argmax()]
        #print("Age:{}".format(age))
        return age



