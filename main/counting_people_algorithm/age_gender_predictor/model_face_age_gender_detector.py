import cv2
import os

path_protocol_face = "main\counting_people_algorithm\\age_gender_predictor\\face_detector\\deploy.prototxt"
path_weight_face="main\counting_people_algorithm\\age_gender_predictor\\face_detector\\res10_300x300_ssd_iter_140000.caffemodel"

path_protocol_age = "main\counting_people_algorithm\\age_gender_predictor\\age_detector\\age_deploy.prototxt"
path_weight_age="main\counting_people_algorithm\\age_gender_predictor\\age_detector\\age_net.caffemodel"

path_protocol_gender = "main\counting_people_algorithm\\age_gender_predictor\\gender_detector\\deploy.prototxt"
path_weight_gender="main\counting_people_algorithm\\age_gender_predictor\\gender_detector\\gender_net.caffemodel"

full_path_protocol_model_face = os.path.join(os.getcwd(),path_protocol_face)
full_path_weight_model_face = os.path.join(os.getcwd(),path_weight_face)

full_path_protocol_model_age = os.path.join(os.getcwd(),path_protocol_age)
full_path_weight_model_age = os.path.join(os.getcwd(),path_weight_age)

full_path_protocol_model_gender = os.path.join(os.getcwd(),path_protocol_gender)
full_path_weight_model_gender = os.path.join(os.getcwd(),path_weight_gender)

class model_face_age_gender_detector:
    def __init__(self,path_protocol,path_weight,classes=""):
        self.m_path_protocol_model=path_protocol
        self.m_path_weight_model=path_weight
        self.m_classes=classes

    def load_model_age_gender_detector(self):
        model = cv2.dnn.readNetFromCaffe(os.path.join(os.getcwd(),self.m_path_protocol_model),
                                         os.path.join(os.getcwd(),self.m_path_weight_model))
        return model

    def get_classes(self):
        return self.m_classes