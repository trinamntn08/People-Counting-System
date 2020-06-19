from django.shortcuts import render
from django.http import HttpResponse
from django.http import StreamingHttpResponse
from enum import Enum
from .counting_people_algorithm.model_all import *
from .counting_people_algorithm.age_gender_predictor.age_gender_predictor import *
from .counting_people_algorithm.age_gender_predictor.model_face_age_gender_detector import *
from imutils.video import VideoStream

import threading
import time

# MODE INPUT VIDEO
class MODE_INPUT(Enum):
    WEBCAM=0,
    VIDEO=1,
    MOBILE=2,
    RASPI=3

url = "http://192.168.1.86:8080/shot.jpg"

video_path= os.path.join(os.getcwd(),"main\counting_people_algorithm\input\example_01.mp4")

# Config detector/tracker parameters
W_frame = None
H_frame = None
nbr_frames_tracking = 10
method_tracker = "dlib_correlation"
person_tracking = person_tracker(maxNbrFramesDisappeared=50, maxDistance=50)
model_detector = model_detector(path_protocol=full_path_protocol_model,
                                   path_weight=full_path_weight_model,
                                   classes=CLASSES_MobileNet_SSD)

faceDetector= model_face_age_gender_detector(path_protocol=path_protocol_face,
                                             path_weight=path_weight_face).load_model_age_gender_detector()
genderDetector= model_face_age_gender_detector(path_protocol=path_protocol_gender,
                                               path_weight=path_weight_gender).load_model_age_gender_detector()
ageDetector= model_face_age_gender_detector(path_protocol=path_protocol_age,
                                            path_weight=path_weight_age).load_model_age_gender_detector()

age_genderPredictor = detect_face_and_predict_age(frame_size=frame_size(W_frame, H_frame),
                                                  model_face_detector=faceDetector,
                                                  model_age_detector=genderDetector,
                                                  model_gender_detector=ageDetector,confidence=0.2)

model_tracking = model_tracker(model_detector=model_detector, frame_size=frame_size(W_frame, H_frame),
                               person_tracker=person_tracking, age_gender_predictor=age_genderPredictor,
                               method_tracker=method_tracker, confidence=0.2)
outputFrame =  None
lock =threading.Lock() # ensure thread-safe when updating outputFrame
                       # (one thread isn't trying to read the frame as it is being updated)


class VideoCamera(object):
    def __init__(self, mode_input=0):
        self.m_video = cv2.VideoCapture(0,cv2.CAP_DSHOW)
        self.m_count_people_detector= counting_people_detector(model_detector=model_detector,
                                                               model_tracking=model_tracking,
                                                               nbr_frames_tracking=nbr_frames_tracking)

    def set_input_video(self,mode_input=0,path_video=""):
        if(mode_input==MODE_INPUT.WEBCAM):
            self.m_video = cv2.VideoCapture(0)
        elif(mode_input==MODE_INPUT.VIDEO):
            self.m_video=cv2.VideoCapture(path_video)


    def __del__(self):
        self.m_video.release()
        #self.m_video.stop()

    def get_frame(self):
        ret, frame = self.m_video.read()
        if ret is True:
            outputFrame = self.m_count_people_detector.run_model(frame)
            return outputFrame
        return None


def generate(camera):
    global lock
    while True:
        output_frame = camera.get_frame()
        with lock:
            if output_frame is None:
                continue
            processed_frame = output_frame.copy()
            if processed_frame is None:
                continue
            # Convert image in JPG format
            (flag, encodeImg) = cv2.imencode(".jpg", processed_frame)
            # ensure the frame was successfully encoded
            if not flag:
                continue

        # yield the output frame in the byte format
        yield (b'--frame\r\n' b'Content-Type: image/jpg\r\n\r\n' +
               bytearray(encodeImg) + b'\r\n')


def video_feed(request):
    video_camera =  VideoCamera()
    video_camera.set_input_video(mode_input=MODE_INPUT.WEBCAM,path_video=video_path)
    return StreamingHttpResponse(generate(video_camera), content_type='multipart/x-mixed-replace; boundary=frame')

def homepage(request):
    return render(request=request,template_name="main\homepage.html")