from django.shortcuts import render
from django.http import HttpResponse
from django.http import StreamingHttpResponse

from .counting_people_algorithm.main_webcam import *
from imutils.video import VideoStream
import threading
import time
# Create your views here.
url = "http://192.168.1.86:8080/shot.jpg"

# Model detector/tracker
W_frame = None
H_frame = None
nbr_frames_tracking = 10
methodTracker = "dlib_correlation"
personTracking = personTracker(maxNbrFramesDisappeared=50, maxDistance=50)
modelDetector = model_detector(path_protocol=full_path_protocol_model,
                                   path_weight=full_path_weight_model,
                                   classes=CLASSES_MobileNet_SSD)
modelTracking = modelTracker(model_detector=modelDetector, frame_size=frameSize(W_frame, H_frame),
                             personTracker=personTracking, methodTracker=methodTracker, confidence=0.2)
outputFrame =  None
lock =threading.Lock() # ensure thread-safe when updating outputFrame (one thread isn't trying to read the frame as it is being updated)

vs = VideoStream(src=0).start()
time.sleep(2.0)


class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(0)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')
        self.m_count_people_detector= counting_people_detector(model_detector=modelDetector,model_tracking=modelTracking,nbr_frames_tracking=nbr_frames_tracking)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, frame = self.video.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        if success is True:
            outputFrame = self.m_count_people_detector.run_model(frame)

            if outputFrame is not None:
                # Convert image in JPEG format
                (flag, encodeImg) = cv2.imencode(".jpg",outputFrame)
        return encodeImg.tobytes()


def generate(camera):
    while True:
        frame = camera.get_frame()
        # yield the output frame in the byte format
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               frame + b'\r\n')

def video_feed(request):
    return StreamingHttpResponse(generate(VideoCamera()), content_type='multipart/x-mixed-replace; boundary=frame')

def homepage(request):
    return render(request=request,template_name="main\homepage.html")