from .person_tracker import *
from .model_detector.model_detector import *
from .model_tracker import *
import imutils

class counting_people_detector():
    def __init__(self,model_detector,model_tracking,nbr_frames_tracking):
        self.m_model_detector=model_detector
        self.m_model_tracking=model_tracking
        self.m_nbr_frames_tracking = nbr_frames_tracking
        self.m_total_in= [0]
        self.m_total_out = [0]
        self.m_total_frames = [0]

    def run_model(self,frame):
        #print("[INFO] processing frame...")
        if frame is not None:
            # resize the frame to have a maximum width of 512 pixels,
            # then convert the frame from BGR to RGB
            frame = imutils.resize(frame, width=512)
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            (H_frame, W_frame) = frame.shape[:2]
            self.m_model_tracking.set_size_frame(H_frame, W_frame)
            #print("Height Frame:", H_frame,"Width Frame:",W_frame)
            #Run model in each frame
            self.m_model_tracking.run_model(frame, self.m_total_frames, self.m_total_in,self.m_total_out,self.m_nbr_frames_tracking)
            return frame
        return None








