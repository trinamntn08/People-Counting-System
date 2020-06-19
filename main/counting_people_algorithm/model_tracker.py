from .list_persons_detected import *
from .age_gender_predictor.age_gender_predictor import *
import cv2
import dlib
import numpy as np
from enum import Enum
import datetime

OBJECT_TRACKERS = {
		"csrt": cv2.TrackerCSRT_create,
		"kcf": cv2.TrackerKCF_create,
		"boosting": cv2.TrackerBoosting_create,
		"mil": cv2.TrackerMIL_create,
		"tld": cv2.TrackerTLD_create,
		"medianflow": cv2.TrackerMedianFlow_create,
		"mosse": cv2.TrackerMOSSE_create,
        "dlib_correlation":dlib.correlation_tracker
	}

class frame_size:
    def __init__(self, width=500, height=500):
        self.m_width = width
        self.m_height = height

class Status(Enum):
    WAITING=0
    DETECTING=1
    TRACKING=2

status=Status(0)

class model_tracker:
    def __init__(self,frame_size,model_detector,person_tracker,age_gender_predictor, method_tracker="dlib_correlation", confidence=0.2):
        self.m_frame_size=frame_size
        self.m_model=model_detector
        self.m_model_detector= model_detector.load_model_detector()
        self.m_person_tracker= person_tracker
        self.m_age_gender_predictor = age_gender_predictor
        self.m_method_tracker=method_tracker
        self.m_confidence=confidence
        self.m_trackers=[] # list to store each of our dlib correlation trackers
        self.m_list_persons_detected={} # dictionnary to map each unique personID to a m_list_persons_detected

    def set_size_frame(self,H_frame, W_frame):
        self.m_frame_size.m_width= W_frame
        self.m_frame_size.m_height = H_frame
        self.m_age_gender_predictor.set_size_frame(H_frame, W_frame)

    def run_model(self,frame,total_frames,total_In,total_Out, nbr_frames_tracking=10):
        pos = []
        success = False
        status = Status.WAITING
        # list of bounding boxes for each frame in tracking phase
        list_bounding_boxes = []

        # DETECION PHASE
        # Run model detection once every "nbr_frames_tracking" frame
        if total_frames[0] % nbr_frames_tracking==0:
            #print("[INFO] DETECTION PHASE")
            status = Status.DETECTING
            self.m_trackers = []

            preprocessed_frame= cv2.dnn.blobFromImage(frame,0.007843,(self.m_frame_size.m_width,self.m_frame_size.m_height),127.5)
            self.m_model_detector.setInput(preprocessed_frame)
            predictions=self.m_model_detector.forward()

            #loop for every b_box detected
            for i in range(0,predictions.shape[2]):
                # extract the confidence associated with the prediction
                confidence = predictions[0, 0, i, 2]

                if confidence > self.m_confidence:
                    idx=int(predictions[0,0,i,1])
                    #print('class: ', CLASSES[idx], ' Confidence: ', confidence)
                    if self.m_model.get_classes()[idx] != "person":
                        continue
                    # get (x,y) coordinate of the bounding box for person
                    b_box= predictions[0,0,i,3:7]*np.array([self.m_frame_size.m_width,self.m_frame_size.m_height,self.m_frame_size.m_width,self.m_frame_size.m_height])
                    (startX, startY, endX, endY) = b_box.astype("int")
                    list_bounding_boxes.append((startX, startY, endX, endY))

                    #print('list bounding boxes: ',list_bounding_boxes)

                    # For dlib correlation tracker
                    if  self.m_method_tracker == "dlib_correlation":
                        tracker = OBJECT_TRACKERS[self.m_method_tracker]()
                        rect = dlib.rectangle(startX, startY, endX, endY)
                        tracker.start_track(frame, rect)
                    else:
                        #For opencv tracker algorithm
                        tracker = OBJECT_TRACKERS[self.m_method_tracker]()
                        pos = (startX, startY, endX-startX, endY-startY) # top-left-x,top-left-y,w,h
                        success = tracker.init(frame,pos)
                        #print("success: ",success,"position initiale: ",pos)

                    # add the tracker to our list of trackers to
                    # utilize it during skip frames
                    self.m_trackers.append(tracker)

        # TRACKING PHASE
        else:
            #print("[INFO] TRACKING PHASE")
            for tracker in self.m_trackers:
                status=Status.TRACKING

                if self.m_method_tracker == "dlib_correlation":
                    tracker.update(frame)
                    pos=tracker.get_position()
                    startX = int(pos.left())
                    startY = int(pos.top())
                    endX = int(pos.right())
                    endY = int(pos.bottom())
                else:
                    success, pos = tracker.update(frame)
                    startX = int(pos[0])
                    startY = int(pos[1])
                    endX = int(pos[0]+pos[2])
                    endY = int(pos[1]+pos[3])
                list_bounding_boxes.append((startX,startY,endX,endY))
                #print('list bounding boxes: ', list_bounding_boxes)

                # GENDER - AGE - PREDICTION
                try:
                    face= frame[startY:endY, startX:endX]
                    if face is None:
                        continue
                    faceBlob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                                     (78.4263377603, 87.7689143744, 114.895847746),
                                                     swapRB=False)
                    gender = self.m_age_gender_predictor.predict_gender(faceBlob)
                    age = self.m_age_gender_predictor.predict_age(faceBlob)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    label = "{}, {}".format(age, gender)
                    cv2.putText(frame, label, (startX, startY - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                except Exception as e:
                    print(str(e))

        # CHECK GO_IN/OUT
        # draw a horizontal line in the center of the frame
        #cv2.line(frame, (0, self.m_frame_size.m_height // 2), (self.m_frame_size.m_width, self.m_frame_size.m_height // 2), (0, 255, 255), 3)

        list_person_tracker =self.m_person_tracker.update(list_bounding_boxes)
        for (personID, personCentroid) in list_person_tracker.items():
            #print("person ID: ",personID)
            personID_exist = self.m_list_persons_detected.get(personID,None)
            # if there is no existing listPersonDetected, create one
            if personID_exist is None:
                personID_exist = list_persons_detected(personID,personCentroid)
            else:
                # the difference between the y-coordinate of the *current*
                # centroid and the mean of *previous* centroids will tell
                # us in which direction the object is moving (negative for
                # 'up' and positive for 'down')
                y=[c[1] for c in personID_exist.m_centroids]
                personID_exist.m_centroids.append(personCentroid)
                # direction = personCentroid[1]-np.mean(y)
                direction = personCentroid[1] - personID_exist.m_centroids[0][1]

                #check if person has been counted
                if not personID_exist.counted:
                    #IN
                    if personID_exist.m_centroids[0][1] < self.m_frame_size.m_height//2 and personCentroid[1] > self.m_frame_size.m_height // 2 :
                        #print("person pos: ", personID_exist.m_centroids)
                        #print("direction: ", direction)
                        total_In[0] += 1
                        #print('Total In: ',total_In[0])
                        personID_exist.counted = True
                    #OUT
                    elif personID_exist.m_centroids[0][1] > self.m_frame_size.m_height//2 and personCentroid[1] < self.m_frame_size.m_height // 2:
                        #print("person pos: ", personID_exist.m_centroids)
                        #print("direction: ", direction)
                        total_Out[0] += 1
                        #print('Total Out: ', total_Out[0])
                        personID_exist.counted = True

            self.m_list_persons_detected[personID]=personID_exist
            #print("Total people ID: ", len(self.m_list_persons_detected))


            # Display personID and the centroid of the object
            text = "ID {}".format(personID)
            cv2.putText(frame, text, (personCentroid[0] - 10, personCentroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (personCentroid[0], personCentroid[1]), 4, (0, 255, 0), -1)

        # #Display total count
        info = [
            ("Out", total_Out[0]),
            ("In", total_In[0]),
        ]
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, self.m_frame_size.m_height - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # grab the current timestamp and draw it on the frame
        timestamp = datetime.datetime.now()
        cv2.putText(frame, timestamp.strftime(
            "%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255,0), 1)

        # Display state of model
        if status == Status.DETECTING:
            str_status = "DETECTING"
        elif status == Status.TRACKING:
            str_status = "TRACKING"
        elif status == Status.WAITING:
            str_status = "WAITING"

        cv2.putText(frame, str_status,
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        total_frames[0] +=1
        #print("Total frames: ",total_frames)
