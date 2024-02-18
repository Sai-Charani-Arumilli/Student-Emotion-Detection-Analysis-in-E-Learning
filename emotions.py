import cv2
import numpy as np
from keras.models import load_model # to load harcascade classifier hdf5 format
from statistics import mode
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input
from gtts import gTTS 
import os 
import time 
from datetime import datetime
from plyer import notification 
import matplotlib.pyplot as plt

USE_WEBCAM = True # If false, loads video file source

okx = 0 
ntokx = 0 
okavgx = 0
ntokavgx = 0
face_label = "" 
emotion_text = ""
ok = "" 
emotion_probability = 0
def myfunc():
    if not hasattr(myfunc, "okppl"):
        myfunc.okppl = 0 
    myfunc.okppl +=1 
    return(myfunc.okppl)    
def myfuncnt():
    if not hasattr(myfuncnt, "ntokppl"):
        myfuncnt.ntokppl = 0 
    myfuncnt.ntokppl +=1  
    return(myfuncnt.ntokppl)
# parameters for loading data and images
emotion_model_path = r'C:\Users\hp\Desktop\emotion_detection_model(final)\models\emotion_model.hdf5'
emotion_labels = get_labels('fer2013')

# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

# loading models
face_cascade = cv2.CascadeClassifier(r'C:\Users\hp\Desktop\emotion_detection_model(final)\models\haarcascade_frontalface_default.xml')
emotion_classifier = load_model(emotion_model_path)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# starting lists for calculating modes
emotion_window = []

# starting video streaming 
cv2.namedWindow('window_frame', cv2.WINDOW_NORMAL)
video_capture = cv2.VideoCapture(0)

# Select video or webcam feed
cap = None
if (USE_WEBCAM == True):
    cap = cv2.VideoCapture(0) # Webcam source
else:
    cap = cv2.VideoCapture('./doc/dinner.mp4') # Video file source

while cap.isOpened(): # True:
    ret, bgr_image = cap.read()

    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    # returns rectangle like coordinates
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5,
			minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    for face_coordinates in faces:

        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction) #gives max value
        emotion_label_arg = np.argmax(emotion_prediction) #returns max value's coordinates
        # getting labels for for the matched emotion image
        emotion_text = emotion_labels[emotion_label_arg] 
        face_label = emotion_text + " " + str(emotion_probability) 
        print(face_label) 
        emotion_window.append(emotion_text) 
        # emotion_window.append(face_label) 

        if emotion_text=='angry' or emotion_text=='sad' or emotion_text=='fear': 
            ntokx = myfuncnt() 
        else:
            okx = myfunc() 

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue
 
        if emotion_text == 'angry':
            # red color
            color = emotion_probability * np.asarray((255, 0, 0))
        elif emotion_text == 'sad':
            # blue color
            color = emotion_probability * np.asarray((0, 0, 255))
        elif emotion_text == 'happy':
            # yellow color
            color = emotion_probability * np.asarray((255, 255, 0))
        elif emotion_text == 'surprise':
            # aqua color
            color = emotion_probability * np.asarray((0, 255, 255))
        else:
            # green shade
            color = emotion_probability * np.asarray((0, 255, 0))

        color = color.astype(int)
        color = color.tolist()
        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, emotion_mode,
                  color, 0, -45, 1, 1) 
        
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR) 
    cv2.imshow('window_frame', bgr_image)     
    #if emotion_text=='angry' or emotion_text=='sad' or emotion_text=='fear':
        #if emotion_probability >= 0.0 : 
            #audio_text = "getting more emotional. Please Calm down! " 
            #notification.notify(
            #title='warning',
            #message= audio_text,
            #app_name='application name'
            #) 
            #audio_text1="Don't worry! Be happy!"
            #language = 'en'
            #myobj = gTTS(text=audio_text1, lang=language, slow=False) 
            #now = datetime.now() 
            #year = now.strftime("%Y_") 
            #month = now.strftime("%m_") 
            #day = now.strftime("%d_") 
            #time_c = now.strftime("%H_%M") 
            #ok = str(day) + str(month) + str(year) + str(time_c)  
            #myobj.save(r"C:\Users\hp\Desktop\emotion_detection_model(final)\stataud"+ok + ".mp3")
            #os.system(r"C:\Users\hp\Desktop\emotion_detection_model(final)\stataud"+ok + ".mp3")'''
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()

t=okx+ntokx
okper = (okx/t)*100
ntokper = (ntokx/t)*100
normal= "Interested - " + str(okper) 
ntnormal= "Not interested - " + str(ntokper)
labels = [normal,ntnormal]
sizes = [okper,ntokper]
colors = ['yellowgreen','red']
patches, texts = plt.pie(sizes, colors=colors, shadow=True, startangle=90)
plt.legend(patches, labels, loc="best")
plt.axis('equal')
plt.tight_layout()

  
plt.show()


