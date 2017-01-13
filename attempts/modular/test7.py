#importing some useful packages
import numpy as np
import cv2
from p4lib.cameraCal import CameraCal
from p4lib.imageFilters import ImageFilters
from moviepy.editor import VideoFileClip


def diagnostic_process_image(img, camCal):
    imgFtr = ImageFilters(camCal, debug=True)

    # Run the functions
    imgFtr.imageQ(img)
    imgFtr.applyFilter4()
    imgFtr.horizonDetect(debug=True)

    # assemble the screen
    diagScreen = np.zeros((1080, 1920, 3), dtype=np.uint8)
    diagScreen[0:540, 0:960] = cv2.resize(imgFtr.diag1.astype(np.uint8), (960,540), interpolation=cv2.INTER_AREA) 
    diagScreen[540:1080, 0:960] = cv2.resize(imgFtr.diag2.astype(np.uint8), (960,540), interpolation=cv2.INTER_AREA) 
    diagScreen[0:540, 960:1920] = cv2.resize(imgFtr.diag3.astype(np.uint8), (960,540), interpolation=cv2.INTER_AREA) 
    diagScreen[540:1080, 960:1920] = cv2.resize(imgFtr.diag4.astype(np.uint8), (960,540), interpolation=cv2.INTER_AREA) 

    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(diagScreen, imgFtr.skyText, (30, 60), font, 1, (255,255,255), 2)
    cv2.putText(diagScreen, imgFtr.skyImageQ, (30, 90), font, 1, (255,255,255), 2)
    cv2.putText(diagScreen, imgFtr.roadImageQ, (30, 120), font, 1, (255,255,255), 2)
    cv2.putText(diagScreen, 'Road Balance: %f'%(imgFtr.roadbalance), (30, 150), font, 1, (255,255,255), 2)
    if imgFtr.horizonFound:
        cv2.putText(diagScreen, 'Road Horizon: %d'%(imgFtr.roadhorizon), (30, 180), font, 1, (255,255,255), 2)
    else:
        cv2.putText(diagScreen, 'Road Horizon: NOT FOUND!', (30, 180), font, 1, (255,255,0), 2)

    return diagScreen

def process_diagnostic_image(image):
    global combined
    global camCal
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image with lines are drawn on lanes)
    result = diagnostic_process_image(image, camCal)
    return result

camCal = CameraCal('camera_cal', 'camera_cal/calibrationdata.p')
mtx, dist, imgSize = camCal.get()

videoout = 'project_video_diagnostics_test_filter9.mp4'
clip1 = VideoFileClip("harder_challenge_video.mp4")
video_clip = clip1.fl_image(process_diagnostic_image) #NOTE: this function expects color images!!
video_clip.write_videofile(videoout, audio=False)

