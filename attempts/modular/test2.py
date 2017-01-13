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

    imgFtr.applyFilter1()
    filter1 = imgFtr.diag4[:,:,0]

    # assemble the screen
    diagScreen = np.zeros((1080, 1920, 3), dtype=np.uint8)
    diagScreen[0:540, 0:960] = cv2.resize(imgFtr.diag1.astype(np.uint8), (960,540), interpolation=cv2.INTER_AREA) 
    diagScreen[540:1080, 0:960] = cv2.resize(imgFtr.diag2.astype(np.uint8), (960,540), interpolation=cv2.INTER_AREA) 
    diagScreen[0:540, 960:1920] = cv2.resize(imgFtr.diag3.astype(np.uint8), (960,540), interpolation=cv2.INTER_AREA) 

    imgFtr.applyFilter2()
    filter2 = imgFtr.diag4[:,:,0]
    diag4 = np.dstack((filter1, filter2, filter2))*4

    diagScreen[540:1080, 960:1920] = cv2.resize(diag4.astype(np.uint8)*4, (960,540), interpolation=cv2.INTER_AREA) 
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

videoout = 'project_video_diagnostics_test_filter3.mp4'
clip1 = VideoFileClip("project_video.mp4")
video_clip = clip1.fl_image(process_diagnostic_image) #NOTE: this function expects color images!!
video_clip.write_videofile(videoout, audio=False)

