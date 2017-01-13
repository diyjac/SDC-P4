#importing some useful packages
import numpy as np
import cv2
from p4lib.cameraCal import CameraCal
from p4lib.imageFilters import ImageFilters
from p4lib.projectionManager import ProjectionManager
from p4lib.roadManager import RoadManager
from p4lib.diagManager import DiagManager
from moviepy.editor import VideoFileClip


def diagnostic_process_image(img, roadMgr, diagMgr):
    # Run the functions
    roadMgr.findLanes(img)
    diagScreen = diagMgr.fullDiag()
    diagScreen = diagMgr.textOverlay(diagScreen,0)
    return diagScreen

def process_diagnostic_image(image):
    global roadMgr
    global diagMgr
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image with lines are drawn on lanes)
    result = diagnostic_process_image(image, roadMgr, diagMgr)
    return result

camCal = CameraCal('camera_cal', 'camera_cal/calibrationdata.p')
roadMgr = RoadManager(camCal, debug=True)
diagMgr = DiagManager(roadMgr)

videoout = 'project_video_diagnostics_test_filter29.mp4'
clip1 = VideoFileClip("challenge_video.mp4")
video_clip = clip1.fl_image(process_diagnostic_image) #NOTE: this function expects color images!!
video_clip.write_videofile(videoout, audio=False)

