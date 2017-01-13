#importing some useful packages
import numpy as np
import cv2
from p4lib.cameraCal import CameraCal
from p4lib.imageFilters import ImageFilters
from p4lib.projectionManager import ProjectionManager
from p4lib.roadManager import RoadManager
from p4lib.diagManager import DiagManager
from moviepy.editor import VideoFileClip

def process_road_image(img, roadMgr):
    # Run the functions
    roadMgr.findLanes(img)
    roadMgr.drawLaneStats()
    return roadMgr.final

def process_image(image):
    global roadMgr
    global diagMgr
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image with lines are drawn on lanes)
    result = process_road_image(image, roadMgr)
    return result

camCal = CameraCal('camera_cal', 'camera_cal/calibrationdata.p')
roadMgr = RoadManager(camCal)

videoout = 'project_video_diagnostics_test_filter28.mp4'
clip1 = VideoFileClip("project_video.mp4")
video_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
video_clip.write_videofile(videoout, audio=False)

