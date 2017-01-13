#!/usr/bin/python
"""
roadManager.py: version 0.1.0

History:
2017/01/07: Initial version converted to a class
"""

import numpy as np
import cv2
import math
from p4lib.cameraCal import CameraCal
from p4lib.imageFilters import ImageFilters
from p4lib.projectionManager import ProjectionManager
from p4lib.line import Line

class RoadManager():
    # Initialize lineManager
    def __init__(self, camCal, keepN=10, debug=False):
        # for both left and right lines
        # set debugging
        self.debug = debug

        # frameNumber
        self.curFrame = None

        # keep last N
        self.keepN = keepN

        # our own copy of the camera calibration results
        self.mtx, self.dist, self.img_size = camCal.get()

        # normal image size
        self.x, self.y = self.img_size

        # mid point
        self.mid = int(self.y/2)

        # create our own projection manager
        self.projMgr = ProjectionManager(camCal, keepN=keepN, debug=debug)

        # default left-right lane masking
        self.maskDelta = 5

        # road statistics
        # the left and right lanes curvature measurement could be misleading - need a threshold to indicate straight road.
        self.roadStraight = False
        # radius of curvature of the line in meters
        self.radiusOfCurvature = None

        #distance in meters of vehicle center is off from road center
        self.lineBasePos = None

        # left lines only
        # left lane identifier
        self.left = 1

        # ghosting of left lane (for use in trouble spots - i.e: bridge or in harder challenges)
        self.lastNLEdges = None

        # left lane line class
        self.leftLane = Line(self.left, self.x, self.y, self.maskDelta)

        # left lane stats
        self.leftLaneLastTop = None

        # right lines only
        # right lane identifier
        self.right = 2

        # ghosting of right lane (for use in trouble spots - i.e: bridge or in harder challenges)
        self.lastNREdges = None

        # right lane line class
        self.rightLane = Line(self.right, self.x, self.y, self.maskDelta)

        # right lane stats
        self.rightLaneLastTop = None

        # road overhead and unwarped views
        self.roadsurface = np.zeros((self.y, self.x, 3), dtype=np.uint8)
        self.roadunwarped = None
        
        # number of points fitted
        self.leftLanePoints = 0
        self.rightLanePoints = 0

        # cloudy mode
        self.cloudyMode = False

        # pixel offset from direction of travel
        self.lastLeftRightOffset = 0

        # boosting
        self.boosting = 0.0

        # resulting image
        self.final = None

        # for debugging only
        if self.debug:
            self.diag1 = np.zeros((self.y, self.x, 3), dtype=np.float32)


    # function to find starting lane line positions
    # return left and right column positions
    def find_lane_locations(self, projected_masked_lines):
        height = projected_masked_lines.shape[0]
        width = projected_masked_lines.shape[1]
        lefthistogram = np.sum(projected_masked_lines[int(height/2):height,0:int(width/2)], axis=0).astype(np.float32)
        righthistogram = np.sum(projected_masked_lines[int(height/2):height,int(width/2):width], axis=0).astype(np.float32)
        leftpos = np.argmax(lefthistogram)
        rightpos =  np.argmax(righthistogram)+int(width/2)
        # print("leftpos",leftpos,"rightpos",rightpos)
        return leftpos, rightpos, rightpos-leftpos

    def findLanes(self, img):
        if self.curFrame is None:
            self.curFrame = 0
        else:
            self.curFrame += 1

        self.curImgFtr = ImageFilters(self.projMgr.camCal, debug=True)
        self.curImgFtr.imageQ(img)

        #Experimental
        #if self.curImgFtr.visibility < -30 or \
        #   self.curImgFtr.skyImageQ == 'Sky Image: overexposed' or \
        #   self.curImgFtr.skyImageQ == 'Sky Image: underexposed':
        #    self.curImgFtr.balanceEx()

        # detected cloudy condition!
        if self.curImgFtr.skyText == 'Sky Condition: cloudy' and self.curFrame == 0:
           self.cloudyMode = True

        # choose a default filter based on weather condition
        # line class can update filter based on what it wants too (different for each lane line).
        if self.cloudyMode:
            self.curImgFtr.applyFilter3()
            self.maskDelta = 20
            self.leftLane.setMaskDelta(self.maskDelta)
            self.rightLane.setMaskDelta(self.maskDelta)
        elif  self.curImgFtr.skyText == 'Sky Condition: clear' or \
              self.curImgFtr.skyText == 'Sky Condition: tree shaded':
            self.curImgFtr.applyFilter2()
        elif self.curFrame < 2:
            self.curImgFtr.applyFilter4()
        else:
            self.curImgFtr.applyFilter5()

        self.curImgFtr.horizonDetect(debug=True)

        # low confidence?
        if self.leftLane.confidence < 0.5 or self.rightLane.confidence < 0.5 or self.curFrame < 2:
            self.projMgr.findInitialRoadCorners(self.curImgFtr)
            self.initialGradient = self.projMgr.curGradient

            # Use visibility to lower the FoV
            # adjust source for perspective projection accordingly
            if self.curImgFtr.horizonFound:
                self.roadHorizonGap = self.projMgr.curGradient - self.curImgFtr.roadhorizon
                newTop = self.curImgFtr.roadhorizon+self.roadHorizonGap
                self.projMgr.setSrcTop(newTop-self.curImgFtr.visibility, self.curImgFtr.visibility)

            self.lastNREdges = self.curImgFtr.curRoadEdge
            self.lastNLEdges = self.curImgFtr.curRoadEdge
            masked_edges = self.curImgFtr.getProjection()
            # print("masked_edges: ", masked_edges.shape)
            masked_edge = masked_edges[:,:,1]
            leftpos, rightpos, distance = self.find_lane_locations(masked_edge)
            self.leftLane.setBasePos(leftpos)
            self.leftLane.find_lane_lines_points(masked_edge)
            self.rightLane.setBasePos(rightpos)
            self.rightLane.find_lane_lines_points(masked_edge)
            self.leftLane.fitpoly()
            leftprojection = self.leftLane.applyLineMask(self.curImgFtr.getProjection(self.leftLane.side))
            self.leftLane.radius_in_meters(distance)
            self.leftLane.meters_from_center_of_vehicle(distance)

            self.rightLane.fitpoly()
            rightprojection = self.rightLane.applyLineMask(self.curImgFtr.getProjection(self.rightLane.side))
            self.rightLane.radius_in_meters(distance)
            self.rightLane.meters_from_center_of_vehicle(distance)

        else:
            # Apply Boosting...
            # For Challenges ONLY
            if self.cloudyMode:
                if self.curImgFtr.skyText == 'Sky Condition: cloudy':
                    self.boosting = 0.4
                    self.lastNREdges = self.curImgFtr.miximg(self.curImgFtr.curRoadEdge, self.lastNREdges, 1.0, 0.4)
                else:
                    self.boosting = 1.0
                    self.lastNREdges = self.curImgFtr.miximg(self.curImgFtr.curRoadEdge, self.lastNREdges, 1.0, 1.0)
                self.curImgFtr.curRoadEdge = self.lastNREdges
            elif self.curImgFtr.skyText == 'Sky Condition: surrounded by trees':
                self.boosting = 0.0
                #self.lastNREdges = self.curImgFtr.miximg(self.curImgFtr.curRoadEdge, self.lastNREdges, 1.0, 0.4)
                #self.curImgFtr.curRoadEdge = self.lastNREdges

            # project the new frame to a plane for further analysis.
            self.projMgr.project(self.curImgFtr, self.lastLeftRightOffset)

            # find approximate left right positions and distance apart
            masked_edges = self.curImgFtr.getProjection()
            masked_edge = masked_edges[:,:,1]

            # Left Lane Projection setup
            leftprojection = self.leftLane.applyLineMask(self.curImgFtr.getProjection(self.leftLane.side))
            leftPoints = np.nonzero(leftprojection)
            #leftPoints = cv2.findNonZero(leftprojection)
            self.leftLane.allX = leftPoints[1]
            self.leftLane.allY = leftPoints[0]
            self.leftLane.fitpoly2()

            # Right Lane Projection setup
            rightprojection = self.rightLane.applyLineMask(self.curImgFtr.getProjection(self.rightLane.side))
            #rightPoints = np.transpose(np.nonzero(rightprojection))
            rightPoints = np.nonzero(rightprojection)
            #rightPoints = cv2.findNonZero(rightprojection)
            self.rightLane.allX = rightPoints[1]
            self.rightLane.allY = rightPoints[0]
            self.rightLane.fitpoly2()

            # take and calculate some measurements
            distance = self.rightLane.pixelBasePos - self.leftLane.pixelBasePos
            self.leftLane.radius_in_meters(distance)
            self.leftLane.meters_from_center_of_vehicle(distance)
            self.rightLane.radius_in_meters(distance)
            self.rightLane.meters_from_center_of_vehicle(distance)

            leftTop = self.leftLane.getTopPoint()
            rightTop = self.rightLane.getTopPoint()

            # Attempt to move up the Lane lines if we missed some predictions
            if self.leftLaneLastTop is not None and \
               self.rightLaneLastTop is not None:

                # If we are in the harder challenge, our visibility is obscured,
                # so only do this if we are certain that our visibility is good.
                # i.e.: not in the harder challenge!
                if self.curImgFtr.visibility > -30:
                    # if either lines differs by greater than 50 pixel vertically
                    # we need to request the shorter line to go higher.
                    if abs(self.leftLaneLastTop[1]-self.rightLaneLastTop[1])>50:
                        if self.leftLaneLastTop[1] > self.rightLaneLastTop[1]:
                            self.leftLane.requestTopY(self.rightLaneLastTop[1])
                        else:
                            self.rightLane.requestTopY(self.leftLaneLastTop[1])

                    # if our lane line has fallen to below our threshold, get it to come back up
                    if leftTop is not None and leftTop[1]>self.mid-100:
                        self.leftLane.requestTopY(leftTop[1]-10)
                    if leftTop is not None and leftTop[1]>self.leftLaneLastTop[1]:
                        self.leftLane.requestTopY(leftTop[1]-10)
                    if rightTop is not None and rightTop[1]>self.mid-100:
                        self.rightLane.requestTopY(rightTop[1]-10)
                    if rightTop is not None and rightTop[1]>self.rightLaneLastTop[1]:
                        self.rightLane.requestTopY(rightTop[1]-10)

                # visibility poor...
                # harder challenge... need to be less agressive going back up the lane...
                # let at least 30 frame pass before trying to move forward.
                elif self.curFrame > 30:
                    # if either lines differs by greater than 50 pixel vertically
                    # we need to request the shorter line to go higher.
                    if abs(self.leftLaneLastTop[1]-self.rightLaneLastTop[1])>50:
                        if self.leftLaneLastTop[1] > self.rightLaneLastTop[1] and leftTop is not None:
                            self.leftLane.requestTopY(leftTop[1]-10)
                        elif rightTop is not None:
                            self.rightLane.requestTopY(rightTop[1]-10)

                    # if our lane line has fallen to below our threshold, get it to come back up
                    if leftTop is not None and leftTop[1]>self.mid+100:
                        self.leftLane.requestTopY(leftTop[1]-10)
                    if leftTop is not None and leftTop[1]>self.leftLaneLastTop[1]:
                        self.leftLane.requestTopY(leftTop[1]-10)
                    if rightTop is not None and rightTop[1]>self.mid+100:
                        self.rightLane.requestTopY(rightTop[1]-10)
                    if rightTop is not None and rightTop[1]>self.rightLaneLastTop[1]:
                        self.rightLane.requestTopY(rightTop[1]-10)

            # Experimental
            # Update location for FoV
            #if leftTop is not None and rightTop is not None and self.curImgFtr.visibility < -30:
            #    self.lastLeftRightOffset = int((self.x/2)-(leftTop[0]+rightTop[0])/2)
            #    # print("lastLeftRightOffset: ", self.lastLeftRightOffset)

        # Update Stats and Top points for next frame.
        self.leftLaneLastTop = self.leftLane.getTopPoint()
        self.rightLaneLastTop = self.rightLane.getTopPoint()
        self.leftLanePoints = len(self.leftLane.allX)
        self.rightLanePoints = len(self.rightLane.allX)

        # Update road statistics for display
        self.lineBasePos = (self.leftLane.lineBasePos + self.rightLane.lineBasePos)
        if self.leftLane.radiusOfCurvature > 0.0 and self.rightLane.radiusOfCurvature > 0.0:
            self.radiusOfCurvature = (self.leftLane.radiusOfCurvature + self.rightLane.radiusOfCurvature)/2.0
            if self.leftLane.radiusOfCurvature > 3000.0:
                self.roadStraight = True
            elif self.rightLane.radiusOfCurvature > 3000.0:
                self.roadStraight = True
            else:
                self.roadStraight = False
        elif self.leftLane.radiusOfCurvature < 0.0 and self.rightLane.radiusOfCurvature < 0.0:
            self.radiusOfCurvature = (self.leftLane.radiusOfCurvature + self.rightLane.radiusOfCurvature)/2.0
            if self.leftLane.radiusOfCurvature < -3000.0:
                self.roadStraight = True
            elif self.rightLane.radiusOfCurvature < -3000.0:
                self.roadStraight = True
            else:
                self.roadStraight = False
        else:
            self.roadStraight = True

        # Experimental
        # adjust source for perspective projection accordingly
        # attempt to dampen bounce
        #if self.leftLaneLastTop is not None and self.rightLaneLastTop is not None:
        #    x = int((self.x-(self.leftLaneLastTop[0]+self.rightLaneLastTop[0]))/4)
        #    self.projMgr.setSrcTopX(x)

        # create road mask polygon for reprojection back onto perspective view.
        roadpoly = np.concatenate((self.rightLane.XYPolyline, self.leftLane.XYPolyline[::-1]), axis=0)
        roadmask = np.zeros((self.y, self.x), dtype=np.uint8)
        cv2.fillConvexPoly(roadmask, roadpoly, 64)
        self.roadsurface[:,:,0] = self.curImgFtr.miximg(leftprojection,self.leftLane.linemask, 0.5, 0.3)
        self.roadsurface[:,:,1] = roadmask
        self.roadsurface[:,:,2] = self.curImgFtr.miximg(rightprojection,self.rightLane.linemask, 0.5, 0.3)
        
        # unwarp the roadsurface 
        self.roadunwarped = self.projMgr.curUnWarp(self.curImgFtr, self.roadsurface)

        # create the final image
        self.final = self.curImgFtr.miximg(self.curImgFtr.curImage, self.roadunwarped, 0.95, 0.75)

        # draw dots and polyline
        if self.debug:
            font = cv2.FONT_HERSHEY_COMPLEX
            # our own diag screen
            self.diag1, M = self.projMgr.unwarp_lane(self.curImgFtr.makefull(self.projMgr.diag1), self.projMgr.curSrcRoadCorners, self.projMgr.curDstRoadCorners, self.projMgr.mtx)
            self.diag1 = np.copy(self.projMgr.diag4)
            self.leftLane.scatter_plot(self.diag1)
            self.leftLane.polyline(self.diag1)
            self.rightLane.scatter_plot(self.diag1)
            self.rightLane.polyline(self.diag1)
            cv2.putText(self.diag1, 'Frame: %d'%(self.curFrame), (30, 30), font, 1, (255,0,0), 2)

            self.leftLane.scatter_plot(self.projMgr.diag4)
            self.leftLane.polyline(self.projMgr.diag4)
            self.rightLane.scatter_plot(self.projMgr.diag4)
            self.rightLane.polyline(self.projMgr.diag4)

            cv2.putText(self.projMgr.diag4, 'Frame: %d'%(self.curFrame), (30, 30), font, 1, (255,255,0), 2)
            cv2.putText(self.projMgr.diag4, 'Left: %d count,  %4.1f%% confidence, detected: %r'%(self.leftLanePoints, self.leftLane.confidence*100, self.leftLane.detected), (30, 60), font, 1, (255,255,0), 2)
            cv2.putText(self.projMgr.diag4, 'Left: RoC: %fm, DfVC: %fcm'%(self.leftLane.radiusOfCurvature, self.leftLane.lineBasePos*100), (30, 90), font, 1, (255,255,0), 2)

            cv2.putText(self.projMgr.diag4, 'Right %d count,  %4.1f%% confidence, detected: %r'%(self.rightLanePoints, self.rightLane.confidence*100, self.rightLane.detected), (30, 120), font, 1, (255,255,0), 2)
            cv2.putText(self.projMgr.diag4, 'Right RoC: %fm, DfVC: %fcm'%(self.rightLane.radiusOfCurvature, self.rightLane.lineBasePos*100), (30, 150), font, 1, (255,255,0), 2)

            if self.boosting>0.0:
                cv2.putText(self.projMgr.diag4, 'Boosting @ %f%%'%(self.boosting), (30, 180), font, 1, (128,128,192), 2)

            self.projMgr.diag4 = self.curImgFtr.miximg(self.projMgr.diag4, self.roadsurface, 1.0, 2.0)
            self.projMgr.diag2 = self.curImgFtr.miximg(self.projMgr.diag2, self.roadunwarped, 1.0, 0.5)
            self.projMgr.diag1 = self.curImgFtr.miximg(self.projMgr.diag1, self.roadunwarped[self.mid:self.y,:,:], 1.0, 2.0)

    def drawLaneStats(self, color=(224,192,0)):
        font = cv2.FONT_HERSHEY_COMPLEX
        if self.roadStraight:
            cv2.putText(self.final, 'Estimated lane curvature: road nearly straight', (30, 60), font, 1, color, 2)
        elif self.radiusOfCurvature>0.0:
            cv2.putText(self.final, 'Estimated lane curvature: center is %fm to the right'%(self.radiusOfCurvature), (30, 60), font, 1, color, 2)
        else:
            cv2.putText(self.final, 'Estimated lane curvature: center is %fm to the left'%(-self.radiusOfCurvature), (30, 60), font, 1, color, 2)

        if self.lineBasePos<0.0:
            cv2.putText(self.final, 'Estimated left of center: %5.2fcm'%(-self.lineBasePos*100), (30, 90), font, 1, color, 2)
        elif self.lineBasePos>0.0:
            cv2.putText(self.final, 'Estimated right of center: %5.2fcm'%(self.lineBasePos*100), (30, 90), font, 1, color, 2)
        else:
            cv2.putText(self.final, 'Estimated at center of road', (30, 90), font, 1, color, 2)

