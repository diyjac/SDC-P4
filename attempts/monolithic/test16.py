#importing some useful packages
import numpy as np
import cv2
import glob
import re
import os
import pickle
from p4lib.cameraCal import cameraCal
from moviepy.editor import VideoFileClip

## Import math module and create some helper functions
## Define a function that masks out everything except for white and yellow colors pixels
def yuv(image):
    # setup inRange to mask off everything except white and yellow
    yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    u = yuv[:,:,1]
    v = yuv[:,:,2]
    ignore_color = np.copy(v)*0
    img = np.dstack((u, ignore_color, v))
    return img
    #lower = np.array([0, 0, 0])
    #upper = np.array([255, 255, 255])
    #mask = cv2.inRange(yuv, lower, upper)
    #return cv2.bitwise_and(yuv, yuv, mask=mask)

def image_only_yellow_white(image):
    # setup inRange to mask off everything except white and yellow
    lower_yellow_white = np.array([140, 140, 64])
    upper_yellow_white = np.array([255, 255, 255])
    mask = cv2.inRange(image, lower_yellow_white, upper_yellow_white)
    return cv2.bitwise_and(image, image, mask=mask)

## Define a function that applies Gaussian Noise kernel
def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

## Define a function that applies Canny transform
def canny(img, low_threshold, high_threshold, kernel_size):
    img = image_only_yellow_white(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur_gray = gaussian_blur(gray, kernel_size)
    return cv2.Canny(blur_gray, low_threshold, high_threshold)

# Define a function that applies Sobel x or y, 
# then takes an absolute value and applies a threshold.
def abs_sobel_thresh(img, orient='x', thresh=(0, 255)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    # 3) Take the absolute value of the derivative or gradient
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    # 6) Return this mask as your binary_output image
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        abs_sobel = np.absolute(sobelx)
    if orient == 'y':
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        abs_sobel = np.absolute(sobely)
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    ret, binary_output = cv2.threshold(scaled_sobel, thresh[0], thresh[1], cv2.THRESH_BINARY)
    # Return the result
    return binary_output

# Define a function that applies Sobel x and y, 
# then computes the magnitude of the gradient
# and applies a threshold
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude 
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # 5) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # 6) Create a binary mask where mag thresholds are met
    ret, mag_binary = cv2.threshold(gradmag, mag_thresh[0], mag_thresh[1], cv2.THRESH_BINARY)
    # 7) Return this mask as your binary_output image
    return mag_binary

# Define a function that applies Sobel x and y, 
# then computes the direction of the gradient
# and applies a threshold.
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the direction of the gradient
    # 4) Take the absolute value
    with np.errstate(divide='ignore', invalid='ignore'):
        dirout = np.absolute(np.arctan(sobely/sobelx))
        # 5) Create a binary mask where direction thresholds are met
        dir_binary =  np.zeros_like(dirout).astype(np.float32)
        dir_binary[(dirout > thresh[0]) & (dirout < thresh[1])] = 1
        # 6) Return this mask as your binary_output image
    # update nan to number
    np.nan_to_num(dir_binary)
    # make it fit
    dir_binary[(dir_binary>0)|(dir_binary<0)] = 128
    return dir_binary.astype(np.uint8)

# Python 3 has support for cool math symbols.
def miximg(img1, img2, α=0.8, β=1., λ=0.):
    """
    The result image is computed as follows:    
    img1 * α + img2 * β + λ
    NOTE: img1 and img2 must be the same shape!
    """
    return cv2.addWeighted(img1.astype(np.uint8), α, img2.astype(np.uint8), β, λ)

# Define a function that thresholds the S-channel of HLS
def hls_s(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    # 2) Apply a threshold to the S channel
    # 3) Return a binary image of threshold result
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s = hls[:,:,2]
    retval, s_binary = cv2.threshold(s.astype('uint8'), thresh[0], thresh[1], cv2.THRESH_BINARY)
    return s_binary

# Define a function that thresholds the S-channel of HLS
def hls_h(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    # 2) Apply a threshold to the S channel
    # 3) Return a binary image of threshold result
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    h = hls[:,:,0]
    retval, h_binary = cv2.threshold(h.astype('uint8'), thresh[0], thresh[1], cv2.THRESH_BINARY)
    return h_binary

# Current pipeline
# - distortion correction
# - condition detection
def condition_detect(image):
    # setup inRange to mask off everything except white and yellow
    yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    x, y = yuv.shape[1], yuv.shape[0]

    skyl = np.average(yuv[0:int(y/2),0:x,0])
    skyr = np.average(image[0:int(y/2),0:x,0])
    skyg = np.average(image[0:int(y/2),0:x,1])
    skyb = np.average(image[0:int(y/2),0:x,2])
    sky = {'l':skyl,'r':skyr,'g':skyg,'b':skyb}
    if skyl > 160:
        sky['iq'] = 'overexposed'
    elif skyl < 50:
        sky['iq'] = 'underexposed'
    elif skyl > 143:
        sky['iq'] = 'normal bright'
    elif skyl < 113:
        sky['iq'] = 'normal dark'
    else:
        sky['iq'] = 'normal'
    if skyl > 128:
        if skyb > skyl:
            if skyr > 128 and skyg > 128:
                if (skyg-skyr) > 20.0:
                    sky['wc'] = 'tree shaded'
                else:
                    sky['wc'] = 'cloudy'
            else:
                sky['wc'] = 'clear'
        else:
            sky['wc'] = 'unknown SKYL>128'
    else:
        if skyg > skyb:
            sky['wc'] = 'surrounded by trees'
        elif skyb > skyl:
            if (skyg-skyr) > 10.0:
                sky['wc'] = 'tree shaded'
            else:
                sky['wc'] = 'very cloudy or under overpass'
        else:
            sky['wc'] = 'unknown'
    
    groundl = np.average(yuv[int(y/2):y,0:x,0])
    groundr = np.average(image[int(y/2):y,0:x,0])
    groundg = np.average(image[int(y/2):y,0:x,1])
    groundb = np.average(image[int(y/2):y,0:x,2])
    ground = {'l':groundl,'r':groundr,'g':groundg,'b':groundb}
    
    ground['correction'] = 'none'
    ground['vf'] = groundl/10
    ground['v'] = 0
    if groundl > 160:
        ground['iq'] = 'overexposed'
        ground['correction'] = 'exposure balance'
        ground['v'] = 3
    elif groundl < 50:
        ground['iq'] = 'underexposed'
        ground['correction'] = 'exposure balance'
        ground['v'] = -3
    elif groundl > skyl:
        if groundl > 110:
            ground['iq'] = 'too bright'
            ground['correction'] = 'exposure balance'
            if sky['wc'] == 'surrounded by trees' and abs(ground['l']-sky['l'])>9.0 and sky['l']>100.0:
                ground['v'] = 0
                ground['iq'] = 'bright/high contras'
                ground['correction'] = 'none'
            else:
                ground['v'] = 2
        else:
            ground['iq'] = 'bright'
            ground['correction'] = 'exposure balance'
            ground['v'] = 1
    else:
        if sky['wc'] == 'clear' or sky['wc'] == 'cloudy':
            if groundl > 110:
                ground['iq'] = 'too bright'
                ground['correction'] = 'exposure balance'
                ground['v'] = 2
            elif groundl > 90:
                ground['iq'] = 'bright'
                ground['correction'] = 'exposure balance'
                ground['v'] = 1
            elif groundl < 60:
                ground['iq'] = 'too dark'
                ground['correction'] = 'exposure balance'
                ground['v'] = -2
            elif groundl < 80:
                ground['iq'] = 'dark'
                ground['correction'] = 'exposure balance'
                ground['v'] = -1
            else:
                ground['iq'] = 'normal'
        else:
            ground['iq'] = 'dark'
            ground['correction'] = 'exposure balance'
            ground['v'] = -1

    
    #print("condition: sky", sky, "ground",ground)    
    return sky, ground, image

def balance_exposure(sky, ground, image):
    # setup
    x, y = image.shape[1], image.shape[0]
    r = image[int(y/2):y,0:x,0]
    g = image[int(y/2):y,0:x,1]
    b = image[int(y/2):y,0:x,2]
    yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    yo = yuv[int(y/2):y,0:x,0].astype(np.float32)
    yc = (yo/ground['vf'])*8.0
    lymask = np.copy(yc)
    lymask[lymask>255] = 255
    if ground['v'] == -1:
        lymask[(lymask>255.0)] = 255.0
        lymask[(r>ground['g'])&(g>ground['g'])] = 0
    elif ground['v'] > -3 and ground['v'] < 3 and sky['l']>70.0:
        lymask[(lymask>255.0)] = 255.0
        lymask[(lymask>128.0)] = 0.0
    elif ground['v'] == 3:
        lymask[(lymask>255.0)] = 255.0
        #lymask[(lymask<128.0)|(lymask>200)] = 0
        #lymask[(lymask<64.0)|(lymask>200)] = 0
    else:
        lymask[(lymask>224.0)] = 0.0
    uymask = np.copy(yc)*0
    yc -= lymask

    if sky['wc'] == 'surrounded by trees':
        if ground['v'] > 0 and ground['v'] < 3:
            if ground['b'] < ground['l']:
                #yc[(b>ground['b'])&(yo>160+(ground['v']*20))] = 242.0
                yc[(b>ground['b'])&(yo>200+(ground['v']*20))] = 242.0
            else:
                yc[(b>ground['b'])&(yo>200-(ground['v']*10))] = 242.0
                #yc[(b>ground['b'])&(yo>200)] = 242.0
                #yc[(b>ground['b'])&(b<yo)] = 242.0
        elif ground['v'] == 3:
            yc[(b>254)&(g>254)&(r>254)] = 242.0
            #yc[(b>ground['b'])&(yo>200-(ground['v']*10))] = 242.0
        elif ground['v'] > -3 and ground['v'] < 0:
            yc[(b>ground['b'])&(b<r)] = 242.0
    elif sky['wc'] == 'clear':
        yc[(b>ground['b'])&(yo>180+(ground['v']*20))] = 242.0
        #yc[(b>ground['r'])&(b<r)] = 242.0
    elif sky['wc'] == 'cloudy':
        if ground['v'] == 3:
            yc[(b>254)&(g>254)&(r>254)] = 242.0
        else:
            yc[(b>ground['b'])&(yo>160+(ground['v']*20))] = 242.0
    else:
        yc[(b>ground['b'])&(yo>200+(ground['v']*10))] = 242.0
        #yc[(b>ground['r'])&(b<r)] = 242.0

    if ground['v'] == 2:
        uymask[(b<ground['l'])&(r>ground['l'])&(g>ground['l'])] = 242.0
    elif ground['v'] == 1:
        uymask[(b<ground['r'])&(r>ground['b'])&(g>ground['b'])] = 242.0
    elif ground['v'] == -1:
        uymask[(b<ground['b'])&(r>ground['r'])&(g>ground['g'])] = 242.0
    elif ground['v'] == 3:
        uymask[(b<ground['l'])&(r>ground['l'])&(g>ground['l'])] = 242.0
    elif ground['v'] > -3:
        uymask[(b<ground['b'])&(r>ground['b'])&(g>ground['b'])] = 242.0

    yc = miximg(yc,uymask,1.0,1.0)
    #yc = miximg(yc,(yo/ground['vf'])*8.0,1.0,0.8)
    
    yuv[int(y/2):y,0:x,0] = yc.astype(np.uint8)
    yuv[(y-40):y,0:x,0] = yo[((y/2)-40):(y/2),0:x].astype(np.uint8)
    return sky, ground, cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)

def balance_exposure_temp(sky, ground, image):
    # setup
    x, y = image.shape[1], image.shape[0]
    r = image[int(y/2):y,0:x,0]
    g = image[int(y/2):y,0:x,1]
    b = image[int(y/2):y,0:x,2]
    yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    yo = yuv[int(y/2):y,0:x,0].astype(np.float32)
    yc = (yo/ground['vf'])*8.0
    lymask = np.copy(yc)
    lymask[(lymask>255.0)] = 255.0
    uymask = np.copy(yc)*0
    yc -= lymask
    if ground['v'] == 3:
        yc[(b>254)&(g>254)&(r>254)] = 242.0
    elif ground['v'] == 1:
        yc[(b>ground['b'])&(yo>160+(ground['v']*20))] = 242.0
    else:
        yc[(b>ground['b'])&(yo>210+(ground['v']*10))] = 242.0
    uymask[(b<ground['l'])&(r>ground['l'])&(g>ground['l'])] = 242.0
    yc = miximg(yc,uymask,1.0,1.0)
    yc = miximg(yc,yo,1.0,0.8)
    yc[int((y/72)*70):y,0:x] = 0
    yuv[int(y/2):y,0:x,0] = yc.astype(np.uint8)
    yuv[(y-40):y,0:x,0] = yo[((y/2)-40):(y/2),0:x].astype(np.uint8)
    return sky, ground, cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)

# create a region of interest mask
def region_of_interest(img, vertices):
    """
    Applies an image mask.
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


# draw outline of given area
def draw_area_of_interest(img, areas, color=[128,0,128], thickness=2):
    for points in areas:
        for i in range(len(points)-1):
            cv2.line(img, (points[i][0], points[i][1]), (points[i+1][0],points[i+1][1]), color, thickness)
        cv2.line(img, (points[0][0], points[0][1]), (points[len(points)-1][0],points[len(points)-1][1]), color, thickness)

# draw outline of given area        
def draw_area_of_interest_for_projection(img, areas, color=[128,0,128], thickness1=2, thickness2=10):
    for points in areas:
        for i in range(len(points)-1):
            if i==0 or i==1:
                cv2.line(img, (points[i][0], points[i][1]), (points[i+1][0],points[i+1][1]), color, thickness1)
            else:
                cv2.line(img, (points[i][0], points[i][1]), (points[i+1][0],points[i+1][1]), color, thickness2)
        cv2.line(img, (points[0][0], points[0][1]), (points[len(points)-1][0],points[len(points)-1][1]), color, thickness1)

def draw_masked_area(img, areas, color=[128,0,128], thickness=2):
    for points in areas:
        for i in range(len(points)-1):
            cv2.line(img, (points[i][0], points[i][1]), (points[i+1][0],points[i+1][1]), color, thickness)
        cv2.line(img, (points[0][0], points[0][1]), (points[len(points)-1][0],points[len(points)-1][1]), color, thickness)

def draw_bounding_box(img, boundingbox, color=[0,255,0], thickness=6):
    x1, y1, x2, y2 = boundingbox
    cv2.line(img, (x1, y1), (x2, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y2), color, thickness)
    cv2.line(img, (x2, y2), (x1, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y1), color, thickness)


# calculate and draw initial estimated lines on the roadway.
def draw_lines(img, lines, color=[255, 0, 0], thickness=6, backoff=0, debug=False):
    if backoff==0:
        backoff=thickness*5
    ysize = img.shape[0]
    midleft = img.shape[1]/2-200+backoff*2
    midright = img.shape[1]/2+200-backoff*2
    top = ysize/2+backoff*2
    rightslopemin = 0.5 #8/backoff
    rightslopemax = 3.0 #backoff/30
    leftslopemax =  -0.5 #-8/backoff
    leftslopemin = -3.0 #-backoff/30
    try:
        # rightline and leftline cumlators
        rl = {'num':0, 'slope':0.0, 'x1':0, 'y1':0, 'x2':0, 'y2':0}
        ll = {'num':0, 'slope':0.0, 'x1':0, 'y1':0, 'x2':0, 'y2':0}
        for line in lines:
            for x1,y1,x2,y2 in line:
                slope = ((y2-y1)/(x2-x1))
                sides = (x1+x2)/2
                vmid = (y1+y2)/2
                if slope > rightslopemin and slope < rightslopemax and sides > midright and vmid > top:   # right
                    if debug:
                        #print("x1,y1,x2,y2: ", x1, y1, x2, y2)
                        cv2.line(img, (x1, y1), (x2, y2), [128, 128, 0], thickness)
                    rl['num'] += 1
                    rl['slope'] += slope
                    rl['x1'] += x1
                    rl['y1'] += y1
                    rl['x2'] += x2
                    rl['y2'] += y2
                elif slope > leftslopemin and slope < leftslopemax and sides < midleft and vmid > top:   # left
                    if debug:
                        #print("x1,y1,x2,y2: ", x1, y1, x2, y2)
                        cv2.line(img, (x1, y1), (x2, y2), [128, 128, 0], thickness)
                    ll['num'] += 1
                    ll['slope'] += slope
                    ll['x1'] += x1
                    ll['y1'] += y1
                    ll['x2'] += x2
                    ll['y2'] += y2        

        if rl['num'] > 0 and ll['num'] > 0:
            # average/extrapolate all of the lines that makes the right line
            rslope = rl['slope']/rl['num']
            rx1 = int(rl['x1']/rl['num'])
            ry1 = int(rl['y1']/rl['num'])
            rx2 = int(rl['x2']/rl['num'])
            ry2 = int(rl['y2']/rl['num'])
            
            # average/extrapolate all of the lines that makes the left line
            lslope = ll['slope']/ll['num']
            lx1 = int(ll['x1']/ll['num'])
            ly1 = int(ll['y1']/ll['num'])
            lx2 = int(ll['x2']/ll['num'])
            ly2 = int(ll['y2']/ll['num'])
            
            # find the right and left line's intercept, which means solve the following two equations
            # rslope = ( yi - ry1 )/( xi - rx1)
            # lslope = ( yi = ly1 )/( xi - lx1)
            # solve for (xi, yi): the intercept of the left and right lines
            # which is:  xi = (ly2 - ry2 + rslope*rx2 - lslope*lx2)/(rslope-lslope)
            # and        yi = ry2 + rslope*(xi-rx2)
            xi = int((ly2 - ry2 + rslope*rx2 - lslope*lx2)/(rslope-lslope))
            yi = int(ry2 + rslope*(xi-rx2))
            
            # calculate backoff from intercept for right line
            if rslope > rightslopemin and rslope < rightslopemax:   # right
                ry1 = yi + int(backoff)
                rx1 = int(rx2-(ry2-ry1)/rslope)
                ry2 = ysize-1
                rx2 = int(rx1+(ry2-ry1)/rslope)
                cv2.line(img, (rx1, ry1), (rx2, ry2), [255, 0, 0], thickness)

            # calculate backoff from intercept for left line
            if lslope < leftslopemax and lslope > leftslopemin:   # left
                ly1 = yi + int(backoff)
                lx1 = int(lx2-(ly2-ly1)/lslope)
                ly2 = ysize-1
                lx2 = int(lx1+(ly2-ly1)/lslope)
                cv2.line(img, (lx1, ly1), (lx2, ly2), [255, 0, 0], thickness)
                
            if lx1 > 0 and ly1 > 0 and rx1 > 0 and ry1 > 0:
                cv2.line(img, (lx1, ly1), (rx1, ry1), [255, 0, 0], thickness)
        return lslope+rslope, lslope, rslope, (lx1,ly1), (rx1,ry1), (rx2, ry2), (lx2, ly2), (xi, yi)
    except:
        return -1000, 0.0, 0.0, (0,0), (0,0), (0,0), (0,0)

# draw parallel lines in a perspective image that will later be projected into a flat surface
def draw_parallel_lines_pre_projection(img, lane_info, color=[128,0,0], thickness=5):
    lx1 = lane_info[3][0]
    rx1 = lane_info[4][0]
    rx2 = lane_info[5][0]
    lx2 = lane_info[6][0]
    ly1 = lane_info[3][1]
    ry1 = lane_info[4][1]
    ry2 = lane_info[5][1]
    ly2 = lane_info[6][1]
    cv2.line(img, (lx1, ly1), (lx2, ly2), color, thickness)
    cv2.line(img, (rx1, ry1), (rx2, ry2), color, thickness)

# generate a set of hough lines and calculates its estimates for lane lines
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, backoff=0, debug=False):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn using the new single line for left and right lane line method.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    masked_lines = np.zeros(img.shape, dtype=np.uint8)
    lane_info = draw_lines(masked_lines, lines, backoff=backoff, debug=debug)
    
    return masked_lines, lane_info


# function to project the undistorted camera image to a plane looking down.
def unwarp_lane(img, src, dst, mtx):
    # Pass in your image, 4 source points src = np.float32([[,],[,],[,],[,]])
    # and 4 destination points dst = np.float32([[,],[,],[,],[,]])
    # Note: you could pick any four of the detected corners 
    # as long as those four corners define a rectangle
    # One especially smart way to do this would be to use four well-chosen
    # use cv2.getPerspectiveTransform() to get M, the transform matrix
    # use cv2.warpPerspective() to warp your image to a top-down view
    M = cv2.getPerspectiveTransform(src, dst)
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    # warped = gray
    return warped, M

# function to find starting lane line positions
# return left and right column positions
def find_lane_locations(masked_lines):
    height = masked_lines.shape[0]
    width = masked_lines.shape[1]
    lefthistogram = np.sum(masked_lines[int(height/2):height,0:int(width/2)], axis=0).astype(np.float32)
    righthistogram = np.sum(masked_lines[int(height/2):height,int(width/2):width], axis=0).astype(np.float32)
    leftpos = np.argmax(lefthistogram)
    rightpos =  np.argmax(righthistogram)+int(width/2)
    #print("leftpos",leftpos,"rightpos",rightpos)
    return leftpos, rightpos, rightpos-leftpos

# function to find lane line positions given histogram row, last column positions and n_neighbors
# return left and right column positions
def find_lane_nearest_neighbors(histogram, lastleftpos, lastrightpos, nneighbors):
    ncol = len(histogram)-1
    leftx = []
    rightx = []
    left = {"count":0, "position":lastleftpos}
    right = {"count":0, "position":lastrightpos}
    for i in range(nneighbors):
        if histogram[lastleftpos+i] > 0:
            leftx.append(lastleftpos+i)
            if left['count'] < histogram[lastleftpos+i]:
                left['count'] = histogram[lastleftpos+i]
                left['position'] = lastleftpos+i
        if (lastleftpos-i) > 0 and histogram[lastleftpos-i] > 0:
            leftx.append(lastleftpos-i)
            if left['count'] < histogram[lastleftpos-i]:
                left['count'] = histogram[lastleftpos-i]
                left['position'] = lastleftpos-i
        if (lastrightpos+i)< ncol and histogram[lastrightpos+i] > 0:
            rightx.append(lastrightpos+i)
            if right['count'] < histogram[lastrightpos+i]:
                right['count'] = histogram[lastrightpos+i]
                right['position'] = lastrightpos+i
        if histogram[lastrightpos-i] > 0:
            rightx.append(lastrightpos-i)
            if right['count'] < histogram[lastrightpos-i]:
                right['count'] = histogram[lastrightpos-i]
                right['position'] = lastrightpos-i
    return left['position'], right['position'], leftx, rightx

# function to find lane lines points using a sliding window histogram given starting positions
# return arrays or left, right and y positions
def find_lane_lines_points(masked_lines, leftpos, rightpos, distance, found=False):
    leftval = []
    rightval = []
    yval = []
    nrows = masked_lines.shape[0] - 1
    neighbors = 12
    if True:
        left1 = leftpos
        right1 = rightpos
        start_row = nrows-16
        for i in range(int((nrows/neighbors))):
            histogram = np.sum(masked_lines[start_row+10:start_row+26,:], axis=0).astype(np.uint8)
            left2, right2, leftx, rightx = find_lane_nearest_neighbors(histogram, left1, right1, int(neighbors*1.3))
            y = start_row + neighbors
            if len(leftx) < len(rightx):
                extra = len(rightx) - len(leftx)
                for i in range(len(rightx)):
                    rightval.append(rightx[i])
                    yval.append(y)
                for i in range(len(leftx)):
                    leftval.append(leftx[i])
                for i in range(extra):
                    leftval.append(rightx[i]-distance)
                if len(leftx) == 0 and not found:
                    distance -= 2
                    left1 = right2-distance
                else:
                    left1 = left2
                right1 = right2
            else:
                extra = len(leftx) - len(rightx)
                for i in range(len(leftx)):
                    leftval.append(leftx[i])
                    yval.append(y)
                for i in range(len(rightx)):
                    rightval.append(rightx[i])
                for i in range(extra):
                    rightval.append(leftx[i]+distance)
                if len(rightx) == 0 and not found:
                    distance -= 2
                    right1 = left2+distance
                else:
                    right1 = right2
                left1 = left2
         
            start_row -= neighbors
    return np.array(leftval), np.array(rightval), np.array(yval)

# Define y-value where we want radius of curvature
# We'll choose the maximum y-value, corresponding to the bottom of the image
def radius_of_curvature(left_fit, right_fit, y_eval):
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5)/np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5)/np.absolute(2*right_fit[0])
    return left_curverad, right_curverad
    
# Define conversions in x and y from pixels space to meters
def radius_in_meters(leftx, rightx, yvals):
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meteres per pixel in x dimension
    y_eval = np.max(yvals)
    left_fit_cr = np.polyfit(yvals*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(yvals*ym_per_pix, rightx*xm_per_pix, 2)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval + left_fit_cr[1])**2)**1.5)/np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval + right_fit_cr[1])**2)**1.5)/np.absolute(2*right_fit_cr[0])
    return left_curverad, right_curverad

def meters_from_center(pixels_off_center):
    xm_per_pix = 3.7/700 # meteres per pixel in x dimension
    return xm_per_pix * pixels_off_center

def scatter_plot(img, x, y, color=(0,0,255), size=3):
    xy_array = np.column_stack((x, y)).astype(np.int32)
    for xy in xy_array:
        cv2.circle(img, (xy[0], xy[1]), size, color, -1)

def diagnostic_process_image(img0, t=0, found=False, plane_info=None, previouscombined=None):
    debug=True
    img_size = (img0.shape[1], img0.shape[0])
    sky, ground, image = condition_detect(cv2.undistort(img0, mtx, dist, None, mtx))
    #if not ground['v'] == 0:
    #    if sky['wc'] == 'cloudy':
    #        sky, ground, image = balance_exposure_temp(sky, ground, image)
    #    else:
    #        sky, ground, image = balance_exposure(sky, ground, image)

    img = np.copy(image)

    # mask calculations
    imshape = image.shape
    xbottom1 = int(imshape[1]/16)
    xbottom2 = int(imshape[1]*15/16)
    xtop1 = int(imshape[1]*14/32)
    xtop2 = int(imshape[1]*18/32)
    ybottom1 = imshape[0]
    ybottom2 = imshape[0]
    ytopbox = int(imshape[0]*9/16)

    # Run the functions
    gradx = abs_sobel_thresh(img, orient='x', thresh=(25, 100))
    grady = abs_sobel_thresh(img, orient='y', thresh=(50, 150))
    magch = mag_thresh(img, sobel_kernel=9, mag_thresh=(50, 250))
    dirch = dir_threshold(img, sobel_kernel=15, thresh=(0.7, 1.3))
    sch = hls_s(img, thresh=(88, 190))
    hch = hls_h(img, thresh=(50, 100))

    # Output "masked_lines" is a single channel mask
    shadow = np.zeros_like(dirch).astype(np.uint8)
    shadow[(sch > 0) & (hch > 0)] = 128

    # remove the green pixels!
    ignore_green = np.zeros_like(dirch).astype(np.uint8)
    ignore_green[(grady > 75) & (magch > 125)] = 128

    rEdgeDetect = ((img[:,:,0]/4)).astype(np.uint8)
    rEdgeDetect = 255-rEdgeDetect
    rEdgeDetect[(rEdgeDetect>210)] = 0

    # create diagnostic screen 1-3
    ignore_color = np.copy(gradx)*0 # creating a blank color channel for combining
    diag1 = np.dstack((rEdgeDetect, gradx, grady))
    diag2 = np.dstack((ignore_color, magch, dirch))
    diag3 = np.dstack((sch, sch, hch))

    # create diagnostic screen 4
    combined = np.zeros_like(dirch).astype(np.uint8)
    combined[((gradx > 0) | (grady > 0) | ((magch > 0) & (dirch > 0)) | (sch > 0)) & (shadow == 0) & (rEdgeDetect>0)] = 35
    previouscombined[:,0:640] = 0
    combined = miximg(combined, previouscombined, 1.0, 0.9)
    combinedlarge = cv2.resize(combined, (1920,1080), interpolation=cv2.INTER_AREA)
    combinedlarge[210:880, 320:1600] = combined[:670,:]
    combined = np.copy(combinedlarge[210:930,320:1600])

    # This time we are defining a four sided polygon to mask
    # We can lift the mask higher now, since the line drawing function is a bit smarter
    vertices = np.array([[(xbottom1,ybottom1),(xtop1, ytopbox), (xtop2, ytopbox), (xbottom2,ybottom2)]], dtype=np.int32)
    masked_edges = region_of_interest(np.copy(combined), vertices)
    diag4 = np.dstack((masked_edges,combined,combined))

    if not found and t<2:
        color = (255,255,0)
        # Define the Hough transform parameters
        # Make a blank the same size as our image to draw on
        rho = 2 # distance resolution in pixels of the Hough grid
        theta = np.pi/180 # angular resolution in radians of the Hough grid
        threshold = 40     # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 120 # 50 75 25 minimum number of pixels making up a line
        max_line_gap = 40    # 40 50 20 maximum gap in pixels between connectable line segments

        line_image, lane_info = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap, backoff=30, debug=debug)
        if lane_info[0] == -1000:
            # Define the Hough transform parameters
            # Make a blank the same size as our image to draw on
            rho = 2 # distance resolution in pixels of the Hough grid
            theta = np.pi/180 # angular resolution in radians of the Hough grid
            threshold = 40     # minimum number of votes (intersections in Hough grid cell)
            min_line_length = 75 # 50 75 25 minimum number of pixels making up a line
            max_line_gap = 40    # 40 50 20 maximum gap in pixels between connectable line segments
            line_image, lane_info = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap, backoff=40, debug=debug)
            if lane_info[0] == -1000:
                # Define the Hough transform parameters
                # Make a blank the same size as our image to draw on
                rho = 2 # distance resolution in pixels of the Hough grid
                theta = np.pi/180 # angular resolution in radians of the Hough grid
                threshold = 40     # minimum number of votes (intersections in Hough grid cell)
                min_line_length = 25 # 50 75 25 minimum number of pixels making up a line
                max_line_gap = 20    # 40 50 20 maximum gap in pixels between connectable line segments
                line_image, lane_info = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap, backoff=50, debug=debug)
                font = cv2.FONT_HERSHEY_COMPLEX
                cv2.putText(img, '3', (30, 60), font, 1, (255,0,0), 2)
            else:
                font = cv2.FONT_HERSHEY_COMPLEX
                cv2.putText(img, '2', (30, 60), font, 1, (255,0,0), 2)
        else:
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(img, '1', (30, 60), font, 1, (255,0,0), 2)
    
        if debug and lane_info[0] > -1000:
            leftbound  = int(lane_info[7][0]-(imshape[1]*0.1))
            rightbound = int(lane_info[7][0]+(imshape[1]*0.1))
            topbound =  int(lane_info[7][1]-(imshape[0]*0.15))
            bottombound = int(lane_info[7][1]+(imshape[0]*0.05))
            boundingbox = (leftbound-2,topbound-2,rightbound+2,bottombound+2)
            draw_masked_area(img, vertices)
            draw_bounding_box(img, boundingbox)

        plane_info = lane_info
    else:
        color = (0,255,0)
        lane_info = plane_info
        line_image = np.copy(ignore_color)
                            
    # calculate the area of interest, this will be used later to reproject just the lanes that needs advance finding
    areaOfInterest = np.array([[(lane_info[3][0]-50,lane_info[3][1]-11),(lane_info[4][0]+50,lane_info[4][1]-11), (lane_info[4][0]+525,lane_info[4][1]+75), (lane_info[4][0]+500,lane_info[5][1]), (lane_info[4][0]-500,lane_info[6][1]), (lane_info[3][0]-525,lane_info[3][1]+75)]], dtype=np.int32)

    # generate estimated line image
    line_image = np.dstack((line_image, ignore_color, ignore_color))
        
    # masked overlay estimated lane line
    masked_edges_purple = np.dstack((masked_edges, ignore_color, masked_edges))
    masked_edges = np.dstack((masked_edges, masked_edges, masked_edges))

    # just the image with the estimated lane line projection
    img1 = miximg(img, masked_edges)
    img1 = miximg(img1, line_image)
    draw_area_of_interest(img1, areaOfInterest, color=[0,128,0], thickness=5)

    # just the image with the estimated lane line projection
    diag5 = miximg(image, masked_edges)
    diag5 = miximg(diag5, line_image)
    draw_area_of_interest(diag5, areaOfInterest, color=[0,128,0], thickness=5)
    leftbound  = int(plane_info[7][0]-(imshape[1]*0.1))
    rightbound = int(plane_info[7][0]+(imshape[1]*0.1))
    topbound =  int(plane_info[7][1]-(imshape[0]*0.15))
    bottombound = int(plane_info[7][1]+(imshape[0]*0.05))
    boundingbox = (leftbound-2,topbound-2,rightbound+2,bottombound+2)
    draw_bounding_box(diag5, boundingbox)
        
    # generate src and dst rects for projection of road to flat plane
    us_lane_width = 12     # US highway width: 12 feet wide
    approx_dest = 20       # Approximate distance to vanishing point from end of rectangle
    scale_factor = 15      # scaling for display
    left = -us_lane_width/2*scale_factor
    right = us_lane_width/2*scale_factor
    top = approx_dest * scale_factor
    src = np.float32([lane_info[3],lane_info[4],lane_info[5],lane_info[6]])
    dst = np.float32([[img_size[0]/2+left,top],[img_size[0]/2+right,top],[img_size[0]/2+right,img_size[1]],[img_size[0]/2+left,img_size[1]]])

    # generate overlay line image
    diag6 = miximg(image, masked_edges_purple*4)
    draw_area_of_interest_for_projection(diag6, areaOfInterest, color=[0,128,0], thickness1=1, thickness2=50)
    draw_parallel_lines_pre_projection(diag6, lane_info, color=[128,0,0], thickness=2)
    diag6, M = unwarp_lane(diag6, src, dst, mtx)

    # generate 
    diag7 = np.copy(masked_edges)
    guide = np.copy(masked_edges)*0
    draw_parallel_lines_pre_projection(guide, lane_info, color=[128,128,0], thickness=50)
    diag7, M = unwarp_lane(diag7, src, dst, mtx)
    guide, M = unwarp_lane(guide, src, dst, mtx)
    guide = cv2.cvtColor(guide, cv2.COLOR_BGR2GRAY)
    projected_lane_lines = cv2.cvtColor(diag7, cv2.COLOR_BGR2GRAY)
        
    # return left and right column positions
    left, right, distance = find_lane_locations(projected_lane_lines)

    # function to find lane lines points using a sliding window histogram given starting positions
    # return arrays or left, right and y positions
    leftx, rightx, yvals = find_lane_lines_points(projected_lane_lines, left, right, distance, found)
        
    if len(yvals) > 0:
        if t>5:
            found = True

        # Fit a second order polynomial to left lane line
        leftcoefficients = np.polyfit(yvals, leftx, 2)
        leftpolynomial = np.poly1d(leftcoefficients)
        left_fitx = leftpolynomial(yvals)

        # Fit a second order polynomial to right lane line
        rightcoefficients = np.polyfit(yvals, rightx, 2)
        rightpolynomial = np.poly1d(rightcoefficients)
        right_fitx = rightpolynomial(yvals)
    
        # create x,y arrays for cv2
        rightxy = np.column_stack((right_fitx, yvals)).astype(np.int32)
        leftxy = np.column_stack((left_fitx, yvals)).astype(np.int32)

        # draw dots and polyline
        scatter_plot(diag7, leftx, yvals, color=(255,0,0))
        scatter_plot(diag7, rightx, yvals, color=(0,0,255))
        cv2.polylines(diag7, [leftxy], 0, (0,255,0), 5)
        cv2.polylines(diag7, [rightxy], 0, (0,255,0), 5)
    
        # create road mask polygon
        roadpoly = np.concatenate((rightxy, leftxy[::-1]), axis=0)
        overlaymask = np.copy(ignore_color)
        cv2.fillConvexPoly(overlaymask, roadpoly, 64)

        # create road mask polygon for quicker lane find
        rightxy1 = np.column_stack((right_fitx+20, yvals)).astype(np.int32)
        rightxy2 = np.column_stack((right_fitx-20, yvals)).astype(np.int32)
        leftxy1 = np.column_stack((left_fitx+20, yvals)).astype(np.int32)
        leftxy2 = np.column_stack((left_fitx-20, yvals)).astype(np.int32)
        roadpoly1 = np.concatenate((rightxy1, rightxy2[::-1]), axis=0)
        roadpoly2 = np.concatenate((leftxy1, leftxy2[::-1]), axis=0)
        roadmask = np.copy(ignore_color)
        cv2.fillConvexPoly(roadmask, roadpoly1, 64)
        cv2.fillConvexPoly(roadmask, roadpoly2, 64)

        diag8 = np.dstack((ignore_color, overlaymask, ignore_color))
        # diag8 = np.dstack((roadmask, overlaymask, roadmask))
        diag9, M = unwarp_lane(image, src, dst, mtx)
        diag9 = miximg(diag9, diag8)

        projected_roadmask, M = unwarp_lane(diag8, dst, src, mtx)
        mainDiagScreen = miximg(image, projected_roadmask)
        
        # lane mask
        lanemask = np.dstack((roadmask, roadmask, roadmask))
        projected_roadmask, M = unwarp_lane(lanemask, dst, src, mtx)
        
        # calculate curvature of road - average between left and right curvature
        left_curverad, right_curverad = radius_in_meters(leftx, rightx, yvals)
        est_road_curverad = (left_curverad+right_curverad)/2.0

        # meters from center
        screen_middel_pixel = img.shape[1]/2
        left_lane_pixel = lane_info[6][0]
        right_lane_pixel = lane_info[5][0]
        car_middle_pixel = int((right_lane_pixel + left_lane_pixel)/2)
        screen_off_center = screen_middel_pixel-car_middle_pixel
        meters_off_center = meters_from_center(screen_off_center)
        
        # using cv2 for drawing text in diagnostic pipeline.
        font = cv2.FONT_HERSHEY_COMPLEX
        middlepanel = np.zeros((120, 1280, 3), dtype=np.uint8)
        cv2.putText(middlepanel, 'Estimated lane curvature: %f'%(est_road_curverad), (30, 60), font, 1, color, 2)
        if meters_off_center>0:
            cv2.putText(middlepanel, 'Estimated Meters right of center: %f'%(meters_off_center), (30, 90), font, 1, color, 2)
        else:
            cv2.putText(middlepanel, 'Estimated Meters left of center: %f'%(-meters_off_center), (30, 90), font, 1, color, 2)
            
        if np.min(yvals) > 300:
            found = False

    else:
        found = False
        diag8 = np.dstack((projected_lane_lines, ignore_color, ignore_color))
    
        diag9, M = unwarp_lane(image, src, dst, mtx)
        diag9 = miximg(diag9, diag8)
    
        mainDiagScreen = np.copy(image)
        
        # using cv2 for drawing text in diagnostic pipeline.
        font = cv2.FONT_HERSHEY_COMPLEX
        middlepanel = np.zeros((120, 1280, 3), dtype=np.uint8)
        cv2.putText(middlepanel, 'Estimated lane curvature: ERROR!', (30, 60), font, 1, (255,0,0), 2)
        cv2.putText(middlepanel, 'Estimated Meters right of center: ERROR!', (30, 90), font, 1, (255,0,0), 2)

    # assemble the screen
    diagScreen = np.zeros((1080,1920,3), dtype=np.uint8)
    diagScreen[0:720, 0:1280] = mainDiagScreen
    diagScreen[0:240, 1280:1600] = cv2.resize(diag1, (320,240), interpolation=cv2.INTER_AREA) 
    diagScreen[0:240, 1600:1920] = cv2.resize(diag2, (320,240), interpolation=cv2.INTER_AREA)
    diagScreen[240:480, 1280:1600] = cv2.resize(diag3, (320,240), interpolation=cv2.INTER_AREA)
    diagScreen[240:480, 1600:1920] = cv2.resize(diag4, (320,240), interpolation=cv2.INTER_AREA)*4
    diagScreen[600:1080, 1280:1920] = cv2.resize(diag7, (640,480), interpolation=cv2.INTER_AREA)*4
    diagScreen[720:840, 0:1280] = middlepanel
    diagScreen[840:1080, 0:320] = cv2.resize(diag5, (320,240), interpolation=cv2.INTER_AREA)
    diagScreen[840:1080, 320:640] = cv2.resize(diag6, (320,240), interpolation=cv2.INTER_AREA)
    diagScreen[840:1080, 640:960] = cv2.resize(diag9, (320,240), interpolation=cv2.INTER_AREA)
    diagScreen[840:1080, 960:1280] = cv2.resize(diag8, (320,240), interpolation=cv2.INTER_AREA)
    return diagScreen, found, plane_info, combined, t+1

def process_diagnostic_image(image):
    global found
    global plane_info
    global combined
    global t
    
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image with lines are drawn on lanes)
    result, found, plane_info, combined, t = diagnostic_process_image(image, t, found, plane_info, combined)
    return result

camCal = cameraCal('camera_cal', 'camera_cal/calibrationdata.p')
mtx, dist, imgSize = camCal.get()
found = False
plane_info = None
combined = np.zeros((720, 1280), dtype=np.uint8)
t = 0
videoout = 'project_video_diagnostics_test_filter18.mp4'
clip1 = VideoFileClip("project_video.mp4")
video_clip = clip1.fl_image(process_diagnostic_image) #NOTE: this function expects color images!!
video_clip.write_videofile(videoout, audio=False)

