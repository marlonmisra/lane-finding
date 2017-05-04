import math
import matplotlib.pyplot as plt
import glob
import cv2
import numpy as np

def read_images():
	test_image_names = glob.glob('./test_images/test_image_*.jpg')
	test_images = []
	for test_image_name in test_image_names:
		test_image = plt.imread(test_image_name)
		test_images.append(test_image)
	return test_images

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def make_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #expects RGB which you get when reading w matplotlib.imread
    #note cv2.imread reads in BGR
    
def canny_edge(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def region_of_interest(img, vertices):
	#applies iamge mask that only keeps region inside vertices
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) == 3:
        channel_count = img.shape[2] # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #fil pixels inside polygon    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning img only where pixels are non-zero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
	#img is output of canny transform
    
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color = [255,0,0], thickness=10)

    return line_img

def extrapolate(x1, y1, m, y2):
    x2 = int(((y2-y1)/m)+x1)
    return x2

def hough_lines_advanced(img, rho, theta, threshold, min_line_len, max_line_gap, max_dist):
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    
    left_line_points = []
    left_slopes = []

    right_line_points = []
    right_slopes = []

    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = 1.0*(y2-y1)/(x2-x1)
            if slope <= 0 and slope > -0.74:
                left_line_points.append([x1, y1])
                left_line_points.append([x2, y2])
                left_slopes.append(slope)

            elif slope > 0:
                right_line_points.append([x1, y1])
                right_line_points.append([x2, y2])
                right_slopes.append(slope)

    #left
    point = np.mean(left_line_points, axis = 0)
    avg_left_slope = np.mean(left_slopes)
    left_xmin = extrapolate(x1 = point[0], y1 = point[1], m = avg_left_slope, y2 = img.shape[0])
    left_xmax = extrapolate(x1 = point[0], y1 = point[1], m = avg_left_slope, y2 = max_dist)
    cv2.line(line_img, (left_xmin, img.shape[0]), (left_xmax, max_dist), color = [255, 0, 0], thickness = 10)

    #right
    point_2 = np.mean(right_line_points, axis = 0)
    avg_right_slope = np.mean(right_slopes)
    right_xmax = extrapolate(x1 = point_2[0], y1 = point_2[1], m = avg_right_slope, y2 = img.shape[0])
    right_xmin = extrapolate(x1 = point_2[0], y1 = point_2[1], m = avg_right_slope, y2 = max_dist)
    cv2.line(line_img, (right_xmax, img.shape[0]), (right_xmin, max_dist), color = [0, 255, 0], thickness = 10)

    return line_img






def weighted_img(img, initial_img, a=0.8, b=1., c=0.):
	#img is output of hough lines
	#initial_img  is img before any processing
 
    return cv2.addWeighted(initial_img, a, img, b, c)




