from moviepy.editor import *
from IPython.display import HTML
from functions import *

#IMAGES
images = read_images()
width = images[0].shape[1]
height = images[0].shape[0]

#PARAMS
blur_kernel = 3 #must be odd
canny_low = 50
canny_high = 300
vertices = np.array([[(100,height), (int(width/2) - 80, 325), (int(width/2) + 80, 325), (width - 100,height)]])
rho = 1
theta = np.pi/180
threshold = 23
min_line_length = 5
max_line_gap = 3
hough_max_dist = 330

def process_frame(image):
	blurred_image = gaussian_blur(image, kernel_size = blur_kernel)
	gray_image = make_gray(blurred_image)
	canny_image = canny_edge(gray_image, low_threshold = canny_low, high_threshold = canny_high)
	masked_image = region_of_interest(canny_image, vertices = vertices)
	hough_image = hough_lines(masked_image, rho = rho, theta = theta, threshold = threshold, min_line_len = min_line_length, max_line_gap = max_line_gap)
	annotated_image = weighted_img(image, hough_image)
	hough_image_advanced = hough_lines_advanced(masked_image, rho = rho, theta = theta, threshold = threshold, min_line_len = min_line_length, max_line_gap = max_line_gap, max_dist = hough_max_dist)
	annotated_image_2 = weighted_img(image, hough_image_advanced)
	return annotated_image_2


def process_video(input_path, output_path):
	input_file = VideoFileClip(input_path)
	output_clip = input_file.fl_image(process_frame)
	output_clip.write_videofile(output_path, audio=False)

process_video('test_videos/test_video_1.mp4', 'test_videos_results/test_video_1_annotated.mp4')
