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

#IMAGES TRANSFORMATIONS
blurred_images = [gaussian_blur(image, kernel_size = blur_kernel) for image in images]
gray_images = [make_gray(blurred_image) for blurred_image in blurred_images]
canny_images = [canny_edge(gray_image, low_threshold = canny_low, high_threshold = canny_high) for gray_image in gray_images]
masked_images = [region_of_interest(canny_image, vertices = vertices) for canny_image in canny_images]
hough_images = [hough_lines(masked_image, rho = rho, theta = theta, threshold = threshold, min_line_len = min_line_length, max_line_gap = max_line_gap) for masked_image in masked_images]
annotated_images = [weighted_img(image, hough_image) for (image, hough_image) in zip(images, hough_images)]
hough_images_advanced = [hough_lines_advanced(masked_image, rho = rho, theta = theta, threshold = threshold, min_line_len = min_line_length, max_line_gap = max_line_gap, max_dist = hough_max_dist) for masked_image in masked_images]
annotated_images_2 = [weighted_img(image, hough_image_advanced) for (image, hough_image_advanced) in zip(images, hough_images_advanced)]

#ALL TRANSFORMATIONS
progress = [images, blurred_images, gray_images, canny_images, masked_images, hough_images, annotated_images, hough_images_advanced, annotated_images_2]

#PLOT ALL IMAGES FOR ONE TRANSFORMATION
def plot_all(images):
	labels = ['Straight white', 'Curved white', 'White and yellow', 'White and yellow 2', 'White and yellow 3', 'White and yellow 4']
	fig, axes = plt.subplots(nrows=2, ncols=2, figsize = (10,6))
	axes = axes.ravel()
	fig.tight_layout()

	for ax, image, label in zip(axes, images[:4], labels[:4]):
		ax.imshow(image, cmap='gray')
		ax.set_title(label)
		ax.axis('off')
	plt.show()
	#plt.savefig('image.png', bbox_inches='tight', cmap='gray')

plot_all(annotated_images_2)

#PLOT TRANSFORMATIONS
def plot_progress(progress, test_image_number):
	labels = ['Original', 'Blurred', 'Gray', 'Canny edge', 'Masked', 'Hough', 'Annotated', 'Hough Advaced', 'Annotated 2']
	fig, axes = plt.subplots(nrows=3, ncols=3, figsize = (15,10))
	axes = axes.ravel()
	fig.tight_layout()

	for ax, transformation, label in zip(axes, progress, labels):
		ax.imshow(transformation[test_image_number], cmap='gray')
		ax.set_title(label)
		ax.axis('off')
	plt.show()
	#plt.savefig('image.png', bbox_inches='tight', cmap='gray')


#plot_progress(progress, test_image_number = 0)



