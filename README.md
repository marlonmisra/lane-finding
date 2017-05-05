## Lane finding project

### Introduction 
In order to navigate a self-driving car, one of core skills is detecting lane markings and extrapolating full lane lines. In this project, I'm  starting with test images and test videos that come from a front-facing car camera, and I'm showing a simple way to output predicted lane lines. 

The steps I'll be describing are: 
* Applying Gaussian blur and grayscale transformations to remove the effects of noise and remove unnecessary colors.
* Using a form of edge detection, called the Canny Edge transform, and tuning the parameters. 
* Applying a mask that removes areas of the image that don't provide value.
* Utilizing the Hough transform, a technique that detects line segments.
* Taking individual lane segments, dividing them into left/right lane buckets based on slope, and extrpolating the full lane lines.
* Annotating the original image with predicated lane lines. 

[//]: # (Image References)

[image1]: ./readme_assets/process.png "Process"
[image2]: ./readme_assets/original_images.png "Original images"
[image3]: ./readme_assets/blurred_grayscale_images.png "Blurred grayscale images"
[image4]: ./readme_assets/edge_images.png "Edge images"
[image5]: ./readme_assets/window_images.png "Window images"
[image6]: ./readme_assets/hough_images.png "Hough images"
[image7]: ./readme_assets/hough_advanced_images.png "Hough images"
[image8]: ./readme_assets/annotated_images.png "Annotated images"
[image9]: ./readme_assets/test_video_1_annotated.gif "Video"

![alt text][image1]


### Files and project navigation 
* test_images and test_videos contain testing data.
* test_images_results and test_videos_results are folders that contain testing data with predicated lane lines.
* functions.py contains transformation functions and helper functions.
* exploratory.py contains parameters and methods to plot test images and transformations.
* pipeline.py contains the video processing pipeline and `process_frame(image)` function which is used on each frame.  

### Pipeline
**Original images**
The test images and frames from the video have the shape (124,23,3), meaning a height of a, a width of x, and 3 RGB channels.

![alt text][image2]

**Gaussian blur and grayscale transform**
As the first step, I applied a Gaussian blur to the images. This is important because we only want real edges from lane lines to stand out, and want to ignore the noise. Using the `cv2.GaussianBlur(img, (kernel_size, kernel_size)` function, I experimented with different kernels and found that a symmetric kernel of size (3,3) worked well. After that, I applied a grayscale filter using the `cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)` function. 


![alt text][image3]

**Edge detection**
There are many ways to detect edges on an image but one of the most popular is the Canny Edge algorithm. The multi-stage algorithm first computes gradient intensity represenations of the the input image, applies thresholding using a lower and upper boundary on gradient values, and then tracks edges using hysteresis (suppressing weak edges that aren't connected to strong edges). I implemented it using  `cv2.Canny(img, low_threshold, high_threshold)`. The parameters I found to work best were 50 for the lower bound and 300 for the upper bound.

![alt text][image4]

**Window**
The entire image doesn't contain useful information. For instance, the top of the image mostly consists of the sky. Because of that, I applied a region of interest mask to the output of the edge detector that only keeps the area of the image we care about. I used the `cv2.fillPoly(mask, vertices, ignore_mask_color)` function to make the non-important area of the image black. The shape of this mask is a symmetric trapezoid that roughly follows the shape of the lane. It's defined as follows: 

```python
vertices = np.array([[(100,height), (int(width/2) - 80, 325), (int(width/2) + 80, 325), (width - 100,height)]])
```

![alt text][image5]

**Hough lines**
The Hough transform is a feature extraction technique that lets you find line segments on an image. It works by first converting Cartesian coordinates (X, Y) to a paremetric space (slope, intercept). In this space every line represents a point in XY space. Now, if many points form a line in XY space, all these points are lines in "Hough space", and the point at which they intersect represents the slope and intercept that form the line.

To get the line segments I used the function `cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)`. The parameters that I found worked best are threshold = 23, min_line_length = 5, and max_line_gap = 3. I then looped through the line segments and drew them on an empty image as follows.

```python
for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_img, (x1, y1), (x2, y2), color = [255,0,0], thickness=10)
```

![alt text][image6]

**Hough lines (Advanced)**
Our goal was not only to detect and annotate lane segments, but rather to predict the location of the full lanes. To do this, I first looped through the line segments and put each point of the line segments into a left lane or right lane bucket, depending on the slope of the line segment. In addition, I also moved the slopes themselves into left slope and right slope buckets. 

To find the coordinates of the two points that make up the lane, I performed these steps for each lane. First, I found the mean point and mean slope from the colleciton of points and slopes I calculated. Then I used the X-Y coordinates of the mean point, the mean slope, and a Y coordinate (either the bottom of the image for the bottom of the lane or the value of the parameter max_dist which represents how far the lanes go), to extrapolate the two X-coordinates of the two points I need. 


```python
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
```


```python
def extrapolate(x1, y1, m, y2):
    x2 = int(((y2-y1)/m)+x1)
    return x2
```

```python
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
```

![alt text][image7]

**Annotated lane lines**
For the final step, I defined a function that annotates the predicted lane lines on the original image. 

```python
def weighted_img(img, initial_img, a=0.8, b=1., c=0.):
	#img is output of hough lines
	#initial_img  is img before any processing
 
    return cv2.addWeighted(initial_img, a, img, b, c)
```

![alt text][image8]

### Video pipeline
In `pipeline.py`, there are two functions defined. The first, `process_frame(image)` applies all the transformation described above in sequence and can be applied on a single frame. The second function, `process_video(input_path, output_path)`, makes use of video libraries to read videos frame by frame, apply the processing function to each one, and save a video of the output file. 

![alt text][image9]

### Discussion
The pipeline I built is relatively robust in normal conditions. But in conditions where it's dark, there is rain or snow, or the lanes are curved, the pipeline doesn't perform well. 

Improvements that can be made include:
* Making use of different color spaces (HLS, HUV, etc.) to better detect lanes
* Being able to determine the curvature of the lanes if they are not straight.
* Doing distortion correction on the original images to reverse the impact of different lenses. 
* Removing line segments which have abnormal slopes. 













