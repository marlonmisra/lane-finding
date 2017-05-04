## Lane finding project

### Introduction 
In order to navigate a self-driving car, one of core skills is to be able to detect lane markings and extrapolate them to full lane lines. In this project, I'm  starting with test images and test videos that come from a front-facing camera, and I'm showing a simple way to output predicted lane lines. 

The steps I'll be describing are as follows:
* Applying Gaussian blurs and grayscale transformations to remove the effects of noise and remove unnecessary colors.
* Using a form of edge detection, called the Canny Edge transform, and tuning the parameters. 
* Applying a mask that removes areas of the image that don't provide value.
* Utilizing the Hough transform, a technique that detects line segments.
* Taking individual lane segments, dividing them into left/right lane buckets based on slope, and extrpolating the full lane lines.
* Annotating the original image with predicated lane lines. 


### Files and project navigation 
* test_images and test_videos contain testing data.
* test_images_results and test_videos_results are folders that contain testing data with predicated lane lines.
* functions.py contains transformation functions and helper functions.
* exploratory.py contains parameters and methods to plot test images and transformations.
* pipeline.py contains the video processing pipeline and `process_frame(image)` function which is used on each frame.  

### Pipeline
**Original images**
The test images and frames from the video have the shape (124,23,3), meaning a height of a, a width of x, and 3 RGB channels.

**Gaussian blur**
As the first step, I decided to apply a Gaussian blur to the images. This is important because we only want real edges from lane lines to stand out, and want to ignore the noise. Using the `cv2.GaussianBlur(img, (kernel_size, kernel_size)` function, I experimented with different kernels and found that a symmetric kernel of size (3,3) worked well. 

**Grayscale conversion**
After applying the blur and before finding edges, I applied a grayscale filter because I want the edge detection algorithm to work independent of color. To do this I used the `cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)` function. 

**Edge detection**
There are many ways to detect edges on an image but one of the most popular is the Canny Edge algorithm. The multi-stage algorithm first computes gradient intensity represenations of the the input image, applies thresholding using a lower and upper boundary on gradient values, and then tracks edges using hysteresis (suppressing weak edges that aren't connected to strong edges). I implemented it using  `cv2.Canny(img, low_threshold, high_threshold)`. The parameters I found to work best were 50 for the lower bound and 300 for the upper bound.

**Window**
The entire image doesn't contain useful information. For instance, the top of the image mostly consists of the sky. Because of that, I applied a region of interest mask to the output of the edge detector that only keeps the area of the image we care about. I used the `cv2.fillPoly(mask, vertices, ignore_mask_color)` function to make the non-important area of the image black. The shape of this mask is a symmetric trapezoid that roughly follows the shape of the lane. It's defined as follows: 

```python
vertices = np.array([[(100,height), (int(width/2) - 80, 325), (int(width/2) + 80, 325), (width - 100,height)]])
```

**Hough lines and lane annotation**

















