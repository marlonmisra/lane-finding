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



### Creating a thresholded binary image
I did exploratory analysis to compare the effectiveness of various techniques. For each technique, I tried various kernels and thresholds. They included:
* absolute sobel threshold (in X and Y directions)
* magnitude sobel threshold
* directional threshold
* RGB thresholds
* HLS (hue/lightness/saturation) thresholds

Ultimately, I found that using a combination of the the HLS threshold and magnitude threshold works the best.

After applying these filters, I also utilized a filter/window to remove the area of the image where lane lines wouldn't be. 

![alt text][image4]
![alt text][image5]

### Perspective transform

The perspective transform changes the image such that you get a bird's eye view. This is important in order to determine lane curvature. 

The method for my perspective transform is called `transform(proc)`. The transformation is done by specifying "source points" and "destination points". Each set of points has 4 unique points and the transformation effectively specified how this 4-sided figure should look in the new space. The source points were manually chosen to be the trapezoid that makes up the main lane area. The destination points were also manually chosen to be a rectangle. My points were the following: 

```python
src = np.float32([(257, 685), (1050, 685), (583, 460),(702, 460)])
dst = np.float32([(200, 720), (1080, 720), (200, 0), (1080, 0)])
```
I then verified that the perspective transformation was working by drawing the source and destination points on a test image and its warped transformation, and ensuring the lines were parallel (left and right lane lines should always be parallel). 

![alt text][image6]
![alt text][image7]

### Identifying lane line pixels and fitting a polynomial

I started by creating a historgram for the buttom half of the transformed image and found the midpoint of the lane by taking the average of the two peaks. 

Then I utilized a sliding window approach to determine the location of the lanes as you go further away form the car. 

Once I had the windows and lane centers, I use the `np.polyfit` function to draw two second-order polynomials on the image to indicate the lane lines. 

![alt text][image8]
![alt text][image9]


### Radius of curvature and lane position relative to car 

The radius of curvature is the radius of a circle that touches a curve at a given point and has the same tangent and curvature at that point. I used standard formulas to calculate the radius on both the left and right lane lines. 

To calculate the lane position relative to the car I compared the center of the image (center of the car) to the midpoint between the left lane and right lane intersections with the bottom of the image. 



### Final image after undoing the transformation 

To undue the transform I used the `warpPerspective` function again but used the source and image points parameters in reverse order. After that I used the `fillPoly` function to color the are in between the lane lines in green. 

![alt text][image10]

### Video pipeline
I created a separate file to process the video, called `process_video.py`. Here I used the moviepy library to read the video, edit it using the process function I defined, and save it. 
The output video file is called `video_annotated.mp4`.

### Discussion
The video pipeleine did a robust job of detecting lane lines, but it didn't perform too great on the challenge project.

In order to make the pipeline even more robust, I need to:
* Explore more ways to process images and apply better filters. There are so many combinations of color spaces and methods to find edges, that there are definitely ones out there which perform better.
* Test the pipelines on more videos to see if it performs well in fog, rain, snow etc.

