**Vehicle Detection Project**
Autor: Chun Hsien(Steve) Ho (chunhsien@umich.edu

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./report_file/hog_HLS.png
[image2]: ./report_file/hog_Lab.png
[image3]: ./report_file/hog_Luv.png
[image4]: ./report_file/hog_RGB.png
[image5]: ./report_file/hog_YCrCb.png
[image6]: ./report_file/hog_YUV.png
[image7]: ./report_file/color_hist.png
[image8]: ./report_file/heatmap.png
[image9]: ./report_file/sliding_window.png


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the Classifier_for_vehicle.ipynb

I started by reading in all the `vehicle` and `non-vehicle` images using glob library and extract HOG features using hog function from skimage.features. Then explored different color spaces and different skimage.hog() parameters (orientations, pixels_per_cell, and cells_per_block). I grabbed random images from each of the two classes and displayed them to get a feel for what the skimage.hog() output looks like.
Here is an example about visualize histogram

![alt text][image7]

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:
![alt text][image1]
![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]



####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters of orient,pixel per cell cells per block by using different color spaces



####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I tried to train SVM classifier using only HOG features across different color spaces and different parameters.



###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

TODO(Olala): identify where's the code The window search is defined as the find_car function in the 3rd code cell of SlidingWindow.ipynb. After some trail and error, I decided to search wth window for 4 different scales

# ystart, ystop, scale, cells_per_step, color
searches = [
    (380, 500, 1.0, 1, (0, 0, 255)),  # 64x64
    (400, 600, 1.587, 2, (0, 255, 0)), # 101x101
    (400, 710, 2.52, 2, (255, 0, 0)),  # 161x161
    (400, 720, 4.0, 2, (255, 255, 0)), # 256x256
]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

| Color Space |(9,8,1)| 9,8,2 | 9,8,3	| 
| YUV         | 0.9037| 0.9502| 0.8992| 
| HLS         | 0.8688| 0.9079| 0.8663| 
| YCrCb       | 0.9015| 0.9521| 0.8933| 
| Luv         | 0.9006| 0.9513| 0.8998|
| RGB         | 0.9144| 0.9417| 0.8994| 
| Lab         | 0.9093| 0.9550| 0.9091| 


I change the HOG features and also the color histogram and spatial bin features. I also tried on different number of histogram bins and different spatial bin resolutions to get the best result and reduce number of features required.

Here is the result for different size of spatial bin

We could find that the 16x16 is the best result 

| Size |  32x32	| 16x16 | 8x8	  | 
| YUV   | 0.9037| 0.9198| 0.8992| 
| HLS   | 0.8688| 0.8992| 0.8663| 
| YCrCb | 0.9015| 0.9172| 0.8933| 
| Luv   | 0.9006| 0.9150| 0.8998|
| RGB   | 0.9144| 0.9302| 0.8994| 
| Lab   | 0.9093| 0.9248| 0.9091| 




Here is the result for different nbins

We could find that the 128 is the best result

| Color |  128	| 64	  | 32	  | 
| YUV   | 0.9307| 0.9226| 0.9065| 
| HLS   | 0.9634| 0.9575| 0.9414| 
| YCrCb | 0.939	| 0.9350| 0.9310| 
| Luv   | 0.9378| 0.9319| 0.9153|
| RGB   | 0.9209| 0.9167| 0.9099| 
| Lab   | 0.9566| 0.9459| 0.9330| 



Result from the HOG,spatial bin and color histogram

| Color space | Feature extraction | Training time | Predict Time | Accuract |
|-------|-------|-------|-------|-------|
| YCrCb | 99.84 | 6.08 | 0.16 | 0.9918 |
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. To get a robost result, I use a deque to record the heatmap of the last 10 frames and sum them in a exponential decay fashion before applying the threshold. Finally, I assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

The code is in the 5th code cell of P5.ipynb)

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:



### Here is a image of sliding window of 4 different scales 
You coule see there is a false positive

![alt text][image8]



### Here is the image of heatmap and the result to detec the bounding boxes
![alt text][image9]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

