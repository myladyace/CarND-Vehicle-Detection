##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_notcar_sample.png
[image2]: ./output_images/car_hog.png
[image3]: ./output_images/windows.png
[image4]: ./output_images/hot_windows.png
[image5]: ./output_images/heatmap.png
[image6]: ./output_images/gray_label.png
[image7]: ./output_images/final_image.png
[video1]: ./output_images/processed_project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the In[6] code cell of the Project.ipynb using `extract_function()`, which is from udacity lesson and can be found in lines 7 through 24 and 47 through 95 of the file called `lesson_function.py`  .  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and find below parameters gives good accuracy and training time. One thing need to mention is althogh only use channel-0 already gives good accuracy, but using all channels can still rise the accuracy by 0.01.

color_space = 'YCrCb' 
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL". 

I also include the `bin_spatial()` and `color_hist()` functions which can also be found in lines from 27 to 43 of `lesson_function.py`
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32 

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using all the features provided above (HOG, color histograms and spatially binned color).
The length of all combined features for each image is 8460.
The data are splitted into training and test dataset mannully to avoid time-series issue with fractions of 0.8/0.2.
The accuracy on test set reaches over 98.5%.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at 4 scales of (64,64), (96,96), (128,128), (160,160) with the overlap = 0.75.
The smaller windows were only searched around the horizon while the bigger ones will search from horizon to the more bottom of the image.
All 4 scales of windows are shown in different colors and came up with this:

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 4 scales of windows using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  
I tried to extract HOG features just once for the entire region of interest in each video frame. But since I have 4 scales of windows and all my training was done on (64, 64) size of image(which should contains only 8460 features), the hog features of other 3 scales of windows with size other than (64,64) can't be extracted directly from the one-time extract HOG feature image. So I still extract HOG features from each window, which is inefficient though.

Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_images/processed_project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video of the `test_video.mp4` and the frames can be found in the folder called `test_video_frames`.  I used the frames from frame1 to frame6 to create the heatmap.
From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. The threshold was set to be min(3, len(frames)+1), which means when current frames is the first image, then at least 2 overlapped boxes will give the correct prediction. If there are previous frames, then at least 3 overlapped boxes can decide the positive detection. The maximum number of previous images will be count in is 6.
I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The pipeline worked so slow since I need to extract HOG features for each window and I have 4 scales of windows. How can I extract HOG features only once and directly grab it when we have multiple sizes of windows?


