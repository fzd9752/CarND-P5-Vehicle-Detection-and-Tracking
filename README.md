## Udacity CarND P5 Writeup
### by [Elsa Wang](fzd9752@msn.com)
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
[image1]: ./output_images/carvsnotcar.jpg
[image2]: ./output_images/features_extraction.jpg
[image3]: ./output_images/windows_sliding.jpg
[image4]: ./output_images/find_cars_tests.jpg
[image5]: ./output_images/heatmap_thresholded.jpg
[image6]: ./output_images/thresholds_tests.jpg
[video1]: ./project_max12_threshold_6.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  
---
#### Project files:
* [_Readme.md_](./README.md)
* [_CarND-Term1-Project5.ipynb_](./CarND-Term1-Project5.ipynb) contains project code
* [_output_\___images_](./output_images/) folder contiains example images of the project
* [_output_\__video.mp4_][video1] cars in the video have been detected and tracked

---
### README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the third code cell of the IPython notebook.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then defined a `explore()` function and explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=11`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters, and decided the current parameters by mainly considering two aspects. I take account of the accuracy of the model prediciton as well as the speed of the process. Extraction from three channels of HOG could improve the accuracy obviously, and a bit of larger `pixels_per_cell` could increase at least 2/3 speed with few loss of the accuracy. Also, two more `direction` of the features permits the model to adapt the unruled-shape of the car better.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using features of HOG, spatial and colour histograms (block 9-10). I combined three features and normalise them with `StandardScaler` from `sklearn` library. And then I shuffled and splited preprocessed the data into training and test sets, since test set could help me to evaluate the model I built.

After those, I trained a linear svm and a `rbf` kernel svc with. With the whole data inputing, the evaluation showed that two models performs similar. Even though `rbf` kernel model has better accuracy, the speed of `LinearSVC()` is at least 50% faster. Because this project purposes imitation of the real-world driving car, so I choose to train a Linear SVM model.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I define `find_cars()` function (block 7) to implementing sliding window search and detecting cars positions. I decided to search window from the bottom of the image for excluding the disturbing objects. I searched window positions  at 1.5 scales with 1 step overlapping.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on only one scale using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image3]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result][video1]


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps and resulting bounding boxes are drawn onto the last frame in the series::

![alt text][image5]

Also, I implemented `collection.deque()` to sum and threshold the false positives in the video.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  

I've used SVM machine learning with HOG, spatial and colour histograms features in this project. There are four issues I met in my implementation: first one is about selecting appropriate features to train the model. It does not perform as DL. ML need us to feed in features which a CNN could extract by it self. Thus, the choice and combination of features decide the quality of the whole project. To try the combination could take much time.

Second one is about the multi-scaled windows sliding. Though it's a powerful approach to detect the different size objects in the image, it also took long time to process the image. This makes it almost impossible to implement in a real-world, which needs the pipeline to reflect instantly.

Third problems of my project is the false positive. `label` and `heatmap` are very useful and convenient tools to draw the box. However, it could not provide a satisfied result to handle the detailed noise.

Fourth, the balance between time and accuracy. I practiced this project all under the assumption of real-world implementation. Most of time, I had to tradoff the accuracy to the time. Also, with a general laptop, to experiment a video is exhausting. It's really slow to execut the pipeline.

#### 2.Where will your pipeline likely fail?  What could you do to make it more robust?

The most dangerous part is the detection, especially when a car entering the view. Because at entering moment, the car can't show the whole image, the trained model could detect it only after the most of the car entering the view. To feed the model a more completed and variable data might improve its performance.

Another dangerous parts of my pipeline located in fail positives on the left side. It mainly caused by the coming cars on the opposite driving side. In the future implementation, the opposite side of the car could be consider to eliminate by add `x_start` and `x_stop` variable into sliding windows code.

One technique I'd like to try in the next -- multi-scaled sliding. It seems a reasonable approach to find a ratio between the window size and distance to the camera.