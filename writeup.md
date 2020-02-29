# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/KeisukeShimokawa/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

Number of train examples = 34799
Number of valid examples = 4410
Number of test examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![](https://i.gyazo.com/3925dfef74c2ecc87cb47b167cd326df.png)


This figure shows the number of images by class included in the training data.


As you can see from the diagram, the data is disproportionate, and when you train the model, you can focus on particular classes and make wrong predictions.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)


As explained in the previous section, the training data is very imbalanced, so we will augment the data to address this problem.

When you augment data, you do not simply copy an existing image, but also add processing such as adding noise to the image or rotating the image.

The code design is based on the implementation of pytorch and albumentations.

```python
class RandomGamma(object):
    def __init__(self, gamma=[0.8, 1.2], p=0.5):
        self.gamma = gamma
        self.p = p
        
        gamma_value = np.random.uniform(gamma[0], gamma[1])
        self.table = 255 * (np.arange(256) / 255) ** gamma_value
        
    def __call__(self, image):
        
        if np.random.rand() < self.p:
            image = cv2.LUT(image, self.table).astype(np.uint8)
            
        return image


class ApplyTransforms(object):
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, image):
        for transform in self.transforms:
            image = transform(image)
            
        return image


transforms = ApplyTransforms([
    RandomRotate([-15, 15], scale=1.1, p=0.5),
    GaussianNoise(0, 5, p=0.5),
    RandomContrast(0.2, p=0.5),
    RandomGamma(gamma=[0.8, 1.2], p=0.5)
])
```

The following shows the number of images by class after augmenting the data.

You can see that the difference in the number of images, which was up to 9 times, has almost disappeared.

![](https://i.gyazo.com/3d65c7069d5c398dbd46a339d778cc39.png)

Below is a sample of augmented data.

If you look at the image, you can see that it is rotated and noise is introduced.

![](https://i.gyazo.com/dd7b8914f9619343cd6ad760a61ebef8.png)

Finally, after limiting the range of pixel values ​​of the training data from 0 to 1, normalization is performed for each channel.

```python
class ChannelNormalizer(object):
    def __init__(self, images):
        images = images / 255.0
        self.means = images.mean(axis=(0,1,2))
        self.stds = images.std(axis=(0,1,2))
    def __call__(self, images):
        images = images / 255.0
        return (images - self.means) / self.stds


normalizer = ChannelNormalizer(X_train)
normalizer.means, normalizer.stds
>>
(array([ 0.33999264,  0.31174879,  0.3209361 ]),
 array([ 0.27170149,  0.2598821 ,  0.26575037]))


X_train, X_valid, X_test = map(normalizer, [X_train, X_valid, X_test])
```


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

A Lenet structure model with Dropout introduced in the linear coupling layer is used.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 28x28x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 10x10x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 10x10x32 				|
| Linear    			| 400 x 120 									|
| Dropout				| 0.5											|
| RELU					|												|
| Linear    			| 120 x 80										|
| Dropout				| 0.5											|
| RELU					|												|
| Linear    			| 80 x n_classes(43)							|

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

| Parameters     | Description   |
|----------------|---------------|
| Optimizer      | Adam          |
| Learning Rate  | 0.0005        |
| Batch Size     | 256           |
| EPOCHS         | 15            |
| Const Function | Cross Entropy |

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

| mode       | accuracy |
|------------|----------|
| Training   | 0.995    |
| Validation | 0.961    |
| Test       | 0.949    |

The trick to achieve the above accuracy is to set the correct learning rate and weight only.

In this case, the training data is limited to values ​​between 0 and 1, so if the learning rate is too high, the weight may not be properly updated every time.

Also, unless the weights themselves were set to a sufficiently small value, learning would not progress well.

### Test a Model on New Images

#### 1. Choose 29 German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web (we have 29 images all):

![](https://i.gyazo.com/32589d911b9a6653de3a440f4954a302.png) ![](https://i.gyazo.com/83be733f6ed4dbbe9730648e06d3cf39.png)

Of the images below, the ones marked "keep right" on the sign may be difficult to classify. This is because the image is not taken directly in front, but is slightly tilted.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

|accuracy|
|-|
|0.9655172413793104|


The model was able to correctly guess 28 of the 29 traffic signs, which gives an accuracy of about 96%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Here we will look at how the model makes predictions for characteristic images.

In the first image, the sign is very inclined, and it is assumed that the shooting environment is different from the learning data.

However, the model is working well, with nearly 90% probability of making correct predictions.

![](https://i.gyazo.com/de3f1aa8a866c7d4a14dceb6900d9079.png)

The second image is the one where the model was predicting correctly, but with a low prediction probability.

You can see that the model cannot predict with such numbers and other images.

![](https://i.gyazo.com/a0799f2e007522dc0e8a9f4a37ae19c4.png)

The last image shows that the model was the only one to make a wrong prediction.

Curiously, even though the image itself is very similar to the second image presented, the model is incorrectly predicting and the probability is very high.

Neural Network is mysterious.

![](https://i.gyazo.com/77519cc35fda1cd9db9103267938007d.png)


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?