# **Behavioral Cloning**

<!-- ##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer. -->

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/mudd.png "muddy"
[image2]: ./examples/center.png "center"
[image3]: ./examples/recover1.png "Recovery Image1"
[image4]: ./examples/recover2.png "Recovery Image2"
[image5]: ./examples/recover3.png "Recovery Image3"
[image6]: ./examples/track2.png "Track 2 Image"
[image7]: ./examples/flipped.png "Flipped Image"
[image8]: ./examples/predict.png "predict"
[image9]: ./examples/camera.png "camera"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results
* (additional) model.ipynb (html snapshot is also attached) is a notebook file containing the main code of model.py as I found it more convenient to work on jupyter than directly running python code．Thus, it is attached for refernece only.

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. Particularly, the arguments should be passed to initiate the training:
```sh
python model.py -f foldername -e epcoh_num -l lr_rate
```
It should be noticed that I did most of the work using jupyter as I found it much more convenient as compared to running the code directly. `model.ipynb` is attached for reference.

### Architecture and Training Documentation

#### 1. An appropriate model architecture has been employed

In this project, I employed the Nvidia referenced model as it has been proven effective. The code can be found in model.py lines 109-121. My model structure is listed sequentially as:
* lambda layer `model.add(Lambda(lambda x:x / 255.0 - 0.5,input_shape=(160,320,3)))`:
This layer is to normalize the input image.
* cropping layer `model.add(Cropping2D(cropping=((50,25), (0,0))))`: By cropping the upper and lower parts of the input image, the model can focuses more on the road shapes and features without interference from the background.  
* convolution layers * 3 `model.add(Convolution2D(filter_depth, 5, 5, subsample = (2,2), activation='relu'))`:
For the first part, the filter size is 5x5 and the depths of these three layers are 24, 36, and 48. Also, subsampling of 2x2 is implemented with `relu` activations.   
* convolution layers * 2 `model.add(Convolution2D(filter_depth, 3, 3, activation='relu'))`:
For the second part, the filter size is 3x3 and the depths of these three layers are 64 and 64. No subsampling but with `relu` activations.   
* dense layers * 4 `model.add(Dense(filter_size, activation='relu'))`: For the FC layers, the filter sizes are 100, 50, 10, and 1. `relu` activations are implemented.

#### 2. Attempts to reduce overfitting in the model

In this project, I do not implement dropout layers as suggested by the Paul Heraty that the performance depends mostly on the data collection than the model architecture. Although I was not fully convinced in the beginning, the development process shows that the Nvidia model does quite a good job. For reducing overfitting, the model was trained and validated on different data sets to ensure that the model was not overfitting (code line 38-40). Thus, the number of epochs is controlled and the validation accuracy is carefully monitored during training to reduce oversitting.

#### 3. Model parameter tuning

Although the adopted adam optimizer is less sensitive to the learning rate, I did slightly adjust the learning rate. When a large dataset was used a learning rate of 5e-4 was used, whereas a smaller learning rate of 1e-5~1e-4 was used when fine-tuning the model with a small dataset.   

#### 4. Appropriate training data

In the first attempt, I followed the instructions from the project guideline by collecting driving data:
1. Three laps of keeping the vehicles in the middle, 2 forward and 1 reverse.
2. One lap of recovery driving by hitting the side of the road and recovering.
3. Two laps of driving smoothly around the curves.

Then, I started fine-tuning the performance of the model by getting more data including data augmentation. More details are discussed in the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to progressively enhance the model capacity. My first attempt was to use a two-layer model with one convolution layer and one dense layer. This model is built to verify the workflow including data collection, model training, model saving, and autonomous driving test. A small dataset of 5k samples was used in order to test the workflow. Of course, the performance of the small network was not very satisfactory. So I gradually increased the number of layers to gain model capacity. Once I have more data, I started to use the Nvidia model and collected a dataset of the size of ~40k data points (the details of augmentation are given in the next subsection). For combatting over-fitting, I used 20% of the dataset for validation and carefully controlled the number of epochs to ensure that the validation accuracy was not too far off the training accuracy.

I think that an efficient workflow is important to improve the model performance. Running the simulator to verify the performance is essential but it also consumes lots of time. So I decided to record the images from the autonomous mode and picked the images that capture the moments when failure happened. I then used those images to serve as an initial test for the new model. If the prediction is good, then I will proceed to run the simulator. This workflow really helps a lot and reduces the turn-over time significantly.

After quite a lot of trials, I finally obtained a model with fine performance; however, I noticed that the model always at the edge of passing a corner, shown as below:

![alt text][image8]

No matter how I fine-tuned the model, the vehicle could not pass the curve in a smooth way. So I examined the real-time prediction from the model and found that the prediction using drive.py was way different from my code even though the same image was used (as can be seen from the image above, the prediction from my model shows -0.703). Then I realized that drive.py uses PIL module to load the image in RGB format while cv2 loads in BGR. After resolving this inconsistency, my model worked smoothly as expected. Moreover, I have also tried to fine-tune the model by freezing some layers (model.py line 131-136).

To enhance the robustness of the model, I also collected some data from the second track. At the end, the performance for the second track was good with small unsmoothness at some corners.

At the end of the process, I obtained around ~240k of original and augmented samples over the initial and fine-tuning processes. Therefore, the vehicle is able to drive autonomously around the first track without leaving the road as recorded in video.mp4.

#### 2. Final Model Architecture

The final model architecture (model.py lines 109-121) was based on the Nvidia reference model. The details have been revealed in the previous section.

#### 3. Creation of the Training Set & Training Process

I first recorded three laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to turn when facing an edge. These images show a recovery attempt on the bridge:

![alt text][image3]
![alt text][image4]
![alt text][image5]

I have also collected data from track 2:

![alt text][image6]

Then I repeated this process on track two in order to get more data points. In total, I think 80% of data points are from track one and 20% from track two. Then I flipped the image to balance the model better and also to get more samples, here is an example:

![alt text][image7]

I have also used the images from right and left cameras, here are some example images:

![alt text][image9]

Then the perspective adjustment is done by adding an compensation term to the steering angles:

```sh
correction_left = 0.25
correction_right = 0.25
left_angle = float(batch_sample[3]) + correction_left
right_angle = float(batch_sample[3]) - correction_right
```

I kept the possibility of unsymmetrical compensation for fine-tuning the model. I have also added a variable to amplify the steering angle if needed:

```sh
amp_factor = 1.0
y_train = np.array(angles)*amp_factor
```

These variables were generally not changed unless when there were obvious issues with the model performance such as staying too much to the left, failing a sharp turn .. etc.


After the collection process, I had data points of ~40k and augmented samples of ~240k (6x). I shuffled the data with two different random lists for flipped and non-flipped images to avoid the coupling between them so that an image is not always trained with its flipped replica. In general, 20% of the samples are used for validation. It should be addressed that these 240k samples were not used at once but collected progressively throughout the fine-tuning process.

Since I did not implement particular methods for prevent over-fitting, I was always careful with the number of epochs and the validation accuracy. It seems like the model can perform quite well even without dropout layers, as recorded in video.mp4.

### Simulation
The following code can be used to test the model.
```sh
python drive.py model.h5
```
I modeifed some code in drive.py to gain better driving performance; particularly I added the following lines:

```sh
speed_ref = float(set_speed)*(1.0 - min(abs(steering_angle) * 0.8, 0.5))
controller.set_desired(speed_ref)
```
The idea is to brake the vehicle proportionally to the steering angles (it makes senses to brake when driving through a curve)

I have also added saturation for the integrator output of the PI controller. This helps to avoid controller wind-up effect as can be seen a lot from the original one.

```sh
if self.integral > 500:
    self.integral = 500
```
