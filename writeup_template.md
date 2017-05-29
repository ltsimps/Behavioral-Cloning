## Approach
Initially, I followed the approach that created a small model By training with my collected data and improving the model. The best results were  achieved with VGG type network, where there are multiple convolution layers with a small 3x3  kernel following a maxpool layer that reduces the dimensions, but 2x2 convolution layers with 3 fully connected layers  gave better results, 

## Challenges
A number of challenges were faced during the project:
### Input Data gathering
Input data proved to be more significant than initially expected and took longer on AWS becuase it took two days to train and I"m not sure why.. My expectation was that it would take an hour or two with the GPU since it took over three days to train on my local machine. )


### Number of various parameters
The model consisted on many parameters that each affected the model behavior when simulating (number or epochs, model architecture, layer sizes, etc). The challenge with that was, that there was no good measure how to evaluate model accuracy because even if validation set was performing well, it did not guarantee predictable behavior in all track locations, it ensured that for most of the track it behaves better.

## Model network architecture
Initially, several models were tested and one of the best results was with the above model. Later 3rd convolution layer was added as well as intermediate connections to first fully connected layer.

The first layer of the network was 3x1x1 convolution that I used also on the traffic sign classifier, in order for the model to be able to adjust parameters for the colorspace. As the conversion between various color spaces, e.g. RGB to HSL is a multiplication of each channel to get the new color channel then this layer is added for the model to train such behavior if necessary.

### Why VGG type network
I like the VGG type network when there are convolution layers that have small kernel and does not affect dimensions with a mix of maxpool layers to reduce dimensions because the model is quite straight forward in terms of input and output shapes. Each high level in this network reduces dimensions by two and has inside several convolution layers.<br/> For me, this model worked very well on traffic sign classifier and also turned out to have good results on this assignment.<br/>
One consideration is that as the input image was not of the dimensions of power of 2, it had a reduction from a dimension of 5 to a dimension of 2 in last maxpool layer. The loss of data was somewhat compensated with stacking also intermediate layers.

### Why stacking intermediate layers
Results from the intermediate maxpool layers also were passed to final fully connected layer. This was done in order not to loose valuable features if there are such in the middle of the network. This gave improved results, especially in the left turn after the bridge. After maxpool layer data is passed to next convolution layer and as well it is flattened and passed to first fully connected layer.

### Why Adam optimizer
Also in this assignment Adam optimizer was chosen, because it has fewer hyperparameters to tune). I am still at the stage when I am learning models and generic approach and even without such hyperparameters, there is a lot of other parameters to tune.I am still in a place, where more coudl be done reliably, but do to time constraints I'm satified with the current peformance..

### Number or epochs
In the final solution, a number of epochs are 5. Larger values that that, started to overfit the model and in the simulator it meant that it preferred to drive straight, therefore starting to miss sharp turns. Less that that and simulator model oscillated more from side to side over the road.

### Data cleanup
From the Udacity dataset, there was some additional cleanup made in the regions where model wanted to behave not according to intention. I checked the images where there is right turn with the red and white borders (to improve results on the 2nd corner after the bridge), for those images, if the steering angle was 0, then they were removed completely from the input dataset.<br/>
Also similar was done to the images for 1st left turn after the bridge, if the angle was 0 then they were removed. This allowed to navigate those corners more sharply and drive the track.

### Camera selection
Initially, the model was trained on center camera, however, it was oscillating quite a lot in the middle of the track and there was very poor recovery. I decided to add also left and right image, with the angle adjustment of 0.08.
This parameter was tuned by trial and error. Initially chosen a larger parameter, and then lowered it so that critical corners are still navigated successfully. The larger the parameter the more oscillation also on the straight track, but the recovery is improved.<br/>
Also tested if only the left and right camera is used without the central camera. This turned out to have the best results. The oscillation was lower that training with all 3 cameras and much better recovery. My hypothesis is that this is because there is a lot of camera data with 0 steering angle, but slightly different car alignments to the track axis. This slightly varying data confuses model training and provides more erratic weights. If there is only left and right camera data with angle adjustment of +/-0.08, then there is almost no data with 0 steering angle.<br/> Possibly this could be also done with altering initial dataset and removing 0 steering data, but that would require additional data collection.

### Region selection and downsampling
Vertical region from above car and below around the horizon was chosen with full width.<br/>
Also the each second pixel was taken in order to reduce image dimensions.

### Normalization
Input RGB images of uint8 values of 0-255 were normalized to -1.0 to 1.0 floating values. I also observed that if the values are normalized then training performs better (gets accuracy quicker) as the weights are more uniform.

## Reflection
This was a time consuming project due to training issues. Also learning abotu cconvolutional networks and how to  operate them was challenging.<br/>
I value this a lot as a learning experience and that I was able to reach a result of driving the lap successfully. This model definitely could be improved and especially with the additional data that can be collected and properly annotated.<br/>
