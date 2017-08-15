
Driving Cars with Neural Networks
Lamar Simpson
The Promise of Self Driving Cars
Ever since the 1920s the dream of cars being autonomous has been alive and well. Experiments first began by trials with radio impulses be transmitted to antennas that were connected to circuit-breakers which controlled the electric motors that directed the car and their movements. In the 1950s other experiments were performed were electronic circuitry was placed in the ground for cars to follow and was eventually tested on a 400 strip of a public highway. These weren’t the last of these types of experiments as companies started to bid for the transformation of the roadways with embedded electronics in the 1960s that was thought could revolutionize the transportation system. Bids were even put forth that this electronically controlled highway would somehow allow the possibility of controlling hover cars. Although a cost benefit analysis was shown that the expense to build the new road would be paid back by the end of the century, funding for the effort dwindled and ultimately the effort was halted. Vision guided systems were next deployed in cars by Mercedes and their vehicle could achieve a speed of 39 miles per hour with no traffic on the road. This is also the year DARPA started to fund autonomous car efforts which came to be known as the DARPA grand challenge. Most of the autonomous cars that would compete were almost vision based systems that used technologies like LIDAR to assist in steering tough terrain. From the 1980s to 2010 funding for autonomous car was largely sponsored by the government as other funding sources seemed to dwindle as interest dwindled. The interest was reinvigorated by better technologies, ambitious companies and the explosion in the interest in machine learning. The hope and the promise is that by automating these machines we will prevent accidents and save lives. Some believe that the technology is almost there and data is being collected to replicate what human drivers do to help the cars learn faster. In this project, I will seek to use neural networks to try to control the actions on a car based on the steering angle. 

“I think the brain is essentially a computer and consciousness is like a computer program. It will cease to run when the computer is turned off. Theoretically, it could be re-created on a neural network, but that would be very difficult, as it would require all one's memories. Stephen Hawking “

What are Neural Networks
Collecting Enough Data

The first step in working with any neural network is collecting enough data to give the model that will be created from the neural network to be able to give accurate predictions. Machine learning involves trying out ideas and testing them to see if they work. If the model is over or underfitting, then I will have to experiment and adjust to make things work. Since this model outputs a single continuous numeric value, one appropriate error metric would be mean squared error. If the mean squared error is high on both a training and validation set, the model is underfitting. If the mean squared error is low on a training set but high on a validation set, the model is overfitting, so collecting more data can help improve a model when the model is overfitting.  Below I show data for a sample size of 5799 images that have different steering angles associated with them as the car moves around the track shown below
.   
   
    





As you can see both the training loss and the validation loss go down as the model I created trains for longer time.  Unfortunately, at this point with 5799 training examples the model still seems to know be generalizing correctly. Often the model will make it around the track with no problems, but on occasion it will go off the track completely. Examples of this are given below, but keep in mind this is models running at four or less epochs with 5799 training examples.

 








Successful Training
Even though the model starts to overfit on the validation day because it seems like the model converges after five epochs the training loss continues to drop while validation seems to peak. It actually got worse on the validation set after five epochs which is why it has an upward trend. I believe I can write some more code to do more cross validation to try to fix this but in the interest of time I had to settle for the trained models I have now. The graph below shows the training over 10 epochs using 20 thousand images.



Approach
Activation functions 
In most cases, using the sigmoid function as the activation function for hidden units in a neural network is preferred especially if the network that is being used for classification on the output unit. However, this is not the only activation function that can be used and unfortunately sigmoid and others has some drawbacks.



The reason why I chose not to go with sigmoid is because the derivative of the sigmoid maxes out at 0.25 (see above). This means when you're performing backpropagation with sigmoid units, the errors going back into the network will be shrunk by at least 75% at every layer. For layers close to the input layer, the weight updates will be tiny if you have a lot of layers and those weights will take a really long time to train. Because this, I chose to go with something simpler.

Rectified Linear Units
Instead of sigmoids, most recent deep learning networks use rectified linear units (ReLUs) for the hidden layers. A rectified linear unit has output 0 if the input is less than 0, and raw output otherwise. That is, if the input is greater than 0, the output is equal to the input. Mathematically, that looks like
f(x)=max(x,0).
The output of the function is either the input, x, or 0, whichever is larger. So, if x=−1, then f(x)=0 and if x=0.5, then f(x)=0.5. Graphically, it looks like:



ReLU activations are the simplest non-linear activation function you can use. When the input is positive, the derivative is 1, so there isn't the vanishing effect you see on backpropagated errors from Sigmoids. it has been shown that ReLUs result in much faster training for large networks. I figured that since I will be using a Convoultional network that could potentially have many layers and lots of data ReLUs would be the logical to go with.

Drawbacks
It's possible that a large gradient can set the weights such that a ReLU unit will always be 0. These "dead" units will always be 0 and a lot of computation will be wasted in training.
Unfortunately, ReLU units can be fragile during training and can “die”. For example, a large gradient flowing through a ReLU neuron could cause the weights to update in such a way that the neuron will never activate on any datapoint again. If this happens, then the gradient flowing through the unit will forever be zero from that point on. That is, the ReLU units can irreversibly die during training since they can get knocked off the data manifold. For example, as much as 40% of the network can be “dead” (i.e. neurons that never activate across the entire training dataset) if the learning rate is set too high. Although, people have written that the learning rate plays a factor and with a correct setting of the learning rate this is less frequently an issue.


Input and Data gathering
Input data proved to be more significant than initially expected and took longer on AWS because it took two days to train or 10 epochs which was surprising. I’m using tensor flow and I didn’t compile it from scratch which could be the reason the training took so long. The initial dataset was only 5779 and less. 
My expectation was that it would take an hour or two with the GPU since it took over four hours to train on my local machine. There were also some drawbacks using the software tools I used. One Library I used is called Keras and it makes it easy to experiment and implement different neural networks, but after the model is trained it seemed to take 2x the time to actually save the model so I could experiment with it and drive the car. 
Number of various parameters
The model consisted on many parameters that each affected the model behavior when simulating (number or epochs, model architecture, layer sizes, etc). The challenge with that was, that there was no good measure how to evaluate model accuracy because even if validation set was performing well, it did not guarantee predictable behavior in all track locations, it ensured that for most of the track it behaves better.
Neural Network Model network architecture
Model architecture and training
The CNN I chose is a pretty standard CNN consisting of 4 convolutional layers with ReLU activations, followed by two fully connected layers with dropout regularization. Finally, a single neuron formed the output that predicted the steering angle. One noteworthy thing is the absence of pooling layers. The rationale behind avoiding pooling layers was that pooling layers make the output of a CNN to some degree invariant to shifts of the input, which is desirable for classification tasks, but counterproductive for keeping a car in the middle of the road. I trained the network on an Ubuntu 14.04 system using an NVIDIA GTX 1080 GPU and I also trained on AWS using a g2.xlarge but encountered issues below. For any given set of hyper parameters, the loss typically stopped decreasing after a few epochs. My model architecture which was implemented in Keras as shown below.




Initially, several models were tested and one of the best results was with the above model. Later a third convolution layer was added as well as intermediate connections to first fully connected layer.
The first layer of the network was 3x1x1 convolution that was used on the initial 5799 images, in order for the model to be able to adjust parameters for the color space. As the conversion between various color spaces like RGB to HSL is a multiplication of each channel to get the new color channel then this layer is added for the model to train such behavior if necessary.
Why VGG type network
The VGG type network when there are convolution layers that have small kernel and does not affect dimensions with a mix of maxpool layers to reduce dimensions because the model is quite straight forward in terms of input and output shapes. Each high level in this network reduces dimensions by two and has inside several convolution layers.
For me, this model worked very well also turned out to have good results on this project.
One consideration is that as the input image was not of the dimensions of power of 2, it had a reduction from a dimension of 5 to a dimension of 2 in last maxpool layer. The loss of data was somewhat compensated with stacking also intermediate layers.
Why stacking intermediate layers
Results from the intermediate maxpool layers also were passed to final fully connected layer. This was done in order not to lose features that might be useful if there were some that were in the middle of the network. This gave improved results, especially in a left turn after the bridge.  Although, the part of the track that contained dirt seemed to throw the model off, but I believe that’s because I didn’t drive enough close to the dirt part of the road. After maxpool layer data is passed to next convolution layer and as well it is flattened and passed to first fully connected layer.
Why Adam optimizer
Also, I chose an Adam optimizer, because it has fewer hyper parameters to tune. I am still trying to learn and experiment with models and see what generic approaches are useful and even without hyper parameter. There are a lot of other parameters to tune and more can be done to make the models more reliable, but do to time constraints and the amount of time it takes to train with each epoch while also giving good performance on the track I'm satisfied with the current performance.
Number or epochs
In the final solution, a number of epochs that seemed to work out turned out to be five. Larger epochs made the model start to overfit unless the data was increased drastically for input for the model and in the simulator, it meant that it preferred to drive straight, therefore starting to miss sharp turns. Less than that and simulator model oscillated more from side to side over the road.

Data cleanup
I generated my own dataset driving the car in simulation but there was some additional cleanup made in the regions where model wanted to behave not according to intention. I checked the images where there is a right turn with the red and white borders (to improve results on the 2nd corner after the bridge), for those images, if the steering angle was 0, then they were removed completely from the input dataset.
Also similar was done to the images for 1st left turn after the bridge, if the angle was 0 then they were removed. This allowed to navigate those corners more sharply and drive the track.
Camera selection
Initially, the model was trained on center camera, however, it was oscillating quite a lot in the middle of the track and there was very poor recovery. I decided to add also left and right image, with the angle adjustment of 0.08. This parameter was tuned by trial and error. Initially chosen a larger parameter, and then lowered it so that critical corners are still navigated successfully. The larger the parameter the more oscillation also on the straight track, but the recovery is improved.
Also tested if only the left and right camera is used without the central camera. This turned out to have the best results. The oscillation was lower that training with all 3 cameras and much better recovery. My hypothesis is that this is because there is a lot of camera data with 0 steering angle, but slightly different car alignments to the track axis. This slightly varying data confuses model training and provides more erratic weights. If there is only left and right camera data with angle adjustment of +/-0.08, then there is almost no data with 0 steering angle. Possibly this could be also done with altering initial dataset and removing 0 steering data, but that would require additional data collection.
Region selection and down sampling
Vertical region from above car and below around the horizon was chosen with full width.
Also each second pixel was taken in order to reduce image dimensions.
Normalization
Input RGB images of uint8 values of 0-255 were normalized to -1.0 to 1.0 floating values. I also observed that if the values are normalized then training performs better (gets accuracy quicker) as the weights are more uniform.

Data Augmentation:
Recovery Laps
If you drive and record normal laps around the track, even if you record a lot of them, it might not be enough to train your model to drive properly.
As stated above some of the problems were not enough data collection and getting enough so there are lots examples for the car being on the track at different steering angles. The main problem was that my training data is all focused on driving down the middle of the road, so my model wouldn’t ever learn what to do if it gets off to the side of the road. As you saw earlier pictures my car decided it wanted to drive off road.  Some things I did was to constantly wander off to the side of the road and then steer back to the middle. This didn’t work as well as I would have liked and I found a better approach is to only record data when the car is driving from the side of the road back toward the center line.
In addition, I found that driving counter-clockwise also helped me on the turns that the model seemed to be left or right turn bias.  I found that If you only drive around the first track in a clock-wise direction, the data will be biased towards left turns. Driving counter-clockwise is also like giving the model a new track to learn from, so the model would generalize better.

Software Tools
Below are the software tools I used to complete the project.

Udacity Simulator:
This Simulator was written by Udacity in unity and gives two options to train the self-driving car on. Clicking on Training mode gives me the option to use the arrow keys are the a,w,s,d keys to control the car. There is a record button that allows you to pick a saved location for the data for that session. This feature came in handy because It allowed to mix and match track data to generalize over both tracks as well has have different training sizes in different folders to test the model. I didn’t have as much time to train on the darker more difficult track but training on it helped on track one. Below are some images to show how this tool is used. 










In addition to simulator I used standard tools for building neural networks like numpy which helped with doing matrix calculations. OpenCv which helped with processing and preprocessing images. Sklearn which helps with providing models for neural networks and other useful functions that can be used for learning and Keras which makes it a lot faster to build and experiment with neural networks. Finally, I used Matplotlib to generate the plots above. 
Using Both Tracks
Unfortunately, I couldn’t train on the second track a lot because of the amount of training that was involved so my model doesn’t work as well as it should on it right now. The second track contains more sharp turns, hill, shadows and cliffs. I believe that cropping the images to augment the data might not work as well since there will likely be cliffs in every shot. 


Challenges and Reflection 
A number of challenges were faced during the project:
Most of the challenges were related to the sheer amount of time it takes to train the models and see results. Even with only 20k samples it took hours for the model to be trained on them and additional hours for the model to be saved by the library I was using. In some cases, I even got a result where the training and validation loss were low but the car was still driving off the track for low sample sizes.  I figured that was because the model was overgeneralizing and I just didn’t have enough data. 
Also at the end of the day the model is as good as I drive around the track, and I using the arrow keys to stay on the track was difficult for me. I got into the habit of going top speed so it easy for me to pollute the data set producing images and angles that resulted in the car going off the track. Ultimately, I really value this project a lot as a learning experience and that I was able to reach a result of the car being able to drive around track successfully. This model definitely could be improved and especially with the additional data that can be collected and properly annotated. Also, there could be more layers added and dropout could be added to possibly get better performance because this would force the model to learn redundant representations. There are definitely better models out there like the NVdia  model, but I think making my own network even though it was inspired by VGG gave me lots of insights to neural networks and the potential of self-driving cars. 







References:
1.	 "'Phantom Auto' will tour city". The Milwaukee Sentinel. Google News Archive. 8 December 1926. Retrieved 23 July 2013.
2.	[1]  R.M.BellandY.Koren.Lessonsfromthenetflixprizechallenge.ACMSIGKDDExplorationsNewsletter, 9(2):75–79, 2007. 
3.	Andrej Karpathy's CS231n course
4.	Udacity Self driving Nanodegre program https://www.udacity.com/drive
5.	



