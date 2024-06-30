# sign-language-recognition-using-CNN
This repository is the extension of the article "Sign language recognition using Convolutional Neural Networks" written by Kobiela et al. [CITE] and contains the detailed report about the literature review and details about the usage, testing and
implementation of the mobile application. 

## Code
> Mobile application developed for the research: https://github.com/adartemiuk/ASL_recognition_app

> Models source code and training scripts: https://github.com/adartemiuk/ASL_recognition_models

## 2. Background and related work
[TODO: Dorzucić też podsummowanie przeglądu literatury (na co najmniej 2 strony - streszczenie przeglądu literatury z magisterki)]

### List of compared models performing gesture recognition tasks: 

| Author  | Model | Gesture type | Sign language type | Accuracy |
|------------- | ------------- | ------------- |  ------------- |  ------------- | 
| Suharjito et al. [[1]](#1) | i3D Inception  | dynamic |  Argentinean |  100% |
| Makarov et al. [[2]](#2) | QuadroConvPoolNet  | static |  American  |  about 100% |
| Content Cell  | Content Cell  | Content Cell |  Content Cell |  Content Cell |
| Content Cell  | Content Cell  | Content Cell |  Content Cell |  Content Cell |
| Content Cell  | Content Cell  | Content Cell |  Content Cell |  Content Cell |
| Content Cell  | Content Cell  | Content Cell |  Content Cell |  Content Cell |

One of the most popular sign language datasets, frequently used on the Kaggle platform, is a set of static gestures developed by researchers inspired by the MNIST datasets [44]. It consists of 24 gesture classes representing the letters of the American Sign Language alphabet (A-Z). The dataset does not include classes representing the letters J and Z, as these letters are represented by dynamic gestures. The training set comprises 27,455 samples, while the test set contains 7,172 images. The image dimensions are 28x28 and are represented in grayscale. Another popular set of static gestures from Kaggle is the ASL Alphabet dataset [45]. This collection includes 29 gestures from American Sign Language (26 alphabet gestures) and 3 word gestures ("space," "delete," "nothing"). The training set comprises 87,000 images, while the test set consists of only 29 samples. The Massey dataset, developed by Barczak et al. [2011], is also one of the more popular datasets for training models to recognize static gestures. The latest version of the dataset consists of 2,524 images of American Sign Language gestures. The dataset includes 36 different gestures, representing both the alphabet and numbers. All images are in color, in PNG format, and feature an isolated hand displaying the gesture against a black background. Regarding available dynamic gesture datasets, one of the most frequently used is the Jester dataset, developed by Materzynska et al. [2019]. The dataset consists of 148,092 three-second videos of people performing simple gestures. The dataset includes 5 classes that can be categorized as static gestures and 20 classes representing dynamic gestures. Additionally, there are two classes that indicate no gesture action. The gestures were captured with the help of 1,376 volunteers who performed them in front of a camera. The data is divided into training, test, and validation sets in a ratio of 8:1:1. When splitting the data, care was taken to ensure that videos from a given volunteer did not appear in both the test and training sets. Each video contains 12 frames, and the resolution is 100px. LSA64, prepared by Ronchetti et al. [2016], is a dataset of dynamic gestures from Argentine Sign Language. It consists of 3,200 video clips in which ten volunteers perform five repetitions of each of the 64 most frequently used gestures. The dataset includes 42 gestures performed with one hand and 22 gestures performed with both hands. Each video contains 60 frames, and the resolution is 1920x1080. The WLASL (Word-Level American Sign Language) dataset, created by Li et al. [2020], is one of the largest datasets in terms of the number of words and samples per gesture. The authors divided the dataset into four subsets, each varying in the number of gestures: WLASL100, consisting of about 2,000 videos; WLASL300, containing over 5,000 videos; WLASL1000, with approximately 13,000 samples; and WLASL2000, which contains about 21,000 samples. The dataset was created using educational websites and videos from YouTube. RWTH-BOSTON-400 is a dataset of American Sign Language gestures developed by Zahedi et al. [2005], which is a subset of a larger dataset from Boston University. It contains 843 different sentences, composed of 483 words. The gestures are performed by four different individuals. Other subsets developed by the same institute include RWTH-BOSTON-50, RWTH-BOSTON-104, and RWTH-BOSTON-Hands. All the datasets described above, along with their key features, are summarized in the table below:

### List of the most frequently used datasets for training sign recognition models

| Dataset  | Source | Gesture type | Sign language type | Number of gestures | Number of samples |
|------------- | ------------- | ------------- |  ------------- |  ------------- | ------------- | 
| MNIST | Kaggle [44]  | static |  American |  24 | 34,637 |
| ASL Alphabet |Kaggle [45]  | static |  American  |  29 | 87,029 |
| Content Cell  | Content Cell  | Content Cell |  Content Cell |  Content Cell | Content Cell |
| Content Cell  | Content Cell  | Content Cell |  Content Cell |  Content Cell | Content Cell |
| Content Cell  | Content Cell  | Content Cell |  Content Cell |  Content Cell | Content Cell |
| Content Cell  | Content Cell  | Content Cell |  Content Cell |  Content Cell | Content Cell |

## 3.2. Model
The network's input consists of one convolutional layer containing 96 filters with dimensions of 11 x 11, which processes images of size 224 x 224 x 1. Additionally, a stride of 2 x 2 is used (which helps reduce the parameters of the initial layers). For each layer of the network, the padding is set to 'same', which does not reduce the dimensions of the image. ReLU is chosen as the activation function and will be applied to each convolutional layer. Following the convolutional layer, a max-pooling layer with dimensions of 2x2, a stride of 2, and padding set to 'same' is added. The model will consistently use this same max-pooling configuration. In the next block, the second and third convolutional layers consist of 128 filters with dimensions of 5x5, followed by a max-pooling layer. The next two blocks of the network feature configurations of three consecutive convolutional layers with 256 filters of 3x3 each, followed by a max-pooling layer. The following block consists of three convolutional layers with 512 filters of 3x3, and a max-pooling layer. The penultimate block consists of three convolutional layers, each with 1024 filters of 3x3, followed by a max-pooling layer. The final block includes a flattening layer, a dense layer with 1024 neurons, a dropout layer with a rate of 0.5, and an output layer with the number of neurons equal to the number of classes (37), with softmax as the activation function.

|Layer (type)     |            Output Shape        |      Param #   |
|------------- | ------------- | ------------- |
|conv2d_40 (Conv2D)      |     (None, 112, 112, 96)  |    11712     |
|max_pooling2d_22 (MaxPooling) | (None, 56, 56, 96)   |     0       |  
|conv2d_41 (Conv2D)    |       (None, 56, 56, 128)  |     307328    |
|conv2d_42 (Conv2D)     |      (None, 56, 56, 128)   |    409728    |
|max_pooling2d_23 (MaxPooling) | (None, 28, 28, 128)  |     0       |  
|conv2d_43 (Conv2D)    |       (None, 28, 28, 256)   |    295168   | 
|conv2d_44 (Conv2D)     |      (None, 28, 28, 256)    |   590080   | 
|conv2d_45 (Conv2D)      |     (None, 28, 28, 256)   |    590080   | 
|max_pooling2d_24 (MaxPooling) | (None, 14, 14, 256)  |     0       |  
|conv2d_46 (Conv2D)     |      (None, 14, 14, 256)   |    590080    |
|conv2d_47 (Conv2D)    |       (None, 14, 14, 256)   |    590080    |
|conv2d_48 (Conv2D)     |      (None, 14, 14, 256)   |    590080    |
|max_pooling2d_25 (MaxPooling) | (None, 7, 7, 256)   |      0       |  
|conv2d_49 (Conv2D)    |       (None, 7, 7, 512)     |    1180160   |
|conv2d_50 (Conv2D)    |       (None, 7, 7, 512)     |    2359808   |
|conv2d_51 (Conv2D)      |     (None, 7, 7, 512)     |    2359808  | 
|max_pooling2d_26 (MaxPooling) | (None, 4, 4, 512)    |     0      |   
|conv2d_52 (Conv2D)      |     (None, 4, 4, 1024)    |    4719616  | 
|conv2d_53 (Conv2D)       |    (None, 4, 4, 1024)    |    9438208   |
|conv2d_54 (Conv2D)       |    (None, 4, 4, 1024)    |    9438208   |
|max_pooling2d_27 (MaxPooling) | (None, 2, 2, 1024)   |     0       |  
|flatten_4 (Flatten)     |     (None, 4096)        |      0         |
|dense_3 (Dense)         |     (None, 1024)        |      4195328   |
|dropout_1 (Dropout)    |      (None, 1024)        |      0         |
|dense_4 (Dense)        |      (None, 37)          |      37925     |
|------------- | ------------- | ------------- |
|Total params: |37,703,397| |
|Trainable params: | 37,703,397 | |
|Non-trainable params:| 0 | |
|------------- | ------------- | ------------- |

## 3.3. Mobile Application Testing the Network
The testing application was run on a Samsung Galaxy S8 with Android 9.0. For each model and accelerator configuration, inference time data was collected. Data was gathered for each gesture, which was shown against a uniform background for approximately 5 seconds.

The application allow users to select the network, delegate, and number of threads for inference. The camera continuously capture frames and provide them to the loaded model. Before delivering frames to the network, the images must be appropriately processed. This process includes segmentation to separate the hand from the background by converting the color space from RGB to HSV, a method that effectively distinguishes skin tones. The segmented image is then converted to grayscale and resized to fit the input layer of the neural network. The installed network makes predictions and provide feedback to the user in the form of text with the percentage prediction result.

[ADD: The details about the usage, testing and implementation of the Mobile Application]

## 3.4. Methods of Network Optimization
A detailed description of the used optimization methods

### Network quantization
Network quantization involves reducing the precision of parameters and intermediate activation maps that are typically stored in floating-point notation. Gholami et al. [[3]](#3) indicate that the first step is to define a quantization function that maps the real values of weights or activations to lower precision values. Usually, such a function maps these values to integer values, according to the following formula:

$Q(r)=Int(\frac{r}{S})$

where: \
Q(r) - quantization value\
Int - integer value\
r - real input value\
S - scaling factor

The above formula refers to symmetric quantization, where values are clipped using a symmetric range of values [-a, a]. Two main approaches in quantization can be distinguished: QAT (Quantization Aware Training) and PTQ (Post Training Quantization). In the first approach, quantization is performed during network training, where parameters are quantized after each gradient update. A major drawback, as indicated by the authors, is the computational cost of retraining the neural network. The second approach, PTQ, does not require retraining the network, as parameters are quantized after the network has been trained. It is a faster and simpler approach compared to QAT, but the model may suffer more in terms of detection accuracy.

###  Knowledge Distillation
Another approach of network optimization is Knowledge Distillation, which involves transferring information from a larger, more complex model to a less complex network. This method is often described in articles as a teacher-student format, where the teacher (the more complex network) imparts its knowledge to the student (the less complex network). Zhang et al. [2019] describe that the information flow is facilitated through a second (intermediate) network using data specifically labeled by the previous network. By utilizing synthetic data, the risk of overfitting the network is reduced, and very good function approximation is ensured. Moreover, this approach enables the compression and acceleration of complex networks.

###  Layer Decomposition
D


## References
<a id="1">[1]</a> Suharjito, Suharjito & Gunawan, Herman & Thiracitta, Narada & Nugroho, Ariadi. (2018). Sign Language Recognition Using Modified Convolutional Neural Network Model. 1-5. 10.1109/INAPR.2018.8627014. 

<a id="2">[2]</a> Makarov, Ilya & Veldyaykin, Nikolay & Chertkov, Maxim & Pokoev, Aleksei. (2019). American and russian sign language dactyl recognition. PETRA '19: Proceedings of the 12th ACM International Conference on PErvasive Technologies Related to Assistive Environments. 204-210. 10.1145/3316782.3316786. 

<a id="3">[3]</a> Gholami, A., Kim, S., Dong, Z., Yao, Z., Mahoney, M. W., Keutzer, K.: A survey of quantization methods for efficient neural network inference. In arXiv preprint arXiv:2103.13630, 2021. 
