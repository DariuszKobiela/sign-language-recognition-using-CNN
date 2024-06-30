# sign-language-recognition-using-CNN
This repository is the extension of the article "Sign language recognition using Convolutional Neural Networks" written by Kobiela et al. [CITE] and contains the detailed report about the literature review and details about the usage, testing and
implementation of the mobile application. 

## Code
> Mobile application developed for the research: https://github.com/adartemiuk/ASL_recognition_app

> Models source code and training scripts: https://github.com/adartemiuk/ASL_recognition_models

## 2. Background and related work
Table below contains summary of models from the literature review. 

### List of compared models performing gesture recognition tasks: 

| Author  | Model | Gesture type | Sign language type | Accuracy |
|------------- | ------------- | ------------- |  ------------- |  ------------- | 
| Gunawan et al. [2018] | i3D Inception  | dynamic |  Argentinean |  100% |
| Makarov et al. [2019] | QuadroConvPoolNet  | static |  American  |  about 100% |
| Bhadra & Kar [2021]  | custom CNN  | static+dynamic |  American |  99.89% |
| Cayamcela & Lim [2019]  | AlexNet  | static |  American |  99.39% |
| Hasan et al. [2020]  | custom CNN  | static |  Bengali |  99.22% |
| Kang et al. [2015]  | CaffeNet  | static |  American |  99% |
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
1. [Malakan & Albaqami, 2021] Malakan, Z. M., Albaqami, H. A.: Classify, Detect and Tell: Real-
Time American Sign Language. In: IEEE 2021 National Computing Colleges Conference (NCCC),
pp. 1-6, March 2021
2. [Le et al., 2020] Le, S., Lei, Q., Wei, X., Zhong, J., Wang, Y., Zhou, J., Wang, W.: Smart
Elevator Cotrol System Based on Human Hand Gesture Recognition. In: 2020 IEEE 6th
International Conference on Computer and Communications (ICCC), pp. 1378-1385, December
2020
3. [Rahim et al., 2019a] Rahim, M. A., Islam, M. R., Shin, J.: Non-touch sign word recognition
based on dynamic hand gesture using hybrid segmentation and CNN feature fusion. Applied
Sciences, 9(18), 3790, 2019
4. [Rahim et al., 2019b] Rahim, M. A., Shin, J., Islam, M. R.: Dynamic Hand Gesture Based Sign
Word Recognition Using Convolutional Neural Network with Feature Fusion. In: 2019 IEEE 2nd
International Conference on Knowledge Innovation and Invention (ICKII), pp. 221-224, July 2019
5. [Cayamcela & Lim, 2019] Cayamcela, M. E. M., Lim, W.: Fine-tuning a pre-trained
convolutional neural network model to translate American sign language in real-time. In: IEEE
2019 International Conference on Computing, Networking and Communications (ICNC), pp. 100-
104, February 2019
6. [Sun et al., 2018] Sun, J. H., Ji, T. T., Zhang, S. B., Yang, J. K., Ji, G. R.: Research on the
hand gesture recognition based on deep learning. In: IEEE 2018 12th International symposium
on antennas, propagation and EM theory (ISAPE), pp. 1-4, December 2018
7. [Liu et al., 2019] Liu, P., Li, X., Cui, H., Li, S., Yuan, Y.: Hand gesture recognition based on
single-shot multibox detector deep learning. Mobile Information Systems, 2019
8. [Kang et al., 2015] Kang, B., Tripathi, S., Nguyen, T. Q.: Real-time sign language fingerspelling
recognition using convolutional neural networks from depth map. In: IEEE 2015 3rd IAPR Asian
Conference on Pattern Recognition (ACPR), pp. 136-140, November, 2015
9. [Makarov et al., 2019] Makarov, I., Veldyaykin, N., Chertkov, M., Pokoev, A.: American and
Russian sign language dactyl recognition. In: Proceedings of the 12th ACM International
Conference on PErvasive Technologies Related to Assistive Environments, pp. 204-210, June
2019
10. [Das et al., 2020] Das, P., Ahmed, T., Ali, M. F.: Static hand gesture recognition for American
Sign Language using deep convolutional neural network. In: 2020 IEEE Region 10 Symposium
(TENSYMP), pp. 1762-1765, June 2020
11. [Taskiran et al., 2018] Taskiran, M., Killioglu, M., Kahraman, N.: A real-time system for
recognition of American sign language by using deep learning. In: IEEE 2018 41st International
Conference on Telecommunications and Signal Processing (TSP), pp. 1-5, July 2018
12. [Daroya et al., 2018] Daroya, R., Peralta, D., Naval, P.: Alphabet sign language image
classification using deep learning. In: TENCON 2018-2018 IEEE Region 10 Conference, pp.
0646-0650, October 2018
13. [Hasan et al., 2020] Hasan, M. M., Srizon, A. Y., Sayeed, A., Hasan, M. A. M.: Classification
of American Sign Language by Applying a Transfer Learned Deep Convolutional Neural Network.
In: IEEE 2020 23rd International Conference on Computer and Information Technology (ICCIT),
pp. 1-6, December 2020
14. [Agrawal et al., 2020] Agrawal, M., Ainapure, R., Agrawal, S., Bhosale, S., Desai, S.: Models
for hand gesture recognition using deep learning. In: 2020 IEEE 5th International Conference on
Computing Communication and Automation (ICCCA), pp. 589-594, October 2020
15. [Mohammed et al., 2019] Mohammed, A. A. Q., Lv, J., Islam, M. S.: Small Deep Learning
Models for Hand Gesture Recognition. In: 2019 IEEE Intl Conf on Parallel & Distributed
Processing with Applications, Big Data & Cloud Computing, Sustainable Computing &
Communications, Social Computing & Networking (ISPA/BDCloud/SocialCom/SustainCom), pp.
1429-1435, December 2019
16. [Nguyen & Do, 2019] Nguyen, H. B., & Do, H. N.: Deep learning for American sign language
fingerspelling recognition system. In: IEEE 2019 26th International Conference on
Telecommunications (ICT), pp. 314-318, April 2019
17. [Kurhekar et al., 2019] Kurhekar, P., Phadtare, J., Sinha, S., Shirsat, K. P.: Real Time Sign
Language Estimation System. In: 2019 3rd International Conference on Trends in Electronics and
Informatics (ICOEI), pp. 654-658, April 2019
18. [Chavan et al., 2021] Chavan, S., Yu, X., Saniie, J.: Convolutional Neural Network Hand
Gesture Recognition for American Sign Language. In: 2021 IEEE International Conference on
Electro Information Technology (EIT), pp. 188-192, May 2021
19. [Pala et al., 2021] Pala, G., Jethwani, J. B., Kumbhar, S. S., Patil, S. D.: Machine Learningbased
Hand Sign Recognition. In: IEEE 2021 International Conference on Artificial Intelligence
and Smart Systems (ICAIS), pp. 356-363, March 2021
20. [Sonare et al., 2021] Sonare, B., Padgal, A., Gaikwad, Y., Patil, A.: Video-Based Sign
Language Translation System Using Machine Learning. In: IEEE 2021 2nd International
Conference for Emerging Technology (INCET), pp. 1-4, May 2021
21. [Farahanipad et al., 2020] Farahanipad, F., Nambiappan, H. R., Jaiswal, A., Kyrarini, M.,
Makedon, F.: HAND-REHA: dynamic hand gesture recognition for game-based wrist
rehabilitation. In: Proceedings of the 13th ACM International Conference on PErvasive
Technologies Related to Assistive Environments, pp. 1-9, June 2020
22. [Lakhotiya et al., 2021] Lakhotiya, H., Pandita, H. S., Shankarmani, R.: Real Time Sign
Language Recognition Using Image Classification. In: IEEE 2021 2nd International Conference
for Emerging Technology (INCET), pp. 1-4, May 2021
23. [Ding et al., 2020] Ding, Y., Huang, S., Peng, R.: Data Augmentation and Deep Learning
Modeling Methods on Edge-Device-Based Sign Language Recognition. In: IEEE 2020 5th
International Conference on Information Science, Computer Technology and Transportation
(ISCTT), pp. 490-497, November 2020
24. [Khan et al., 2019] Khan, S. A., Joy, A. D., Asaduzzaman, S. M., Hossain, M.: An efficient
sign language translator device using convolutional neural network and customized ROI
segmentation. In: IEEE 2019 2nd International Conference on Communication Engineering and
Technology (ICCET), pp. 152-156, April 2019
25. [Hossen et al., 2018] Hossen, M. A., Govindaiah, A., Sultana, S., Bhuiyan, A.: Bengali sign
language recognition using deep convolutional neural network. In: IEEE 2018 joint 7th
international conference on informatics, electronics & vision (iciev) and 2018 2nd international
conference on imaging, vision & pattern recognition (icIVPR), pp. 369-373, June 2018
26. [Hasan et al., 2020] Hasan, M. M., Srizon, A. Y., Hasan, M. A. M.: Classification of Bengali
sign language characters by applying a novel deep convolutional neural network. In: 2020 IEEE
Region 10 Symposium (TENSYMP), pp. 1303-1306, June 2020
27. [Sayeed et al., 2020] Sayeed, A., Hasan, M. M., Srizon, A. Y.: Bengali Sign Language
Characters Recognition by Utilizing Transfer Learned Deep Convolutional Neural Network. In:
IEEE 2020 11th International Conference on Electrical and Computer Engineering (ICECE), pp.
423-426, December 2020
28. [Rafi et al., 2019] Rafi, A. M., Nawal, N., Bayev, N. S. N., Nima, L., Shahnaz, C., Fattah, S.
A.: Image-based bengali sign language alphabet recognition for deaf and dumb community. In:
2019 IEEE global humanitarian technology conference (GHTC), pp. 1-7, October 2019
29. [Huynh & Ngo, 2020] Huynh, L., Ngo, V.: Recognize Vietnamese Sign Language Using Deep
Neural Network. In: IEEE 2020 7th NAFOSTED Conference on Information and Computer
Science (NICS), pp. 191-196, November 2020
30. [Shurid et al., 2020] Shurid, S. A., Amin, K. H., Mirbahar, M. S., Karmaker, D., Mahtab, M.
T., Khan, F. T., …, Alam, M. A.: Bangla Sign Language Recognition and Sentence Building Using
Deep Learning. In: 2020 IEEE Asia-Pacific Conference on Computer Science and Data
Engineering (CSDE), pp. 1-9, December 2020
31. [Tasmere & Ahmed, 2020] Tasmere, D., Ahmed, B.: Hand gesture recognition for Bangla
sign language using deep convolution neural network. In: IEEE 2020 2nd International
Conference on Sustainable Technologies for Industry 4.0 (STI), pp. 1-5, December 2020
32. [Krishnan & Balasubramanian, 2019] Krishnan, P. T., Balasubramanian, P.: Detection of
alphabets for machine translation of sign language using deep neural net. In: IEEE 2019
International Conference on Data Science and Communication (IconDSC), pp. 1-3, March 2019
33. [Aich et al., 2020] Aich, D., Al Zubair, A., Hasan, K. Z., Nath, A. D., Hasan, Z.: A Deep
Learning Approach for Recognizing Bengali Character Sign Langauage. In: IEEE 2020 11th
International Conference on Computing, Communication and Networking Technologies
(ICCCNT), pp. 1-5, July 2020
34. [Bantupalli & Xie, 2018] Bantupalli, K., Xie, Y.: American sign language recognition using
deep learning and computer vision. In: 2018 IEEE International Conference on Big Data (Big
Data), pp. 4896-4899, December 2018
35. [Zamora-Mora & Chacón-Rivas, 2019] Zamora-Mora, J., Chacón-Rivas, M.: Real-Time
Hand Detection using Convolutional Neural Networks for Costa Rican Sign Language
Recognition. In: IEEE 2019 International Conference on Inclusive Technologies and Education
(CONTIE), pp. 180-1806, October 2019
36. [Bhadra & Kar, 2021] Bhadra, R., Kar, S.: Sign Language Detection from Hand Gesture
Images using Deep Multi-layered Convolution Neural Network. In: 2021 IEEE Second
International Conference on Control, Measurement and Instrumentation (CMI), pp. 196-200,
January 2021
37. [Qianzheng et al., 2021] Qianzheng, Z., Xiaodong, L., Jie, R., Yuanyuan, Q.: Real Time Hand
Gesture Recognition Applied for Flight Simulator Controls. In: 2021 IEEE 7th International
Conference on Virtual Reality (ICVR), pp. 407-411, May 2021
38. [Lu et al., 2019] Lu, Z., Qin, S., Li, L., Zhang, D., Xu, K., Hu, Z.: One-shot learning hand
gesture recognition based on lightweight 3D convolutional neural networks for portable
applications on mobile systems. IEEE Access, 7, 131732-131748, 2019
39. [Park et al., 2020] Park, H., Lee, J. S., Ko, J.: Achieving Real-Time Sign Language
Translation Using a Smartphone's True Depth Images. In: IEEE 2020 International Conference
on COMmunication Systems & NETworkS (COMSNETS), pp. 622-625, January 2020
40. [Gunawan et al., 2018] Gunawan, H., Thiracitta, N., Nugroho, A.: Sign language recognition
using modified convolutional neural network model. In: IEEE 2018 Indonesian Association for
Pattern Recognition International Conference (INAPR), pp. 1-5, September 2018
41. [Siriak et al., 2019] Siriak, R., Skarga-Bandurova, I., Boltov, Y.: Deep convolutional network
with long short-term memory layers for dynamic gesture recognition. In: 2019 10th IEEE
International Conference on Intelligent Data Acquisition and Advanced Computing Systems:
Technology and Applications (IDAACS), Vol. 1, pp. 158-162, September 2019
42. [Rao et al., 2018] Rao, G. A., Syamala, K., Kishore, P. V. V., Sastry, A. S. C. S.: Deep
convolutional neural networks for sign language recognition. In: IEEE 2018 Conference on Signal
Processing And Communication Engineering Systems (SPACES), pp. 194-197 January, 2018
43. [Yang & Zhu, 2017] Yang, S., Zhu, Q.: Video-based Chinese sign language recognition using
convolutional neural network. In: 2017 IEEE 9th International Conference on Communication
Software and Networks (ICCSN), pp. 929-934, May 2017
44. https://www.kaggle.com/datamunge/sign-language-mnist (ostatni dostęp 22.03.2022)
45. https://www.kaggle.com/grassknoted/asl-alphabet (ostatni dostęp 22.03.2022)
46. [Barczak et al., 2011] Barczak, A. L. C., Reyes, N. H., Abastillas, M., Piccio, A., Susnjak, T.:
A new 2D static hand gesture colour image dataset for ASL gestures, 2011
47. [Materzynska et al., 2019] Materzynska, J., Berger, G., Bax, I., Memisevic, R.: The jester
dataset: A large-scale video dataset of human gestures. In Proceedings of the IEEE/CVF
International Conference on Computer Vision Workshops, pp. 0-0, 2019
48. [Ronchetti et al., 2016] Ronchetti, F., Quiroga, F., Estrebou, C. A., Lanzarini, L. C., Rosete,
A: LSA64: an Argentinian sign language dataset. In XXII Congreso Argentino de Ciencias de la
Computación (CACIC 2016), 2016
49. [Li et al., 2020] Li, D., Rodriguez, C., Yu, X., Li, H.: Word-level deep sign language recognition
from video: A new large-scale dataset and methods comparison. In Proceedings of the IEEE/CVF
winter conference on applications of computer vision, pp. 1459-1469, 2020
50. [Zahedi et al., 2005] Zahedi, M., Keysers, D., Ney, H.: Pronunciation clustering and modeling
of variability for appearance-based sign language recognition. In International gesture workshop,
pp. 68-79, Springer, Berlin, Heidelberg, 2005, May
51. [Han et al., 2015] Han, S., Mao, H., Dally, W. J.: Deep compression: Compressing deep
neural networks with pruning, trained quantization and huffman coding. In arXiv preprint
arXiv:1510.00149, 2015
52. [Polyak & Wolf, 2015] Polyak, A., & Wolf, L.: Channel-level acceleration of deep face
representations. IEEE Access, 3, 2163-2175, 2015
53. [Li et al., 2016] Li, H., Kadav, A., Durdanovic, I., Samet, H., Graf, H. P.: Pruning filters for
efficient convnets. In arXiv preprint arXiv:1608.08710, 2016
54. [Gholami et al., 2021] Gholami, A., Kim, S., Dong, Z., Yao, Z., Mahoney, M. W., Keutzer, K.:
A survey of quantization methods for efficient neural network inference. In arXiv preprint
arXiv:2103.13630, 2021
55. [Zhang et al., 2019] Zhang, Q., Zhang, M., Chen, T., Sun, Z., Ma, Y., & Yu, B.: Recent
advances in convolutional neural network acceleration. In Neurocomputing, 323, 37-51, 2019
56. [Lebedev et al., 2014] Lebedev, V., Ganin, Y., Rakhuba, M., Oseledets, I., & Lempitsky, V.:
Speeding-up convolutional neural networks using fine-tuned cp-decomposition. In arXiv preprint
arXiv:1412.6553, 2014
57. [Kim et al., 2015] Kim, Y. D., Park, E., Yoo, S., Choi, T., Yang, L., & Shin, D.: Compression
of deep convolutional neural networks for fast and low power mobile applications. In arXiv preprint
arXiv:1511.06530, 2015
