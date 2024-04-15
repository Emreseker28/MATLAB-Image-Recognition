<h1>1 Introduction</h1>
The field of computer vision has witnessed significant advancements in recent years, with deep learning techniques proving to be particularly effective in various tasks. This project focuses on leveraging Convolutional Neural Networks (CNNs) for the task of image classification.
The primary goal of this project is to develop a CNN-based image classification model capable of accurately classifying and identifying images. To achieve this, I have employed a carefully selected dataset, partitioned into training, validation, and testing subsets. This report outlines the theoretical foundation of my chosen CNN model, the adopted data preparation procedures, and a comprehensive analysis of the model's performance.
<h1>2 Dataset</h1>
The dataset used in this project consists of a diverse set of images, catering to the complexity and variability inherent in real-world scenarios. The dataset is divided into training, validation, and testing subsets, each serving a specific purpose in the model development and evaluation process. The following sections provide an in-depth description of the dataset, including its size, proportions among subsets, and illustrative examples.
The dataset consists of six different image classes. All images in these classes are from real-world scenarios. These classes are as follows: buildings, forest, glacier, mountain, sea and street. For training, each of these classes contain more than 2000 images. Test and prediction images are divided manually. Each of the classes also contain approximately 500 extra images for testing. And the prediction images collectively consist of 7301 images. In total, there are 14034 images for training and 3000 for testing. 
The dataset is taken from the website ‘kaggle.com’ which provides free-to-use datasets inside. The name of the dataset is ‘Intel Image Classification’ and can be found on the aforementioned website for free. In figures 2.1, 2.2, 2.3, 2.4, 2.5, 2.6 the example images for the classes can be observed.
 ![image](https://github.com/Emreseker28/MATLAB-Image-Recognition/assets/54375145/a6f63291-1545-4c9c-9e98-83ec04453d84)

Figure 2.1: An example image for the buildings class from training set.
 ![image](https://github.com/Emreseker28/MATLAB-Image-Recognition/assets/54375145/5759305d-5433-4918-97a9-5c47936bdc66)

Figure 2.2: An example image for the forest class from training set.
 ![image](https://github.com/Emreseker28/MATLAB-Image-Recognition/assets/54375145/3553be06-de60-4754-b018-882d64a765f5)

Figure 2.3: An example image for the glacier class from training set.
 ![image](https://github.com/Emreseker28/MATLAB-Image-Recognition/assets/54375145/14a7d14c-bc7c-4152-a0bf-249e0b6fe139)

Figure 2.4: An example image for the mountain class from training set.
 ![image](https://github.com/Emreseker28/MATLAB-Image-Recognition/assets/54375145/f0d54c57-115f-4d1f-805d-bce6023d9837)

Figure 2.5: An example image for the sea class from training set.
 ![image](https://github.com/Emreseker28/MATLAB-Image-Recognition/assets/54375145/03c3020d-5a71-4c65-b050-eadb92a9c611)

Figure 2.6: An example image for the street class from training set.
<h1>3 Methodology</h1>
This part of the paper consists of the information about the machine learning model that is used in this project. As mentioned in the introduction part, CNN is the machine learning model that is being used. As requested, there are 5 convolutional layers in the model. Additionally, there are 5 maximum pooling layers after each convolutional layer. Lastly, there are 2 fully connected layers in the model. 
<h2>3.1 Convolutional Neural Network</h2>
CNN uses the layer system same as deep neural networks. Generally, a CNN model consists of four types of layers: Convolutional layer, pooling layer, flattening layer and lastly a fully connected layer (sometimes referred as dense layer).  
Each layer has different characteristics. In convolutional layer, ‘filters’ are used for the feature extraction. These filters are usually 3x3, 5x5 matrices. With these filters, important features from the data can be extracted by making a matrix multiplication. Applying these filters will reduce the dimensionality of the original data, and this may result in losing important data. To prevent losing too much of the original data, a padding method can be used. This method adds 0s to the outer line of the data (in a case of the data is an image, it adds 0s to the outer line of the image) before filters applied to it. But this doesn’t fully prevent losing the data. After each convolutional layer, some data will be lost.
When it comes to the pooling layers, there are two types: maximum pooling layer and average pooling layer. Their names are pretty self-explanatory, maximum pooling layer takes the maximum value inside the filter, whereas the average pooling layer takes the average of the values inside the filter. Pooling layer is generally used for reducing the dimensionality of the input. This makes the machine learning model work faster and it prevents the model from ‘memorizing’ the input data (in other words: overfitting).
Flattening layer is one of the optional layers in the CNN layer system. As its name suggests, its job is to turn the shape of the data into a flat shape. This step may be required for the fully connected layers, if the fully connected layer’s input shape is different than the original data’s input shape. 
The fully connected layer is the last step of the layered system in CNN. It is similar to the traditional DNN layer, each of the neurons are connected to all neurons on the next layer. A machine learning model can have multiple fully connected layers. The number of neurons in the fully connected layers depends on the user’s preferences, the size of the data and other parameters. A higher number of neurons sometimes does not mean higher accuracy. Therefore, several tests need to be done to find the suitable number of neurons. 
All of the convolutional layers in the model use Rectified Linear Unit (ReLU) as the activation function. ReLU introduces non-linearity and sparsity to the model. The function of the ReLU is as follows:
f(x) = max⁡(0, x)
Where x is the input of the model. Basically, ReLU turns negative values into 0, and keeps the positive values as the same. This can be represented by a graph as in figure 3.1. Therefore, it is computationally less efficient compared to the other activation functions. It is also important to mention that all convolutional layers are 2-dimensional (because the input data is also 2-dimensional), and the filter size is 32. 
![image](https://github.com/Emreseker28/MATLAB-Image-Recognition/assets/54375145/1769e84f-4955-4766-999a-eb60859865dd)

 Figure 3.1: Graph of ReLU activation function.
The fully connected layers differ in their neuron numbers. The first fully connected layer has 125 neurons whereas the last one (also can be considered as the output layer) has only 6. Their activation functions also differ. The first one uses the ReLU activation function and the last one uses ‘softmax’ activation function. 
The softmax layer is commonly used in the output layer of a neural network, particularly in classification tasks. The primary purpose of the softmax layer is to normalize the raw scores produced by the preceding layer, transforming them into probabilities that sum to 1. The output of the softmax layer can be interpreted as the model's confidence scores for each class. The class with the highest probability is considered the model's prediction. The softmax layer is often used in conjunction with the cross-entropy loss function for training classification models. The cross-entropy loss measures the dissimilarity between the predicted probabilities and the true distribution of class labels. The function of softmax layer can be given as:
![image](https://github.com/Emreseker28/MATLAB-Image-Recognition/assets/54375145/4e78b8f0-1f1d-405c-8d5c-805dad2bb5ee)

Where z represents the given input factor (e.g. z=[z1 ,z2 ,...,zk ]), zi is the raw score of the class i, e is the base of the natural logarithm. 
Training options are the most time spent to find the right configuration for higher accuracy in this project. A lot of different parameters have been tested. The initial test results were unacceptable to say the least. The accuracy results for the initial run were around 30%, which is considered as a “bad” machine learning model. After a lot of configurations and testing, the accuracy is now more than 70%, which is optimistic and promising for a machine learning model. 
For the training options, ‘adam’ optimization method has been selected, after trying ‘sgdm’ and ‘rmsprop’. The optimization method ‘sgdm’ finishes training early for the reason being loss function value becomes too low after a couple of iterations. And ‘rmsprop’ does not offer as high accuracy as ‘adam’ optimization method does. For the ‘MiniBatchSize’, the number 40 is the optimal value for my computer. If this number gets too high, my graphics card’s VRAM begins to not be able to keep up with the MATLAB and throws an error. This batch size number can be changed if this program is being tested on a different and powerful computer than mine. There are only 7 epochs in the training, because it is observed that after the 7th epoch, the accuracy and loss stay the same. Therefore, to prevent overfitting, this number has been decided to be set to 7. More detailed information can be found in the code that is located in ‘APPENDIX’ part of this paper. Additionally, figure 3.2 consists of the analysis results from the built-in MATLAB feature called Deep Network Designer.
 ![image](https://github.com/Emreseker28/MATLAB-Image-Recognition/assets/54375145/d0c3d7a6-a428-4a69-90e8-1f10ef782e50)

 Figure 3.2: Deep Network Designer analysis result.
<h1>4 Results</h1>
In this part of the paper, the results of the machine learning model will be discussed. The training accuracy of the model was 73.73% after 7 epochs as can be observed in figure 4.3. The machine has been trained with 12 and 20 epochs as well. Figure 4.2 shows the training with 20 epochs, and it can be observed that the model starts to show the features of overfitting. In order to avoid overfitting, the epoch number is reduced to 12, which also shows some overfitting as well as can be observed from figure 4.1. It has been decided that after the 7th epoch, the machine learning model starts to memorize the dataset given, even though it gets shuffled every epoch. 
As for the loss during the training, the mini-batch loss is 0.678, and the validation loss is 0.733. These values are taken from the output of the command window, after setting the ‘verbose’ option to true in the training options. 
 ![image](https://github.com/Emreseker28/MATLAB-Image-Recognition/assets/54375145/0fb47fad-b029-4e38-a2c2-dafdb5bf18da)

Figure 4.1: Training graph of 12 epochs.
 ![image](https://github.com/Emreseker28/MATLAB-Image-Recognition/assets/54375145/c2f30a9e-57c9-430e-8bcf-032410d2d7b7)

Figure 4.2: Training graph of 20 epochs. 
 ![image](https://github.com/Emreseker28/MATLAB-Image-Recognition/assets/54375145/a3ca0250-8206-4b9f-bbdd-a128d1c0d3c2)

Figure 4.3: Training graph with 7 epochs.
In figure 4.4 an example output of the machine learning model can be found. As mentioned before, the model has around 73% accuracy. This means it is not expected for every prediction to be correct. And in the example output from figure 4.4, it can be observed that only four of the 6 images are correctly classified. These correct images are 1943, 983, 578 and 2679.
![image](https://github.com/Emreseker28/MATLAB-Image-Recognition/assets/54375145/7d2149c0-8b6e-4b4f-9dde-05bc070884f0)

 Figure 4.4: A result of randomly selected 6 images.
For the accuracy metrics, the confusion matrix of the machine learning model can be found in figure 4.5. In the diagonal axis of the matrix, the true positive values can be observed. These true positive values represent the number of correctly predicted images during the training. It also can be said that the mountain and the buildings classes have the lowest accuracy among all classes. From that, we can assume that it is hard for the model to differentiate the mountains and buildings from other images. One of the ways to fix this issue could be increasing the number of images used during the training of these classes.
 ![image](https://github.com/Emreseker28/MATLAB-Image-Recognition/assets/54375145/60c5c8a7-ec89-409f-8988-f413d82fe748)

Figure 4.5: Confusion matrix of the model. 
There are a couple other metrics for accuracy that are being used in this project as well. In table 4.1 and table 4.2, the metrics that are being used in this project and their values can be found. 
Table 4.1: Specificity and Precision values for each class.
Metric\Class	1	2	3	4	5	6
Specificity	0.927	0.981	0.932	0.917	0.937	0.972
Precision	0.638	0.898	0.708	0.658	0.693	0.805
F1-Score	0.723	0.723	0.723	0.723	0.723	0.723

Table 4.2: Accuracy metrics and their values.
Metric	Value
Accuracy	0.908
Recall	0.722
Matthew’s Correlation Coefficient	0.669

Accuracy is a fundamental metric representing the overall correctness of the model's predictions. An accuracy of 0.908 implies that approximately 90.8% of the predictions made by the model are correct. 
Recall, also known as sensitivity or true positive rate, measures the model's ability to correctly identify instances of a specific class. The average recall across all classes is 0.722, indicating that, on average, the model correctly identifies about 72.2% of instances for all classes.
Specificity measures the ability of the model to correctly identify negative instances. Each specificity value corresponds to a specific class. Higher specificity values indicate better performance in correctly identifying true negatives. The specificity values for each class range from 0.917 to 0.981, demonstrating good performance in distinguishing negative instances.
Precision assesses the accuracy of positive predictions made by the model. The precision values for each class vary, with an average precision of 0.733. Higher precision values indicate fewer false positives, and lower precision values suggest more false positives.
The F1-Score is the harmonic mean of precision and recall. F1-Scores are consistent across all classes, indicating a balanced trade-off between precision and recall. The uniformity of F1-Scores suggests that the model performs consistently across different classes.
The Matthews Correlation Coefficient takes into account true positives, true negatives, false positives, and false negatives, providing a balanced measure of classification performance. An MCC of 0.669 indicates a moderate level of agreement between predicted and true class labels.
The model seems to perform well overall, with high accuracy and balanced performance across precision, recall, and F1-Score. It's essential to consider the specific requirements of the application to determine which metrics are most critical. For instance, in medical diagnosis, high sensitivity might be crucial even at the expense of precision. In this project’s case, we can take accuracy as the most crucial one among all of them. 

<h1>5 Conclusion</h1>
The model demonstrates strong overall performance, with high accuracy and balanced precision, recall, and F1-Score across classes. The choice of the most crucial metric depends on specific application requirements, but accuracy, with a value of 0.908, stands out as a fundamental measure of correctness. In conclusion, the project succeeds in developing an effective CNN-based image classification model, showcasing promising results and providing insights into the model's behavior across diverse classes.
Getting accuracy like 73% was not an easy task. It took several different training options for me to get to this point. I had to change the original dataset I found in the first place to this current one. Another hard task for me to do was to adjust the hyperparameters of the machine learning model so that I could use my GPU to its full utilization. Because my GPU only has 4 GB of VRAM, it gets full very quickly. And when it gets full, MATLAB gives an error and crashes, and I have to open up MATLAB again every time.
Some of the classes (like buildings and mountain) have lower true positive values than other classes, like I mentioned in the results part. This issue can be resolved by adding more images to these classes in the training.
