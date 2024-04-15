imdsTrain = imageDatastore('seg_train\', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');


imdsTest = imageDatastore('seg_test\', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

imdsPredict = imageDatastore('seg_pred\', ...
    'IncludeSubfolders',true);

[imdsTrain1, imdsValidation] = splitEachLabel(imdsTrain,0.8);

tbl = countEachLabel(imdsTrain);


lenTraing = length(imdsTrain.Labels);
lenTest = length(imdsTest.Labels);

targetSize = [256, 256];
augmenter = imageDataAugmenter('RandXReflection',true, 'RandYReflection',true, ...
    'RandXScale',[0.5 1.5], 'RandYScale',[0.5 1.5]);
augmentedDS = augmentedImageDatastore(targetSize, imdsTrain1, 'DataAugmentation', augmenter);
augmentedDS2 = augmentedImageDatastore(targetSize, imdsTest, 'DataAugmentation', augmenter);
augmentedDSValidation = augmentedImageDatastore(targetSize, imdsValidation, 'DataAugmentation', augmenter);
%%
%CNN layers
layers = [...
    imageInputLayer([256 256 3])
    convolution2dLayer([3 3], 32)
    reluLayer
    maxPooling2dLayer(2, 'Stride',2)
    convolution2dLayer([3 3], 32)
    reluLayer
    maxPooling2dLayer(2, 'Stride',2)
    convolution2dLayer([3 3], 32)
    reluLayer
    maxPooling2dLayer(2, 'Stride',2)
    convolution2dLayer([3 3], 32)
    reluLayer
    maxPooling2dLayer(2, 'Stride',2)
    convolution2dLayer([3 3], 32)
    reluLayer
    maxPooling2dLayer(2, 'Stride',2)
    fullyConnectedLayer(125)
    reluLayer
    fullyConnectedLayer(6)
    softmaxLayer
    classificationLayer];

miniBatchSize = 40;
valFrequency = floor(numel(imdsTrain.Files)/miniBatchSize);
%Neural network options
options = trainingOptions('adam',...
    'MiniBatchSize', miniBatchSize, ...
    'MaxEpochs',7, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData' ,augmentedDSValidation, ...
    'ValidationFrequency',100, ...
    'Verbose',true, ...
    'Plots','training-progress', ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor',0.5 , ...
    'LearnRateDropPeriod', 100);


net = trainNetwork(augmentedDS, layers, options);
save('trained_network.mat', 'net');
%% 
load('trained_network.mat');
[YPred, probs] = classify(net, augmentedDS2);
YTest = imdsTest.Labels;
totalImages = numel(imdsTrain.Files);
numImagesToDisplay = 6;
randomIndices = randperm(totalImages, numImagesToDisplay);
randomIndices = mod(randomIndices - 1, numel(YPred)) + 1;
disp(randomIndices);
linearIndices = randomIndices;
figure;
trueLabels = categorical();
for i = 1:numImagesToDisplay
    % Read the image using the random index
    img = readimage(imdsTest, randomIndices(i));
    currentPrediction = YPred(linearIndices(i));
    trueLabel = readimage(imdsTest, randomIndices(i));
    trueLabel = categorical(trueLabel);
    isPredictionCorrect = currentPrediction == trueLabel;
    subplot(2, 3, i)
    imshow(img);
    title(sprintf('Image %d\nPrediction: %s', randomIndices(i), char(currentPrediction)));
end

%%
% Compute confusion matrix
confMat = confusionmat(YTest, YPred);
numClasses = size(confMat, 1);
TP = zeros(1, numClasses);
FP = zeros(1, numClasses);
FN = zeros(1, numClasses);
TN = zeros(1, numClasses);
for i =1:numClasses
    TP(i) = (confMat(i, i));
    FP(i) = sum(confMat(:, i)) - TP(i);
    FN(i) = sum(confMat(i, :)) - TP(i);
    TN(i) = sum(sum(confMat(:))) - TP(i) - FN(i) - FP(i);
end

disp(['TP is : ', num2str(TP)]);
disp(['FP is : ', num2str(FP)]);
disp(['FN is : ', num2str(FN)]);
disp(['TN is : ', num2str(TN)]);

% Display metrics
figure
cm = confusionchart(YTest, YPred);
cm.Title = 'Confusion Matrix';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

%%
disp('---------Accuracy---------');
accuracy = ((TP+TN)/(TP+TN+FP+FN));
disp(['Accuracy is: ', num2str(accuracy)]);

disp('---------Recall---------');
recall = (TP/(TP+FN));
disp(['Recall is: ', num2str(recall)]);

disp('---------Specificity---------');
s1 = zeros(1, numClasses);
for i = 1:numClasses
    s1(i) = (TN(i)/(TN(i) + FP(i)));
    disp(['Specificity for ', num2str(i), ' is: ', num2str(s1(i))]);
end

disp('---------Precision---------');
precision = zeros(1, numClasses);
for i = 1:numClasses
    precision(i) = (TP(i)/(TP(i) + FP(i)));
    disp(['Precision for ', num2str(i), ' is: ', num2str(precision(i))]);
end

disp('---------F1-Score---------');
f1 = zeros(1, numClasses);
for i = 1:numClasses
    f1(i) = (2.*TP/((2.*TP) + FP + FN));
    disp(['F1-Score for ', num2str(i), ' is: ', num2str(f1(i))]);
end

disp('---------Matthews Correlation Coefficient---------');
mcc = (((TP.*TN)-(FP.*FN))/sqrt((TP+FP).*(TP+FN).*(TN+FP).*(TN+FN)));
disp(['Matthews Correlation Coefficient is: ', num2str(mcc)]);