net = alexnet;

rootFolder = '101_ObjectCategories';
categories = {'leopards','pizza','buddha','emu'};
imds = imageDatastore(fullfile(rootFolder,categories),'LabelSource', 'foldernames');
imds.ReadFcn = @readFunctionTrain;

[trainingSet, ~] = splitEachLabel(imds, 50, 'randomize'); 
% Extract features from the training set images(
%Features are extracted through activations, which will pull the features
%learned from the CNN 
%up to that point in the architecture.  
%If you use a network trained on millions of images, such as alexnet, 
%you can expect the features pulled from the network will be very rich,
%complex features that describe the objects.

%%cifar10 dataset
% cpu 1st deer 50-9mins wrong 2nd-6mins(no cat)right 3rd deer 100-15mins wrong 4th deer 500-38min wrong)
%(5th frog 50 -4 mins right 6th frog net 50-6mins wrong 7th-frog net 100-8mins wrong 8th-frog net 500- wrong 38mins )
%9th airplane 50-4 mins

%gpu 1st automobile 50 30secs 2nd deer 50 26secs wrong 3rd deer 100
%32 secs right 4th bird 100 35secs 5th airplane 500 60secs
%Train 2000(500*4) Test 4000(1000*4) images 170secs 86.15
%Train 4000(1000*4) Test 4000(1000*4) images 135 secs 86.55
%Train 8000(2000*4) Test 4000(1000*4) images 190 secs 86.35
%Train 10000(1000*10) Test 10000(1000*10) images 390 secs 65.73
%Train 50000(5000*10) Test 10000(1000*10) images 1000 secs 66.20 fitcnb 

%%%caltech101
%gpu
%1st elephant 40*4  10-16secs
%1st pizza 40*4  8-11secs
%1st pizza 20*4  8-9secs
%1st pizza 10*4  6-7secs

featureLayer = 'fc7';
trainingFeatures = activations(net, trainingSet, featureLayer,'ExecutionEnvironment','gpu');

%train the SVM classifier
classifier = fitcnb(trainingFeatures, trainingSet.Labels);

%rootFolder = 'cifar10Test';
%testSet = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
%testSet.ReadFcn = @readFunctionTrain;

picture = imread('image7.jpg');              % Take a picture    
picture = imresize(picture,[227,227]);  % Resize the picture


% Extract features from the test set images, and test SVM classifer
testFeatures = activations(net, picture, featureLayer);
predictedLabels = predict(classifier, testFeatures);

image(picture);     % Show the picture
title(char(predictedLabels));
%confMat = confusionmat(testSet.Labels, predictedLabels);
%confMat = confMat./sum(confMat,2);
%mean(diag(confMat))

% This function simply resizes the images to fit in AlexNet
function I = readFunctionTrain(filename)
% Resize the images to the size required by the network.
I = imread(filename);

I = imresize(I, [227 227]);
end