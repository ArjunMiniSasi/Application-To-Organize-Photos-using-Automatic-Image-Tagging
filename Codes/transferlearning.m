%01:28(30)
% train 100*4=400 test 1 image 
net = alexnet;
layers = net.Layers;
rootFolder = '101_ObjectCategories';
categories = {'leopards','elephant','crocodile','emu'};
imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
imds = splitEachLabel(imds, 10, 'randomize') % we only need 500 images per class
imds.ReadFcn = @readFunctionTrain;

layers = layers(1:end-3);
layers(end+1) = fullyConnectedLayer(64, 'Name', 'special_2');
layers(end+1) = reluLayer;
layers(end+1) = fullyConnectedLayer(4, 'Name', 'fc8_2 ');
layers(end+1) = softmaxLayer;
layers(end+1) = classificationLayer()

layers(end-2).WeightLearnRateFactor = 10;
layers(end-2).WeightL2Factor = 1;
layers(end-2).BiasLearnRateFactor = 20;
layers(end-2).BiasL2Factor = 0;

opts = trainingOptions('sgdm', ...
    'ExecutionEnvironment','gpu',...
    'LearnRateSchedule', 'none',...
    'InitialLearnRate', .01,... 
    'MaxEpochs', 20, ...
    'MiniBatchSize', 256);

convnet = trainNetwork(imds, layers, opts);

picture = imread('image7.png');              % Take a picture    
picture = imresize(picture,[227,227]);  % Resize the picture
label = classify(convnet, picture);        % Classify the picture
       
image(picture);     % Show the picture
title(char(label));
%rootFolder = 'cifar10Test';
%testDS = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
%testDS.ReadFcn = @readFunctionTrain;

%[labels,err_test] = classify(convnet, testDS, 'MiniBatchSize', 64);


%confMat = confusionmat(testDS.Labels, labels);
%confMat = confMat./sum(confMat,2);
%mean(diag(confMat))

% This function simply resizes the images to fit in AlexNet
% Copyright 2017 The MathWorks, Inc.
function I = readFunctionTrain(filename)
% Resize the images to the size required by the network.
I = imread(filename);

I = imresize(I, [227 227]);
end