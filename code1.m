imds = imageDatastore('C:\Users\Muhammad Shoaib\Desktop\khalid DataSet\affactnet results\train_classification','IncludeSubfolders',true,'LabelSource','foldernames');
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');
%%
net =  inceptionv3;
layer = 'predictions';
%%
k=1;
for i= 43720:100000
     a= [imdsTrain.Files(i)];
   a = imresize(imread(char(a)),[299 299]);
featuresTrain1(k,:) = activations(net,a,layer,'OutputAs','rows');
k=k+1;
i
end
%%
k=1;
for i= 54074:86295
     a=[imdsValidation.Files(i)];
      a = imresize(imread(char(a)),[299 299]);
featuresTest(k,:) = activations(net,a,layer,'OutputAs','rows');
k=k+1;
i
end
%%

NoTrain = 3886
NoTest = 1652
T1 = readtable('a.xlsx');
tableTrain = table(T1(1:NoTrain,1:1000),categorical(imdsTrain.Labels(1:NoTrain,1)))
%
T2 = readtable('b.xlsx');
tableTest = table(T2(1:NoTest,1:1000),categorical(imdsValidation.Labels(1:NoTest,1)))
%
t1 = table(categorical(imdsTrain.Labels(1:NoTrain,1)));
tt = [tableTrain.Var1(1:NoTrain,:),t1];
%%
t2 = table(categorical(imdsValidation.Labels(1:NoTest,1)));
ttt = [T2,t2];
%%

numFeatures = 1000;
numClasses = 104;
 
layers = [
    featureInputLayer(numFeatures,'Normalization', 'zscore')
    fullyConnectedLayer(50)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];
% layers = [
%     featureInputLayer(numFeatures,'Normalization', 'zscore')
%     fullyConnectedLayer(numClasses)
%     softmaxLayer
%     classificationLayer];
miniBatchSize = 64;

options = trainingOptions('adam', ...
    'MaxEpochs',200, ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle','every-epoch', ...
    'ValidationData',ttt, ...
    'Plots','training-progress', ...
    'Verbose',false);
%%
net = trainNetwork(tt,layers,options);
%%
Y_Pred  = classify(net,featuresTest)
%%
plotconfusion(categorical(imdsValidation.Labels),Y_Pred)
