Var1 = gTruth.LabelData.Face; % selecting the ground truth
Var2 = gTruth.DataSource.Source;

%%
T1 = table(Var2,Var1);

%%
options = trainingOptions('sgdm', ...
      'MiniBatchSize', 10, ...
      'InitialLearnRate', 1e-3, ...
      'MaxEpochs', 50, ...
      'VerboseFrequency', 200, ...
      'CheckpointPath', tempdir);
  %%
  detector = trainFasterRCNNObjectDetector(T1, layers, options, ...
        'NegativeOverlapRange',[0 0.3], ...
        'PositiveOverlapRange',[0.6 1]);
    %%
    img = imread('C:\Users\zikar\Desktop\MATLAB\Khalid faster RCNN code\Dataset\GettyImages-1092658864_hero-1024x575.jpg');


[bbox, score, label] = detect(detector,img);
detectedImg = insertShape(img,'Rectangle',bbox);
figure
imshow(detectedImg)
    