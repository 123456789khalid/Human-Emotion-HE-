%Alexnet Model is Modified to detect various objects according to its training
load('googleLayers') % Load Alexnet Model

%% Ground Truth of annotated objects and regions i.e. Robot, Ball, Goal and field Lines
img = Face.DataSource
img2 = img.Source
img_lab = Face.LabelData.Face
new = table(img2,img_lab)
%% Grounping images and their corresponding groud truth (bounding box location) in a table
% T1 = table(img,gT_Robot); 

%% Regional Convolution Neural Network option and parameters selection 
options = trainingOptions('sgdm', ...
  'MiniBatchSize', 4, ...
  'InitialLearnRate', 000.1, ...
  'MaxEpochs', 10, ...
  'ExecutionEnvironment','gpu');
  

%% Training of R-CNN Model 
%Model for Robot Detection
rcnn = trainRCNNObjectDetector(new, inceptionLayers, options, 'NegativeOverlapRange', [0 0.3]);

%%
% Validating the RCNN for detecting Robot, Ball, Goal and Field Lines

  
  test_img = imread('C:\Users\HP\Desktop\Khalid Code\newDataset\img-15.jpg'); %Load the images to detect various regions
% test_img = imresize(test_img,[350 350]); %Resize the Test image to a Resolution of 350x350 Pixels 

  %bbox will consists of objects coordinates
  %score represent the detection accuracy 
  %label consists of information about the type of object 
%%
%Detect Robot if present in the image, 
ok = detect(rcnn, test_img);
%%


detect_robot = insertObjectAnnotation(test_img, 'rectangle', ok, 'Face');

%Display the image and detected objects
figure
imshow(detect_robot)
%%
newface = imcrop(test_img,ok(1,:))
