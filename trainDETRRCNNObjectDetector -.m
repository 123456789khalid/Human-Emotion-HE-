function [detector,info] = trainFasterRCNNObjectDetector(trainingData, network, options, varargin)
%trainRCNNObjectDetector Train an R-CNN deep learning object detector
% Use of this function requires that you have the Deep Learning
% Toolbox(TM) and the Statistics and Machine Learning Toolbox(TM).
%
% detector = trainRCNNObjectDetector(trainingData, network, options) trains
% an R-CNN (Regions with CNN features) based object detector using deep
% learning. An R-CNN detector can be trained to detect multiple object
% classes.
%
% <a href="matlab:helpview(fullfile(docroot,'toolbox','vision','vision.map'),'rcnnConcept')">Learn more about R-CNN.</a>
%
% [..., info] = trainRCNNObjectDetector(...) additionally returns
% information on training progress such as training loss and accuracy. info
% is a struct with the following fields:
%
%   TrainingLoss     - Training loss at each iteration. This is the
%                      combination of the classification and regression
%                      loss used to train the Faster R-CNN network.
%   TrainingAccuracy - Training set accuracy at each iteration.
%   BaseLearnRate    - Learning rate at each iteration.
%
% Inputs:
% -------
% trainingData  - a table with 2 or more columns. The first column must
%                 contain image file names. The images can be grayscale or
%                 true color, and can be in any format supported by IMREAD.
%                 The remaining columns must contain M-by-4 matrices of [x,
%                 y, width, height] bounding boxes specifying object
%                 locations within each image. Each column represents a
%                 single object class, e.g. person, car, dog. The table
%                 variable names define the object class names. You can use
%                 the imageLabeler app to create this table.
%
% network       - Specify the network to train as a SeriesNetwork, an array
%                 of Layer objects, a LayerGraph object, or by name. Valid
%                 network names are listed below, and require installation
%                 of the associated Add-On:
%
%                 <a href="matlab:helpview('deeplearning','alexnet')">'alexnet'</a>
%                 <a href="matlab:helpview('deeplearning','vgg16')">'vgg16'</a>
%                 <a href="matlab:helpview('deeplearning','vgg19')">'vgg19'</a>
%                 <a href="matlab:helpview('deeplearning','resnet18')">'resnet18'</a>
%                 <a href="matlab:helpview('deeplearning','resnet50')">'resnet50'</a>
%                 <a href="matlab:helpview('deeplearning','resnet101')">'resnet101'</a>
%                 <a href="matlab:helpview('deeplearning','inceptionv3')">'inceptionv3'</a>
%                 <a href="matlab:helpview('deeplearning','googlenet')">'googlenet'</a>
%                 <a href="matlab:helpview('deeplearning','inceptionresnetv2')">'inceptionresnetv2'</a>
%                 <a href="matlab:helpview('deeplearning','squeezenet')">'squeezenet'</a>
%                 <a href="matlab:helpview('deeplearning','mobilenetv2')">'mobilenetv2'</a>
%
%                 When the network is specified as a SeriesNetwork, or by
%                 name, the network is automatically adjusted to support
%                 the number of object classes defined within the
%                 trainingData plus an extra "Background" class.
%
%                 When the network is specified by as an array of Layer
%                 objects, or a LayerGraph object, the network must have a
%                 classification layer that supports the number of object
%                 classes defined in the the training data plus a
%                 background class.
%
%                 The SeriesNetwork, Layer, and LayerGraph objects are
%                 available in the Deep Learning Toolbox. See <a href="matlab:doc SeriesNetwork">SeriesNetwork</a>,
%                 <a href="matlab:doc nnet.cnn.layer.Layer">Layer</a>, <a href="matlab:doc nnet.cnn.LayerGraph">LayerGraph</a> documentation for more details.
%
% options       - training options defined by the trainingOptions function
%                 from Deep Learning Toolbox. The training options define
%                 the training parameters of the neural network. 
%
%                 See <a href="matlab:doc trainingOptions">trainingOptions documentation</a> for more  
%                 details.
%
% [...] = trainRCNNObjectDetector(..., Name, Value) specifies additional
% name-value pair arguments described below:
%
% 'PositiveOverlapRange' A two-element vector that specifies a range of
%                        bounding box overlap ratios between 0 and 1.
%                        Region proposals that overlap with ground truth
%                        bounding boxes within the specified range are used
%                        as positive training samples.
%
%                        Default: [0.5 1]
%
% 'NegativeOverlapRange' A two-element vector that specifies a range of
%                        bounding box overlap ratios between 0 and 1.
%                        Region proposals that overlap with ground truth
%                        bounding boxes within the specified range are used
%                        as negative training samples.
%
%                        Default: [0.1 0.5]
%
% 'NumStrongestRegions'  The maximum number of strongest region proposals
%                        to use for generating training samples. Reduce
%                        this value to speed-up processing time at the cost
%                        of training accuracy. Set this to inf to use all
%                        region proposals.
%
%                        Default: 2000
%
% 'BoxRegressionLayer'   The name of a layer in the input network. This
%                        layer's output activations are used as features to
%                        train a regression model for refining the detected
%                        bounding boxes. Valid values are 'auto' or the
%                        name of a layer in the input network. When the
%                        value is 'auto', a layer from the input network is
%                        automatically selected based on the type of input
%                        network:
%
%                        Auto selection of box regression layer
%                        --------------------------------------
%                        * If the input is a SeriesNetwork or an array of
%                          Layers, the last convolution layer is selected.
%
%                        * If the input is a LayerGraph, the source of the
%                          last fully connected layer is selected.
%
%                        Default: 'auto'
%
% [...] = trainRCNNObjectDetector(..., 'RegionProposalFcn', proposalFcn)
% optionally train an R-CNN detector using a custom region proposal
% function, proposalFcn.  If a custom region proposal function is not
% specified, a variant of the EdgeBoxes algorithm is automatically used. A
% custom proposalFcn must have the following functional form:
%
%    [bboxes, scores] = proposalFcn(I)
%
% where the input I is an image defined in the trainingData table. The
% function must return rectangular bounding boxes in an M-by-4 array. Each
% row of bboxes contains a four-element vector, [x, y, width, height]. This
% vector specifies the upper-left corner and size of a bounding box in
% pixels. The function must also return a score for each bounding box in an
% M-by-1 vector. Higher score values indicate that the bounding box is more
% likely to contain an object. The scores are used to select the strongest
% N regions, where N is defined by the value of 'NumStrongestRegions'.
%
% Notes:
% ------
% - trainRCNNObjectDetector supports parallel computing using
%   multiple MATLAB workers. Enable parallel computing using the
%   <a href="matlab:preferences('Computer Vision Toolbox')">preferences dialog</a>.
%
% - This implementation of R-CNN does not train an SVM classifier for each
%   object class.
%
% - The overlap ratio used in 'PositiveOverlapRange' and
%  'NegativeOverlapRange' is defined as area(A intersect B) / area(A union B),
%   where A and B are bounding boxes.
%
% - Use the trainingOptions function to enable or disable verbose printing.
%
% - When the network is a SeriesNetwork, the network is automatically
%   adjusted to support the number of object classes defined within the
%   trainingData plus an extra "Background" class.
%
% - When the network is an array of Layer objects or a LayerGraph object,
%   the network must have a classification layer that supports the number
%   of object classes plus a background class. Use this input type when you
%   want to customize the learning rates of each layer. You may also use
%   this type of input to resume training from a previous training session.
%   This can be useful if the network requires additional rounds of
%   fine-tuning or if you wish to train with additional training data.
%
% Example: Train a stop sign detector
% ------------------------------------
% load('rcnnStopSigns.mat', 'stopSigns', 'layers')
%
% % Add fullpath to image files
% stopSigns.imageFilename = fullfile(toolboxdir('vision'),'visiondata', ...
%     stopSigns.imageFilename);
%
% % Set network training options to use mini-batch size of 32 to reduce GPU
% % memory usage. Lower the InitialLearningRate to reduce the rate at which
% % network parameters are changed. This is beneficial when fine-tuning a
% % pre-trained network and prevents the network from changing too rapidly.
% % Set network training options.
% %  * Lower the InitialLearningRate to reduce the rate at which network
% %    parameters are changed.
% %  * Set the CheckpointPath to save detector checkpoints to a temporary
% %    directory.
% options = trainingOptions('sgdm', ...
%     'MiniBatchSize', 32, ...
%     'InitialLearnRate', 1e-6, ...
%     'MaxEpochs', 10);
%
% % Train the R-CNN detector. Training can take a few minutes to complete.
% [rcnn, info] = trainRCNNObjectDetector(stopSigns, layers, options, 'NegativeOverlapRange', [0 0.3]);
%
% % Test the R-CNN detector on a test image.
% img = imread('stopSignTest.jpg');
%
% [bbox, score, label] = detect(rcnn, img, 'MiniBatchSize', 32);
%
% % Display strongest detection result
% [score, idx] = max(score);
%
% bbox = bbox(idx, :);
% annotation = sprintf('%s: (Confidence = %f)', label(idx), score);
%
% detectedImg = insertObjectAnnotation(img, 'rectangle', bbox, annotation);
%
% figure
% imshow(detectedImg)
%
% % <a href="matlab:showdemo('DeepLearningRCNNObjectDetectionExample')">Learn more about training an R-CNN Object Detector.</a>
%
% See also rcnnObjectDetector, SeriesNetwork, DAGNetwork, trainingOptions,
%          trainNetwork, imageLabeler, trainCascadeObjectDetector,
%          trainFastRCNNObjectDetector, trainFasterRCNNObjectDetector,
%          nnet.cnn.layer.Layer, nnet.cnn.LayerGraph.

% References:
% -----------
% Girshick, Ross, et al. "Rich feature hierarchies for accurate object
% detection and semantic segmentation." Proceedings of the IEEE conference
% on computer vision and pattern recognition. 2014.
%
% Girshick, Ross. "Fast r-cnn." Proceedings of the IEEE International
% Conference on Computer Vision. 2015.
%
% Zitnick, C. Lawrence, and Piotr Dollar. "Edge boxes: Locating object
% proposals from edges." Computer Vision-ECCV 2014. Springer International
% Publishing, 2014. 391-405.

% Copyright 2017-2018 The MathWorks, Inc.

if nargin > 3
    [varargin{:}] = convertStringsToChars(varargin{:});
end

vision.internal.requiresStatisticsToolbox(mfilename);
vision.internal.requiresNeuralToolbox(mfilename);

% Initialize warning logger. Logs warnings issued during training and
% reissues them at end of training when Verbose is true.
vision.internal.cnn.WarningLogger.initialize();

[network, params] = parseInputs(trainingData, network, options, mfilename, varargin{:});

if ischar(network) || isstring(network)
    % Generate R-CNN network requested by user.
    lgraphOrLayers = vision.internal.cnn.RCNNLayers.create(params.NumClasses, network, 'rcnn');
        
    % Fill remaining training parameters.
    params = checkNetworkAndFillRemainingParameters(trainingData, lgraphOrLayers, params);
    
    if string(params.BoxRegressionLayer) == "auto" && string(network) == "squeezenet"
        % SqueezeNet uses a convolution layer as the classification layer
        % instead of a fully connected layer. Therefore, the auto selection
        % approach does not apply. Manually select the layer feeding the
        % last conv layer.
        params.BoxRegressionLayer = 'fire9-concat';
    end
else
    % Check network input by user.
    params = checkNetworkAndFillRemainingParameters(trainingData, network, params);
    
    try
        % Transform network input by user into R-CNN network.
        lgraphOrLayers = vision.internal.cnn.RCNNLayers.create(params.NumClasses, network, 'rcnn');
    catch ME
        throwAsCaller(ME);
        
    end
end
lgraphOrLayers = vision.internal.rcnn.removeAugmentationIfNeeded(lgraphOrLayers,'randcrop');

[detector, ~, info] = rcnnObjectDetector.train(trainingData, lgraphOrLayers, options, params);

%--------------------------------------------------------------------------
function params = checkNetworkAndFillRemainingParameters(trainingData, lgraphOrLayers, params)
% Run network analyzer to get any network related errors.
analysis = nnet.internal.cnn.analyzer.NetworkAnalyzer(lgraphOrLayers);
analysis.applyConstraints();
try
    analysis.throwIssuesIfAny();
catch ME
    throwAsCaller(ME);
end

% Check BBox Feature Layer parameters. Auto selection of this parameter
% happens in rcnnObjectDetector.train() if not provided by user.
bboxFeatureLayerName = iCheckBBoxFeatureLayerName(params.BoxRegressionLayer, analysis, mfilename);
 
vision.internal.cnn.validation.checkNetworkLayers(analysis);

if ~params.IsNetwork
    % check if layers or layerGraph has correct number of output classes
    % for detection task (numClasses + 1 for background).
    vision.internal.cnn.validation.checkNetworkClassificationLayer(analysis, trainingData);
end

inputSize = vision.internal.cnn.imageInputSizeFromNetworkAnalysis(analysis);

params.InputSize          = inputSize;
params.BoxRegressionLayer = char(bboxFeatureLayerName);

%--------------------------------------------------------------------------
function [network, params] = parseInputs(trainingData, network, options, fname, varargin)

vision.internal.cnn.validation.checkGroundTruth(trainingData, fname);

network = vision.internal.cnn.validation.checkNetwork(network, fname, ...
    {'SeriesNetwork', 'nnet.cnn.layer.Layer','nnet.cnn.LayerGraph'}, ...
    vision.internal.cnn.RCNNLayers.SupportedPretrainedNetworks);

vision.internal.cnn.validation.checkTrainingOptions(options, fname);

if options.MiniBatchSize < 4
    error(message('vision:rcnn:miniBatchSizeTooSmall'));
end

p = inputParser;
p.addParameter('RegionProposalFcn', @rcnnObjectDetector.proposeRegions);
p.addParameter('UseParallel', vision.internal.useParallelPreference());
p.addParameter('PositiveOverlapRange', [0.5 1]);
p.addParameter('NegativeOverlapRange', [0.1 0.5]);
p.addParameter('NumStrongestRegions', 2000);
p.addParameter('BoxRegressionLayer', 'auto');
p.parse(varargin{:});

userInput = p.Results;

rcnnObjectDetector.checkRegionProposalFcn(userInput.RegionProposalFcn);

useParallel = vision.internal.inputValidation.validateUseParallel(userInput.UseParallel);

vision.internal.cnn.validation.checkOverlapRatio(userInput.PositiveOverlapRange, fname, 'PositiveOverlapRange');
vision.internal.cnn.validation.checkOverlapRatio(userInput.NegativeOverlapRange, fname, 'NegativeOverlapRange');

vision.internal.cnn.validation.checkStrongestRegions(p.Results.NumStrongestRegions, fname);

params.IsNetwork = isa(network,'SeriesNetwork') || isa(network,'DAGNetwork');

params.NumClasses = width(trainingData) - 1;
params.PositiveOverlapRange          = double(userInput.PositiveOverlapRange);
params.NegativeOverlapRange          = double(userInput.NegativeOverlapRange);
params.RegionProposalFcn             = userInput.RegionProposalFcn;
params.UsingDefaultRegionProposalFcn = ismember('RegionProposalFcn', p.UsingDefaults);
params.NumStrongestRegions           = double(userInput.NumStrongestRegions);
params.UseParallel                   = useParallel;
params.BackgroundLabel               = vision.internal.cnn.uniqueBackgroundLabel(trainingData);

% BoxRegressionLayer will be validated later after network is transformed
% into R-CNN network or after network user input network is validated.
params.BoxRegressionLayer  = userInput.BoxRegressionLayer;

vision.internal.cnn.validation.checkPositiveAndNegativeOverlapRatioDoNotOverlap(params);

%--------------------------------------------------------------------------
function val = iCheckBBoxFeatureLayerName(val, networkAnalysis, fname)
validateattributes(val,{'string','char'},{'scalartext'},fname,'BoxRegressionLayer');
allNames = {networkAnalysis.ExternalLayers.Name};

% partial match not allowed for layer name value.
if string(val) == "auto"
    iErrorIfAnyLayerNameIs('auto',allNames);
else
    % Check that layer name exists in network.
    val = validatestring(val,allNames,fname,'BoxRegressionLayer');
end

%--------------------------------------------------------------------------
function iErrorIfAnyLayerNameIs(name,allNames)
if any(strcmp(name,allNames))
    error(message('vision:rcnn:autoOrNoneNotAllowedAsLayerName'))
end
