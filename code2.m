net =  inceptionv3;
layer = 'predictions';
%%
k=1;
for i= 143534:201356
     a= [imdsTrain.Files(i)];
   a = imresize(imread(char(a)),[299 299]);
featuresTrain3(k,:) = activations(net,a,layer,'OutputAs','rows');
k=k+1;
i
end
