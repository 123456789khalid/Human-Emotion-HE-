net =  inceptionv3;
layer = 'predictions';
%%
k=1;
for i= 100001:150000
     a= [imdsTrain.Files(i)];
   a = imresize(imread(char(a)),[299 299]);
featuresTrain(k,:) = activations(net,a,layer,'OutputAs','rows');
k=k+1;
i
end