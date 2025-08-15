net =  inceptionv3;
layer = 'predictions';
%%
k=1;
for i= 54074:86295
     a= [imdsValidation.Files(i)];
   a = imresize(imread(char(a)),[299 299]);
featuresTest1(k,:) = activations(net,a,layer,'OutputAs','rows');
k=k+1;
i
end