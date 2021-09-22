%%
% Times noted below were based on a i5-6600k CPU 
%
% link below holds all dependencies to run methods 1 and 2. 
% https://rb.gy/k2jows
%
% 1. Download Python Script and models here ----> https://rb.gy/k2jows
% 2. Place  "facenet_keras.h5", Script.exe", and "VGG_model.h5" in the same
%    directory as "Evaulation.m", "faceNet.m", and "deepVGG.m"
% 3. Run "Evaulation.m
%%
t = [10,20,15,5];
a = [5,6,10,20];
%test = 1/5 * 
r = sqrt(sum((t - a) .^ 2));
disp(r)


clear all;
close all;
trainPath = 'FaceDatabase\Train\'; % provide full path here
testPath = 'FaceDatabase\Test\';
%% Baseline Method - Cross-correlation [Accuracy = 25.37%] [Time =  ~34s]
% tic;
% outputLabel = FaceRecognition_cc(trainPath, testPath);
% baseLineTime = toc;
% 
% load testLabel
% correctP = 0;
% for i=1:size(testLabel,1)
%     if strcmp(outputLabel(i, :), testLabel(i, :))
%         correctP = correctP + 1;
%     end
% end
% recAccuracy = correctP / size(testLabel, 1) * 100  %Recognition accuracy%


%% Method 1 - FaceNet (One-shot Learning) [Accuracy = 79.47%] [Time =  ~82s]
% tic;
% outputLabel1 = faceNet(trainPath, testPath);
% method1Time = toc;
% 
% load testLabel
% correctP = 0;
% for i=1:size(testLabel, 1)
%  
%    if strcmp(outputLabel1(i, :), testLabel(i, :))
%        correctP = correctP + 1;
%    end
% 
% end
% 
% recAccuracy = correctP / size(testLabel, 1) * 100  %Recognition accuracy%


%% Method 2 - Deep VGG (One-shot Learning) [Accuracy = 87.72%] [Time = ~256s]
% tic;
% outputLabel2 = deepVGG(trainPath, testPath);
% method2Time = toc;
% 
% load testLabel
% correctP = 0;
% for i = 1:size(testLabel, 1)
%    if strcmp(outputLabel2(i, :), testLabel(i, :))
%        correctP = correctP + 1;
%    end
% end
% recAccuracy = correctP / size(testLabel,1) * 100  %Recognition accuracy%

