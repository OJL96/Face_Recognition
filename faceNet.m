function  [outputLabel] = faceNet(trainPath, testPath)


% Generate config file to pass variables to .exe file
fileID = fopen("config.txt", 'w');
fprintf(fileID,'%s', trainPath);
fprintf(fileID,"\n");
fprintf(fileID,'%s', testPath);
fprintf(fileID,"\n");
fprintf(fileID,'%s', "(160, 160)");
fprintf(fileID,"\n");
fprintf(fileID,'%s', "facenet_keras.h5");
fprintf(fileID,"\n");
fprintf(fileID,'%s', "embedded_X_train.mat");
fprintf(fileID,"\n");
fprintf(fileID,'%s', "embedded_X_test.mat");
fclose(fileID);

% Call Python script
system('Script.exe');

folderNames=ls(trainPath);
labelImgSet=folderNames(3:end,:);

% Read file genearted from script
X_test = struct2cell(load('embedded_X_test.mat'));
X_train = struct2cell(load('embedded_X_train.mat'));

X_test = X_test{1};
X_train = X_train{1};

outputLabel = strings(size(X_test, 1), 1);

% Loop over values from generated files for comparison
for i = 1:size(X_test, 1)
    %euclidean_dist = pdist2(X_test(i, :), X_train, 'euclidean');
    cosine_dist = pdist2(X_test(i, :), X_train, 'cosine');

    [~, argmin] = min(cosine_dist);
    outputLabel(i, 1) = labelImgSet(argmin, :);
     
end

% Delete files we don't need anymore
% You can comment out the code below if you wish to see each file's content
delete config.txt
delete embedded_X_test.mat
delete embedded_X_train.mat
end


