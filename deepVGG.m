function  [outputLabel] = deepVGG(trainPath, testPath)


% Generate config file to pass variables to .exe file
fileID = fopen("config.txt", 'w');
fprintf(fileID,'%s', trainPath);
fprintf(fileID,"\n");
fprintf(fileID,'%s', testPath);
fprintf(fileID,"\n");
fprintf(fileID,'%s', "(224, 224)");
fprintf(fileID,"\n");
fprintf(fileID,'%s', "VGG_model.h5");
fprintf(fileID,"\n");
fprintf(fileID,'%s', "vectorised_X_train.mat");
fprintf(fileID,"\n"); 
fprintf(fileID,'%s', "vectorised_X_test.mat");
fclose(fileID);

% Call Python script
system('Script.exe');

folderNames=ls(trainPath);
labelImgSet=folderNames(3:end,:);

% Read file genearted from script
X_test = struct2cell(load('vectorised_X_test.mat'));
X_train = struct2cell(load('vectorised_X_train.mat'));

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
delete config.txt
delete vectorised_X_train.mat
delete vectorised_X_test.mat

end