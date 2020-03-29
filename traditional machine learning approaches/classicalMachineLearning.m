clear; clc;

%%%%%A) self-created data set
% Load data
dataFirst = readtable("dataset2Label.csv");
dataFirst.Properties.VariableNames{8} = 'text';
dataFirst.Properties.VariableNames{9} = 'label';
dataFirst.label = categorical(dataFirst.label);
dataFirst = dataFirst(:,[8,9]);
%head(dataset2Lable)

% Partition the data into a training partition and a held-out test set. Specify the holdout percentage to be 10%. 
cvp = cvpartition(dataFirst.label,'Holdout',0.2);
%cvp = cvpartition(n,'KFold',0.2)
dataFirst_Train = dataFirst(cvp.training,:);
dataFirst_Test = dataFirst(cvp.test,:);

% Extract the text data and labels from the tables.
text_DataFirst_Train = dataFirst_Train.text;
text_DataFirst_Test = dataFirst_Test.text;
Y_DataFirst_Train = dataFirst_Train.label;
Y_DataFirst_Test = dataFirst_Test.label;

% Tokenize and preprocesse the text data
clean_text_DataFirst_Train = textPreprocessing(text_DataFirst_Train);

% Create a bag-of-words model from the tokenized documents
bag_DataFirst = bagOfWords(clean_text_DataFirst_Train);

% Remove words from the bag-of-words model that do not appear more than two times in total. Remove any documents containing no words from the bag-of-words model, and remove the corresponding entries in labels.
bag_DataFirst = removeInfrequentWords(bag_DataFirst,2);
[bag_DataFirst,idx] = removeEmptyDocuments(bag_DataFirst);
Y_DataFirst_Train(idx) = [];

% Train supervised classification models (naive bayes, svm, random forest) using tf-ifd
X_DataFirst_Train = bag_DataFirst.Counts;
%X_DataFirst_Train = tfidf(bag_DataFirst);
naiveBayes_DataFirst = fitcnb(X_DataFirst_Train,Y_DataFirst_Train); %naive bayes
svm_DataFirst = fitcsvm(X_DataFirst_Train,Y_DataFirst_Train); %svm
randomForest = TreeBagger(500,X_DataFirst_Train,Y_DataFirst_Train); %random forest

% Predict the labels of the test data using the trained model
% Preprocess the test data using the same preprocessing steps as the
% training data. Encode the resulting test documents according to the
% tf-ifd model
clean_text_DataFirst_Test = textPreprocessing(text_DataFirst_Test);
X_DataFirst_Test = encode(bag_DataFirst, clean_text_DataFirst_Test);

% Predict the labels of the test data using the trained model and calculate the classification accuracy.
Y_DataFirst_Pred_naiveBayes = predict(naiveBayes_DataFirst, X_DataFirst_Test);
acc = sum(Y_DataFirst_Pred_naiveBayes == Y_DataFirst_Test)/numel(Y_DataFirst_Test)






%%%%%A) self-created data set
% Load data
dataSecond = readtable("AdHocAnnouncements.csv");
dataSecond.Properties.VariableNames{2} = 'text';
dataSecond.Properties.VariableNames{6} = 'label';
dataSecond.label = categorical(dataSecond.label);
dataSecond = dataSecond(:,[2,6]);