%% LSTM model - 
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the



%% Initialization
clear; close all; clc
addpath('preprocess');
addpath('evaluation');

%% =========== Part 1: Preprocessing =============
%% Load data

[dataset] = loadData("dataset2Lable.csv");

fprintf('Program paused. Press enter to continue.\n');
pause;

%% partioning data set

[train_set, eval_set, test_set] = partioning(dataset);

text_train = train_set.text;
text_validation = eval_set.text;
text_test = test_set.text;

Y_train = train_set.label;
Y_val = eval_set.label;
Y_test = test_set.label;

fprintf('Program paused. Press enter to continue.\n');
pause;

%% clean data set

[clean_text_train] = textPreprocessing(text_train);
[clean_text_eval] = textPreprocessing(text_validation);
[clean_text_test] = textPreprocessing(text_test);

fprintf('Program paused. Press enter to continue.\n');
pause;
%% visualize data
figure
wordcloud(text_train)

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 2: Word Encoding ================

%% create dictionayry

enc = wordEncoding(clean_text_train);
documentLengths = doclength(clean_text_train);
figure
histogram(documentLengths)
title("Document Lengths")
xlabel("Length")
ylabel("Number of Documents")

fprintf('Program paused. Press enter to continue.\n');
pause;

%% encoding word with doc2sequence function. Set the threshold number of words based on the document length distribution

X_train = doc2sequence(enc,clean_text_train,'Length',25);
X_val = doc2sequence(enc,clean_text_eval,'Length',25);
X_test = doc2sequence(enc,clean_text_test,'Length',25);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 3: LSTM model ==============================
%% model Configuration.

modeConfig = modelConfig();
inputSize = 1;
embeddingDimension = 100;
numWords = enc.NumWords;
numHiddenUnits = 180;
numClasses = numel(categories(categorical(Y_train)));

%% LSTM model

lstmModel = [ ...
    sequenceInputLayer(inputSize)
    wordEmbeddingLayer(embeddingDimension,numWords)
    lstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

%% training option for the model

try
    nnet.internal.cnngpu.reluForward(1);
catch ME
end
options = trainingOptions('adam', ...
    'ExecutionEnvironment','auto',...
    'MaxEpochs',10, ...    
    'GradientThreshold',1, ...
    'InitialLearnRate',0.001, ...
    'ValidationData',{X_val,categorical(Y_val)}, ...
    'Plots','training-progress', ...
    'Verbose',false);

%% training model

ile=gpuArray(0.0001);
lstmModel = trainNetwork(X_train,categorical(Y_train),lstmModel,options);
%% save model

%% predict model

Y_pred = classify(lstmModel,X_test);
lstm_acc = model_Acc(categorical(Y_test),Y_pred);
fprintf('Accuracy for BiLSTM: %f\n',lstm_acc*100);

[lsPre, lsRe, lsFS] = model_FScore(categorical(Y_test),Y_pred);
fprintf('Precision for BiLSTM: %f\n',lsPre);
fprintf('Recal for BiLSTM: %f\n',lsRe);
fprintf('FScore for BiLSTM: %f\n',lsFS);

%% ================ Part 4: BiLSTM model ==============================
%% model Configuration.

modeConfig = modelConfig();
inputSize = 1;
embeddingDimension = 100;
numWords = enc.NumWords;
numHiddenUnits = 180;
numClasses = numel(categories(categorical(Y_train)));

%% BiLSTM model

bilstmModel = [ ...
    sequenceInputLayer(inputSize)
    bilstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

%% training option for the model

try
    nnet.internal.cnngpu.reluForward(1);
catch ME
end
options = trainingOptions('adam', ...
    'ExecutionEnvironment','auto',...
    'MaxEpochs',10, ...    
    'GradientThreshold',1, ...
    'InitialLearnRate',0.001, ...
    'ValidationData',{X_val,categorical(Y_val)}, ...
    'Plots','training-progress', ...
    'Verbose',false);

%% training model

ile=gpuArray(0.0001);
bilstmModel = trainNetwork(X_train,categorical(Y_train),bilstmModel,options);

%% predict model

Y_pred = classify(bilstmModel,X_test);

bilstm_acc = model_Acc(categorical(Y_test),Y_pred);
fprintf('Accuracy for BiLSTM: %f\n',bilstm_acc*100);

[biPre, biRe, biFS] = model_FScore(categorical(Y_test),Y_pred);
fprintf('Precision for BiLSTM: %f\n',biPre);
fprintf('Recal for BiLSTM: %f\n',biRe);
fprintf('FScore for BiLSTM: %f\n',biFS);

%%
rmpath('preprocess');
rmpath('evaluation');