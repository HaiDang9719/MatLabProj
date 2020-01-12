%% LSTM model - 
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the



%% Initialization
clear; close all; clc
addpath('preprocess');

%% =========== Part 1: Preprocessing =============
%% Load data

[dataset] = loadData("./data/train.csv");

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

X_train = doc2sequence(enc,clean_text_train,'Length',210);
X_val = doc2sequence(enc,clean_text_eval,'Length',210);
X_test = doc2sequence(enc,clean_text_test,'Length',210);

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
    'ExecutionEnvironment','cpu',...
    'MaxEpochs',10, ...    
    'GradientThreshold',1, ...
    'InitialLearnRate',0.001, ...
    'ValidationData',{X_val,categorical(Y_val)}, ...
    'Plots','training-progress', ...
    'Verbose',false);

%% training model
ile=gpuArray(0.0001);
lstmModel = trainNetwork(X_train,categorical(Y_train),lstmModel,options);

%% =============== Part 4: Stochastic Gradient Descent Training ==========
% Given the loss function, now you will implement Stochastic Gradient
% Descent (SGD) (trainSGD.m) to train the neural networks. 
%

learning_rate = 0.5;
lambda = 5e-6;
num_iters = 10000;
batch_size = 200;
tic
[params,L_history] = trainSGD(params,train.X,train.y,learning_rate,...
    lambda,num_iters,batch_size);
fprintf('Optimization took %f seconds.\n', toc);
figure;
plot(L_history);
xlabel('Iteration','FontSize',20);
ylabel('Loss','FontSize',20);
pause;

%% ================= Part 5: Implement Predict ===========================
%  After training the neural network, we would like to use it to predict
%  the labels. You will now implement the "nnPredict" function to use the
%  neural network to predict the labels of the training set and test set. 
%  You will achieve around 100% training accuracy and around 98.0% test accuracy.
%

p = nnPredict(params,train.X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(p == train.y)) * 100);
p = nnPredict(params,test.X);
fprintf('\nTest Set Accuracy: %f\n', mean(double(p == test.y)) * 100);


%%
rmpath('preprocess');
