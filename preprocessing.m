
clear; close all; clc
initialization();

if ~(exist('cfg'))
    cfg = modelConfig();
end
if  cfg.execMode == "test"
    cfg.dataset.kfoldvalidation = false;
end
%Load dataset
fprintf('Load dataset.\n');
[dataset] = loadData(cfg.dataset.path);

%Preprocess data
fprintf('Preprocess dataset.\n');
if (~cfg.dataset.kfoldvalidation)
    fprintf('Preprocess dataset as normal.\n');
    [train_set, eval_set, test_set] = partioning(dataset,0);

    text_train = train_set.text;
    text_validation = eval_set.text;
    text_test = test_set.text;

    Y_train = train_set.label;
    Y_val = eval_set.label;
    Y_test = test_set.label;

    [clean_text_train] = textPreprocessing(text_train);
    [clean_text_eval] = textPreprocessing(text_validation);
    [clean_text_test] = textPreprocessing(text_test);
else
    [~, ~, ~, fold] = partioning(dataset,cfg.dataset.kfold);
    clean_text_train = cell(cfg.dataset.kfold,1);
    clean_text_eval = cell(cfg.dataset.kfold,1);
    clean_text_test = cell(cfg.dataset.kfold,1);
    Y_train=cell(cfg.dataset.kfold,1);
    Y_val=cell(cfg.dataset.kfold,1);
    Y_test=cell(cfg.dataset.kfold,1);
    for i=1:cfg.dataset.kfold
        % Call index of training & testing sets
        trainIdx=fold.training(i); testIdx=fold.test(i);
        % Call training & testing features and labels
        train_set=dataset(trainIdx,:); 
        testset=dataset(testIdx,:); 
        
        cvp = cvpartition(testset.label,'HoldOut',0.5);
        eval_set = testset(training(cvp),:);
        test_set = testset(test(cvp),:);
        
        text_train = train_set.text;
        text_validation = eval_set.text;
        text_test = test_set.text;

        Y_train{i} = train_set.label;
        Y_val{i} = eval_set.label;
        Y_test{i} = test_set.label;
        
        [clean_text_train{i}] = textPreprocessing(text_train);
        [clean_text_eval{i}] = textPreprocessing(text_validation);
        [clean_text_test{i}] = textPreprocessing(text_test);
    end
end

function initialization()
    
    addpath('preprocess');
    addpath('evaluation');
    addpath('model');
    addpath('save_models');
end