

%% model Configuration.

inputSize = 1;
numHiddenUnits = 180;
numClasses = 2;

%% training and testing model
if (~cfg.dataset.kfoldvalidation)
    
    % create dictionary
    if (cfg.model.fastTextWordEmbedding)
        enc = fastTextWordEmbedding;
        embeddingDimension = enc.Dimension;
        words = enc.Vocabulary;
        numWords = numel(words);
    else
        enc = wordEncoding(clean_text_train);
        embeddingDimension = 100;
        numWords = enc.NumWords;
    end
    % encoding word with doc2sequence function. Set the threshold number of words based on the document length distribution

    X_train = doc2sequence(enc,clean_text_train,'Length',16);
    X_val = doc2sequence(enc,clean_text_eval,'Length',16);
    X_test = doc2sequence(enc,clean_text_test,'Length',16);

    % training option for the model

    try
        nnet.internal.cnngpu.reluForward(1);
    catch ME
    end
    
    %BiLSTM model
    bilstmModel = [ ...
    sequenceInputLayer(inputSize)
    wordEmbeddingLayer(embeddingDimension,numWords)
    bilstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];
    
    %Option
    options = trainingOptions('adam', ...
        'ExecutionEnvironment','auto',...
        'MaxEpochs',10, ...    
        'GradientThreshold',1, ...
        'InitialLearnRate',0.001, ...
        'ValidationData',{X_val,categorical(Y_val)}, ...
        'Plots','training-progress', ...
        'Verbose',false, ...
        'OutputFcn',@(info)stopIfAccuracyNotImproving(info,3));
    
    %training
    bilstmModel = trainNetwork(X_train,categorical(Y_train),bilstmModel,options);
    
    %save model
    saveModel(bilstmModel);
    
    %testing
    fprintf('Result of BiLSTM on test set: \n');
    Y_pred_ls = classify(bilstmModel,X_test);
    bilstm_acc = model_Acc(categorical(Y_test),Y_pred_ls);
    fprintf('Accuracy for BiLSTM: %f\n',bilstm_acc*100);

    [lsPre, lsRe, lsFS,lsFB,lsAUC] = model_FScore(categorical(Y_test),Y_pred_ls);
    fprintf('Precision for BiLSTM: %f\n',lsPre);
    fprintf('Recal for BiLSTM: %f\n',lsRe);
    fprintf('FScore for BiLSTM: %f\n',lsFS);
    fprintf('FP for BiLSTM: %f\n',lsFB);
    fprintf('AUC for BiLSTM: %f\n',lsAUC);
else
    bilstm_acc=zeros(cfg.dataset.kfold,1); 
    lsPre=zeros(cfg.dataset.kfold,1); 
    lsRe=zeros(cfg.dataset.kfold,1); 
    lsFS=zeros(cfg.dataset.kfold,1); 
    lsFB=zeros(cfg.dataset.kfold,1); 
    lsAUC=zeros(cfg.dataset.kfold,1); 
    for i=1:cfg.dataset.kfold
        % create dictionary
        if (cfg.model.fastTextWordEmbedding)
            enc = fastTextWordEmbedding;
            embeddingDimension = enc.Dimension;
            words = enc.Vocabulary;
            numWords = numel(words);
        else
            enc = wordEncoding(clean_text_train{i});
            embeddingDimension = 100;
            numWords = enc.NumWords;
        end
        % encoding word with doc2sequence function. Set the threshold number of words based on the document length distribution

        X_train = doc2sequence(enc,clean_text_train{i},'Length',10);
        X_val = doc2sequence(enc,clean_text_eval{i},'Length',10);
        X_test = doc2sequence(enc,clean_text_test{i},'Length',10);

        % training option for the model

        try
            nnet.internal.cnngpu.reluForward(1);
        catch ME
        end
        
        %BiLSTM model
        bilstmModel = [ ...
        sequenceInputLayer(inputSize)
        wordEmbeddingLayer(embeddingDimension,numWords)
        bilstmLayer(numHiddenUnits,'OutputMode','last')
        fullyConnectedLayer(numClasses)
        softmaxLayer
        classificationLayer];
        
        %Option
        options = trainingOptions('adam', ...
            'ExecutionEnvironment','auto',...
            'MaxEpochs',10, ...    
            'GradientThreshold',1, ...
            'InitialLearnRate',0.001, ...
            'ValidationData',{X_val,categorical(Y_val{i})}, ...
            'Verbose',false, ...
            'OutputFcn',@(info)stopIfAccuracyNotImproving(info,3));
        
        %training
        bilstmModel = trainNetwork(X_train,categorical(Y_train{i}),bilstmModel,options);

        %testing
        
        Y_pred_ls = classify(bilstmModel,X_test);
        bilstm_acc(i) = model_Acc(categorical(Y_test{i}),Y_pred_ls);
        

        [lsPre(i), lsRe(i), lsFS(i),lsFB(i),lsAUC(i)] = model_FScore(categorical(Y_test{i}),Y_pred_ls);
        
    end
    %save model
    saveModel(bilstmModel);
    
    %print result
    fprintf('Result of BiLSTM on test set with kfold cross validation: \n');
    fprintf('Accuracy for BiLSTM: %f\n',mean(bilstm_acc)*100);
    fprintf('Precision for BiLSTM: %f\n',mean(lsPre));
    fprintf('Recal for BiLSTM: %f\n',mean(lsRe));
    fprintf('FScore for BiLSTM: %f\n',mean(lsFS));
    fprintf('FP for BiLSTM: %f\n',mean(lsFB));
    fprintf('AUC for BiLSTM: %f\n',mean(lsAUC));
end
%% save model
function saveModel(bilstmModel)
    save('save_models/bilstmModel','bilstmModel')
end

%% early stoping function
function stop = stopIfAccuracyNotImproving(info,N)

stop = false;

% Keep track of the best validation accuracy and the number of validations for which
% there has not been an improvement of the accuracy.
persistent bestValAccuracy
persistent valLag

% Clear the variables when training starts.
if info.State == "start"
    bestValAccuracy = 0;
    valLag = 0;
    
elseif ~isempty(info.ValidationLoss)
    
    % Compare the current validation accuracy to the best accuracy so far,
    % and either set the best accuracy to the current accuracy, or increase
    % the number of validations for which there has not been an improvement.
    if info.ValidationAccuracy > bestValAccuracy
        valLag = 0;
        bestValAccuracy = info.ValidationAccuracy;
    else
        valLag = valLag + 1;
    end
    
    % If the validation lag is at least N, that is, the validation accuracy
    % has not improved for at least N validations, then return true and
    % stop training.
    if valLag >= N
        stop = true;
    end
    
end

end