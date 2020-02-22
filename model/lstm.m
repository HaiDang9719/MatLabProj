

%% model Configuration.

inputSize = 1;
numHiddenUnits = 180;
numClasses = 2;

%% LSTM model

lstmModel = [ ...
    sequenceInputLayer(inputSize)
    wordEmbeddingLayer(embeddingDimension,numWords)

    lstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

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

    X_train = doc2sequence(enc,clean_text_train,'Length',10);
    X_val = doc2sequence(enc,clean_text_eval,'Length',10);
    X_test = doc2sequence(enc,clean_text_test,'Length',10);

    % training option for the model

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
        'Verbose',false, ...
        'OutputFcn',@(info)stopIfAccuracyNotImproving(info,3));
        %training
        lstmModel = trainNetwork(X_train,categorical(Y_train),lstmModel,options);

    %testing
    fprintf('Result of LSTM on test set: \n');
    Y_pred_ls = classify(lstmModel,X_test);
    lstm_acc = model_Acc(categorical(Y_test),Y_pred_ls);
    fprintf('Accuracy for LSTM: %f\n',lstm_acc*100);

    [lsPre, lsRe, lsFS,lsFB,lsAUC] = model_FScore(categorical(Y_test),Y_pred_ls);
    fprintf('Precision for LSTM: %f\n',lsPre);
    fprintf('Recal for LSTM: %f\n',lsRe);
    fprintf('FScore for LSTM: %f\n',lsFS);
    fprintf('FP for LSTM: %f\n',lsFB);
    fprintf('AUC for LSTM: %f\n',lsAUC);
else
    lstm_acc=zeros(cfg.dataset.kfold,1); 
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
        % LSTM model

        lstmModel = [ ...
            sequenceInputLayer(inputSize)
            wordEmbeddingLayer(embeddingDimension,numWords)

            lstmLayer(numHiddenUnits,'OutputMode','last')
            fullyConnectedLayer(numClasses)
            softmaxLayer
            classificationLayer];
        options = trainingOptions('adam', ...
            'ExecutionEnvironment','auto',...
            'MaxEpochs',10, ...    
            'GradientThreshold',1, ...
            'InitialLearnRate',0.001, ...
            'ValidationData',{X_val,categorical(Y_val)}, ...
            'Verbose',false, ...
            'OutputFcn',@(info)stopIfAccuracyNotImproving(info,3));
            %training
            lstmModel = trainNetwork(X_train,categorical(Y_train),lstmModel,options);

        %testing
        
        Y_pred_ls = classify(lstmModel,X_test);
        lstm_acc(i) = model_Acc(categorical(Y_test),Y_pred_ls);
        

        [lsPre(i), lsRe(i), lsFS(i),lsFB(i),lsAUC(i)] = model_FScore(categorical(Y_test),Y_pred_ls);
        
    end
    fprintf('Result of LSTM on test set: \n');
    fprintf('Accuracy for LSTM: %f\n',mean(lstm_acc)*100);
    fprintf('Precision for LSTM: %f\n',mean(lsPre));
    fprintf('Recal for LSTM: %f\n',mean(lsRe));
    fprintf('FScore for LSTM: %f\n',mean(lsFS));
    fprintf('FP for LSTM: %f\n',mean(lsFB));
    fprintf('AUC for LSTM: %f\n',mean(lsAUC));
end
%% save model

%% predict model

    

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