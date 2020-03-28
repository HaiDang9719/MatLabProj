%% model Configuration.

inputSize = 1;
numHiddenUnits = 180;
numClasses = cfg.dataset.numClass;

%% Pretrained word embedding
if (cfg.model.fastTextWordEmbedding)
        enc = fastTextWordEmbedding;
        embeddingDimension = enc.Dimension;
        words = enc.Vocabulary;
        numWords = numel(words);
        %LSTM model
        lstmModel = [ ...
            sequenceInputLayer(embeddingDimension)
            lstmLayer(numHiddenUnits,'OutputMode','last')
            fullyConnectedLayer(numClasses)
            softmaxLayer
            classificationLayer];
end
%% training and testing model
if (~cfg.dataset.kfoldvalidation)
    
    % train word embedding end-to-end
    if (~cfg.model.fastTextWordEmbedding)
        enc = wordEncoding(clean_text_train);
        embeddingDimension = 100;
        numWords = enc.NumWords;
        %LSTM model
        lstmModel = [ ...
            sequenceInputLayer(inputSize)
            wordEmbeddingLayer(embeddingDimension,numWords)
            lstmLayer(numHiddenUnits,'OutputMode','last')
            fullyConnectedLayer(numClasses)
            softmaxLayer
            classificationLayer];
    end
    % encoding word with doc2sequence function. Set the threshold number of words based on the document length distribution

    X_train = doc2sequence(enc,clean_text_train,'Length',cfg.dataset.sequenceLength);
    X_val = doc2sequence(enc,clean_text_eval,'Length',cfg.dataset.sequenceLength);
    X_test = doc2sequence(enc,clean_text_test,'Length',cfg.dataset.sequenceLength);
    
    %training
    BuildandTrainLSTM(lstmModel,X_val,Y_val,X_test,Y_test,X_train,Y_train,false)
else
    lstm_acc=zeros(cfg.dataset.kfold,1); 
    lsPre=zeros(cfg.dataset.kfold,1); 
    lsRe=zeros(cfg.dataset.kfold,1); 
    lsFS=zeros(cfg.dataset.kfold,1); 
    lsFB=zeros(cfg.dataset.kfold,1); 
    lsAUC=zeros(cfg.dataset.kfold,1); 
    for i=1:cfg.dataset.kfold
        % create dictionary
        if (~cfg.model.fastTextWordEmbedding)
            enc = wordEncoding(clean_text_train{i});
            embeddingDimension = 100;
            numWords = enc.NumWords;
            % LSTM model
            lstmModel = [ ...
                sequenceInputLayer(inputSize)
                wordEmbeddingLayer(embeddingDimension,numWords)
                lstmLayer(numHiddenUnits,'OutputMode','last')
                fullyConnectedLayer(numClasses)
                softmaxLayer
                classificationLayer];
        end
        % encoding word with doc2sequence function. Set the threshold number of words based on the document length distribution

        X_train = doc2sequence(enc,clean_text_train{i},'Length',cfg.dataset.sequenceLength);
        X_val = doc2sequence(enc,clean_text_eval{i},'Length',cfg.dataset.sequenceLength);
        X_test = doc2sequence(enc,clean_text_test{i},'Length',cfg.dataset.sequenceLength);
        
        %training
        [lstm_acc(i),lsPre(i), lsRe(i), lsFS(i),lsFB(i),lsAUC(i)] = BuildandTrainLSTM(lstmModel,X_val,Y_val{i},X_test,Y_test{i},X_train,Y_train{i},true);
        
    end
    %save model
    saveModel(lstmModel);
    
    %print result
    fprintf('Result of LSTM on test set with kfold cross validation: \n');
    fprintf('Average Accuracy for LSTM: %f\n',mean(lstm_acc)*100);
    fprintf('Average Precision for LSTM: %f\n',mean(lsPre));
    fprintf('Average Recal for LSTM: %f\n',mean(lsRe));
    fprintf('Average FScore for LSTM: %f\n',mean(lsFS));
    fprintf('Average FP for LSTM: %f\n',mean(lsFB));
    fprintf('Average AUC for LSTM: %f\n',mean(lsAUC));
end

%% BiLSTM model
function[lstm_acc,lsPre, lsRe, lsFS,lsFB,lsAUC] = BuildandTrainLSTM(lstmModel,X_val,Y_val,X_test,Y_test,X_train,Y_train,Kfold)
    
    %Option
    if (Kfold)
        options = trainingOptions('adam', ...
            'ExecutionEnvironment','auto',...
            'MaxEpochs',10, ...    
            'GradientThreshold',1, ...
            'InitialLearnRate',0.001, ...
            'ValidationData',{X_val,categorical(Y_val)}, ...
            'Verbose',false, ...
            'OutputFcn',@(info)stopIfAccuracyNotImproving(info,3));
    else
         options = trainingOptions('adam', ...
            'ExecutionEnvironment','auto',...
            'MaxEpochs',10, ...    
            'GradientThreshold',1, ...
            'InitialLearnRate',0.001, ...
            'ValidationData',{X_val,categorical(Y_val)}, ...
            'Plots','training-progress', ...
            'Verbose',false, ...
            'OutputFcn',@(info)stopIfAccuracyNotImproving(info,3));
    end
    %training
    lstmModel = trainNetwork(X_train,categorical(Y_train),lstmModel,options);
    
    %save model
    saveModel(lstmModel);
    
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
end

%% save model
function saveModel(lstmModel)
    save('save_models/lstmModel','lstmModel')
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