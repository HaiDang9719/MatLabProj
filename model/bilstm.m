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
        %BiLSTM model
        bilstmModel = [ ...
            sequenceInputLayer(embeddingDimension)
            bilstmLayer(numHiddenUnits,'OutputMode','last')
            fullyConnectedLayer(numClasses)
            softmaxLayer
            classificationLayer];
end
%% training and testing model
if (~cfg.dataset.kfoldvalidation)
    
    % create dictionary
    if (~cfg.model.fastTextWordEmbedding)
    else
        enc = wordEncoding(clean_text_train);
        embeddingDimension = 100;
        numWords = enc.NumWords;
            
        %BiLSTM model
        bilstmModel = [ ...
            sequenceInputLayer(inputSize)
            wordEmbeddingLayer(embeddingDimension,numWords)
            bilstmLayer(numHiddenUnits,'OutputMode','last')
            fullyConnectedLayer(numClasses)
            softmaxLayer
            classificationLayer];
    end
    % encoding word with doc2sequence function. Set the threshold number of words based on the document length distribution

    X_train = doc2sequence(enc,clean_text_train,'Length',cfg.dataset.sequenceLength);
    X_val = doc2sequence(enc,clean_text_eval,'Length',cfg.dataset.sequenceLength);
    X_test = doc2sequence(enc,clean_text_test,'Length',cfg.dataset.sequenceLength);

    % training option for the model

    try
        nnet.internal.cnngpu.reluForward(1);
    catch ME
    end
    
    
    %training
    BuildandTrainBiLSTM(bilstmModel,X_val,Y_val,X_test,Y_test,X_train,Y_train,false);
    

else
    bilstm_acc=zeros(cfg.dataset.kfold,1); 
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
            %BiLSTM model
            bilstmModel = [ ...
                sequenceInputLayer(inputSize)
                wordEmbeddingLayer(embeddingDimension,numWords)
                bilstmLayer(numHiddenUnits,'OutputMode','last')
                fullyConnectedLayer(numClasses)
                softmaxLayer
                classificationLayer];
        end
        % encoding word with doc2sequence function. Set the threshold number of words based on the document length distribution

        X_train = doc2sequence(enc,clean_text_train{i},'Length',cfg.dataset.sequenceLength);
        X_val = doc2sequence(enc,clean_text_eval{i},'Length',cfg.dataset.sequenceLength);
        X_test = doc2sequence(enc,clean_text_test{i},'Length',cfg.dataset.sequenceLength);
        
        %training
        [bilstm_acc(i),lsPre(i), lsRe(i), lsFS(i),lsFB(i),lsAUC(i)] = BuildandTrainBiLSTM(bilstmModel,X_val,Y_val{i},X_test,Y_test{i},X_train,Y_train{i},true);
        
    end
    %save model
    saveModel(bilstmModel);
    
    %print result
    fprintf('Result of BiLSTM on test set with kfold cross validation: \n');
    fprintf('Average Accuracy for BiLSTM: %f\n',mean(bilstm_acc)*100);
    fprintf('Average Precision for BiLSTM: %f\n',mean(lsPre));
    fprintf('Average Recal for BiLSTM: %f\n',mean(lsRe));
    fprintf('Average FScore for BiLSTM: %f\n',mean(lsFS));
    fprintf('Average FP for BiLSTM: %f\n',mean(lsFB));
    fprintf('Average AUC for BiLSTM: %f\n',mean(lsAUC));
end

%% BiLSTM model
function[bilstm_acc,bilsPre, bilsRe, bilsFS,bilsFB,bilsAUC] = BuildandTrainBiLSTM(bilstmModel,X_val,Y_val,X_test,Y_test,X_train,Y_train,Kfold)
    
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
    bilstmModel = trainNetwork(X_train,categorical(Y_train),bilstmModel,options);
    %save model
    saveModel(bilstmModel);
    
    %testing
    fprintf('Result of BiLSTM on test set: \n');
    Y_pred_ls = classify(bilstmModel,X_test);
    bilstm_acc = model_Acc(categorical(Y_test),Y_pred_ls);
    fprintf('Accuracy for BiLSTM: %f\n',bilstm_acc*100);

    [bilsPre, bilsRe, bilsFS,bilsFB,bilsAUC] = model_FScore(categorical(Y_test),Y_pred_ls);
    fprintf('Precision for BiLSTM: %f\n',bilsPre);
    fprintf('Recal for BiLSTM: %f\n',bilsRe);
    fprintf('FScore for BiLSTM: %f\n',bilsFS);
    fprintf('FP for BiLSTM: %f\n',bilsFB);
    fprintf('AUC for BiLSTM: %f\n',bilsAUC);
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