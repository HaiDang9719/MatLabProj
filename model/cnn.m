if (~cfg.dataset.kfoldvalidation)
    
    % create dictionary
    if (cfg.model.fastTextWordEmbedding)
        enc = fastTextWordEmbedding;
        embeddingDimension = enc.Dimension;
        words = enc.Vocabulary;
        numWords = numel(words);
    else
        enc = trainWordEmbedding(clean_text_train);
        embeddingDimension = 100;
 
    end
   
        
    % encoding word with doc2sequence function. Set the threshold number of words based on the document length distribution

    X_train = doc2sequence(enc,clean_text_train,'Length',cfg.dataset.sequenceLength);
    X_val = doc2sequence(enc,clean_text_eval,'Length',cfg.dataset.sequenceLength);
    X_test = doc2sequence(enc,clean_text_test,'Length',cfg.dataset.sequenceLength);

    predictorsTrain = cellfun(@(X) permute(X,[3 2 1]),X_train,'UniformOutput',false);
    predictorsTest = cellfun(@(X) permute(X,[3 2 1]),X_test,'UniformOutput',false);
    predictorsEval = cellfun(@(X) permute(X,[3 2 1]),X_val,'UniformOutput',false);
    
    responsesTrain = categorical(Y_train,unique(Y_train));
    responsesTest = categorical(Y_test,unique(Y_test));
    responsesEval = categorical(Y_val,unique(Y_val));
    
    dataTransformedTrain = table(predictorsTrain,responsesTrain);
    dataTransformedTest = table(predictorsTest,responsesTest);
    dataTransformedEval = table(predictorsEval,responsesEval);
    numClasses = cfg.dataset.numClass;
    sequenceLength = cfg.dataset.sequenceLength;
    buildandTrainCNN(embeddingDimension,Y_train,Y_test,sequenceLength,numClasses,dataTransformedEval,dataTransformedTest,dataTransformedTrain, false);

elseif (cfg.dataset.kfoldvalidation)
    cnn_acc=zeros(cfg.dataset.kfold,1); 
    cnnPre=zeros(cfg.dataset.kfold,1); 
    cnnRe=zeros(cfg.dataset.kfold,1); 
    cnnFS=zeros(cfg.dataset.kfold,1); 
    cnnFB=zeros(cfg.dataset.kfold,1); 
    cnnAUC=zeros(cfg.dataset.kfold,1); 
    for i=1:cfg.dataset.kfold
        % create dictionary
        if (cfg.model.fastTextWordEmbedding)
            enc = fastTextWordEmbedding;
            embeddingDimension = enc.Dimension;
            words = enc.Vocabulary;
            numWords = numel(words);
        else
            enc = trainWordEmbedding(clean_text_train{i});
            embeddingDimension = 100;

        end

        % encoding word with doc2sequence function. Set the threshold number of words based on the document length distribution

        X_train = doc2sequence(enc,clean_text_train{i},'Length',cfg.dataset.sequenceLength);
        X_val = doc2sequence(enc,clean_text_eval{i},'Length',cfg.dataset.sequenceLength);
        X_test = doc2sequence(enc,clean_text_test{i},'Length',cfg.dataset.sequenceLength);

        predictorsTrain = cellfun(@(X) permute(X,[3 2 1]),X_train,'UniformOutput',false);
        predictorsTest = cellfun(@(X) permute(X,[3 2 1]),X_test,'UniformOutput',false);
        predictorsEval = cellfun(@(X) permute(X,[3 2 1]),X_val,'UniformOutput',false);

        responsesTrain = categorical(Y_train{i},unique(Y_train{i}));
        responsesTest = categorical(Y_test{i},unique(Y_test{i}));
        responsesEval = categorical(Y_val{i},unique(Y_val{i}));

        dataTransformedTrain = table(predictorsTrain,responsesTrain);
        dataTransformedTest = table(predictorsTest,responsesTest);
        dataTransformedEval = table(predictorsEval,responsesEval);
        numClasses = cfg.dataset.numClass;
        sequenceLength = cfg.dataset.sequenceLength;
        [cnn_acc(i),cnnPre(i),cnnRe(i),cnnFS(i),cnnFB(i),cnnAUC(i)] = buildandTrainCNN(embeddingDimension,Y_train{i},Y_test{i},sequenceLength,numClasses,dataTransformedEval,dataTransformedTest,dataTransformedTrain, true);
    end
    %print result
    fprintf('Result of CNN on test set with kfold cross validation: \n');
    fprintf('Average Accuracy for CNN: %f\n',mean(cnn_acc)*100);
    fprintf('Average Precision for CNN: %f\n',mean(cnnPre));
    fprintf('Average Recal for CNN: %f\n',mean(cnnRe));
    fprintf('Average FScore for CNN: %f\n',mean(cnnFS));
    fprintf('Average FP for CNN: %f\n',mean(cnnFB));
    fprintf('Average AUC for CNN: %f\n',mean(cnnAUC));
    

end

%% CNN model
function[cnn_acc,cnnPre,cnnRe,cnnFS,cnnFB,cnnAUC] = buildandTrainCNN(embeddingDimension,Y_train,Y_test,sequenceLength,numClasses,dataTransformedEval,dataTransformedTest,dataTransformedTrain,Kfold)
%     classNames = unique(labels);
    numObservations = numel(Y_train);
    
    numFeatures = embeddingDimension;
    inputSize = [1 sequenceLength numFeatures];
    numFilters = 200;

    ngramLengths = [2 3 4 5];
    numBlocks = numel(ngramLengths);

    
    layer = imageInputLayer(inputSize,'Normalization','none','Name','input');
    lgraph = layerGraph(layer);
    
    for j = 1:numBlocks
        N = ngramLengths(j);

        block = [
            convolution2dLayer([1 N],numFilters,'Name',"conv"+N,'Padding','same')
            batchNormalizationLayer('Name',"bn"+N)
            reluLayer('Name',"relu"+N)
            dropoutLayer(0.2,'Name',"drop"+N)
            maxPooling2dLayer([1 sequenceLength],'Name',"max"+N)];

        lgraph = addLayers(lgraph,block);
        lgraph = connectLayers(lgraph,'input',"conv"+N);
    end
    layers = [
    depthConcatenationLayer(numBlocks,'Name','depth')
    fullyConnectedLayer(numClasses,'Name','fc')
    softmaxLayer('Name','soft')
    classificationLayer('Name','classification')];

    lgraph = addLayers(lgraph,layers);
    
    for j = 1:numBlocks
        N = ngramLengths(j);
        lgraph = connectLayers(lgraph,"max"+N,"depth/in"+j);
    end
    miniBatchSize = 128;
    numObservations = numel(Y_train);
    numIterationsPerEpoch = floor(numObservations/miniBatchSize);
    if (Kfold)
        options = trainingOptions('adam', ...
        'MaxEpochs',10, ...
        'InitialLearnRate',0.001, ...
        'MiniBatchSize',miniBatchSize, ...
        'Shuffle','never', ...
        'ValidationData',dataTransformedEval, ...
        'ValidationFrequency',numIterationsPerEpoch, ...
        'Verbose',false, ...
        'OutputFcn',@(info)stopIfAccuracyNotImproving(info,3));
        cnnModel = trainNetwork(dataTransformedTrain,lgraph,options);
    else
        options = trainingOptions('adam', ...
        'MaxEpochs',10, ...
        'InitialLearnRate',0.001, ...
        'MiniBatchSize',miniBatchSize, ...
        'Shuffle','never', ...
        'ValidationData',dataTransformedEval, ...
        'ValidationFrequency',numIterationsPerEpoch, ...
        'Plots','training-progress', ...
        'Verbose',false, ...
        'OutputFcn',@(info)stopIfAccuracyNotImproving(info,3));
        cnnModel = trainNetwork(dataTransformedTrain,lgraph,options);
    end
    

%     
    %save model
    saveModel(cnnModel);
    
    %testing
    fprintf('Result of CNN on test set: \n');
    Y_pred_ls = classify(cnnModel,dataTransformedTest);
    cnn_acc = model_Acc(categorical(Y_test),Y_pred_ls);
    fprintf('Accuracy for CNN: %f\n',cnn_acc*100);

    [cnnPre, cnnRe, cnnFS,cnnFB,cnnAUC] = model_FScore(categorical(Y_test),Y_pred_ls);
    fprintf('Precision for CNN: %f\n',cnnPre);
    fprintf('Recal for CNN: %f\n',cnnRe);
    fprintf('FScore for CNN: %f\n',cnnFS);
    fprintf('FP for CNN: %f\n',cnnFB);
    fprintf('AUC for CNN: %f\n',cnnAUC);
end
%% save model
function saveModel(cnnModel)
    save('save_models/cnnModel','cnnModel')
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