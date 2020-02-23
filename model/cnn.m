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
    
    numFeatures = 100;
    inputSize = [1 16 numFeatures];
    numFilters = 200;

    ngramLengths = [2 3 4 5];
    numBlocks = numel(ngramLengths);

    numClasses = numel(classNames);
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

    options = trainingOptions('adam', ...
    'MaxEpochs',10, ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle','never', ...
    'ValidationData',[X_val,num2cell(Y_val)], ...
    'ValidationFrequency',numIterationsPerEpoch, ...
    'Plots','training-progress', ...
    'Verbose',false);
    net = trainNetwork([X_train,num2cell(Y_train)],lgraph,options);
    % training option for the model
% 
%     try
%         nnet.internal.cnngpu.reluForward(1);
%     catch ME
%     end
%     
%     %LSTM model
%     lstmModel = [ ...
%     sequenceInputLayer(inputSize)
%     wordEmbeddingLayer(embeddingDimension,numWords)
% 
%     lstmLayer(numHiddenUnits,'OutputMode','last')
%     fullyConnectedLayer(numClasses)
%     softmaxLayer
%     classificationLayer];
%     
%     %Option
%     options = trainingOptions('adam', ...
%         'ExecutionEnvironment','auto',...
%         'MaxEpochs',10, ...    
%         'GradientThreshold',1, ...
%         'InitialLearnRate',0.001, ...
%         'ValidationData',{X_val,categorical(Y_val)}, ...
%         'Plots','training-progress', ...
%         'Verbose',false, ...
%         'OutputFcn',@(info)stopIfAccuracyNotImproving(info,3));
%     
%     %training
%     lstmModel = trainNetwork(X_train,categorical(Y_train),lstmModel,options);
%     
%     %save model
%     saveModel(lstmModel);
%     
%     %testing
%     fprintf('Result of LSTM on test set: \n');
%     Y_pred_ls = classify(lstmModel,X_test);
%     lstm_acc = model_Acc(categorical(Y_test),Y_pred_ls);
%     fprintf('Accuracy for LSTM: %f\n',lstm_acc*100);
% 
%     [lsPre, lsRe, lsFS,lsFB,lsAUC] = model_FScore(categorical(Y_test),Y_pred_ls);
%     fprintf('Precision for LSTM: %f\n',lsPre);
%     fprintf('Recal for LSTM: %f\n',lsRe);
%     fprintf('FScore for LSTM: %f\n',lsFS);
%     fprintf('FP for LSTM: %f\n',lsFB);
%     fprintf('AUC for LSTM: %f\n',lsAUC);
end