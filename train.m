cfg = modelConfig();
if cfg.execMode == "train"
    fprintf('Train %s ...: \n',upper(cfg.model.MLmodel));
    if(cfg.model.MLmodel == "lstm")
        lstm()
    elseif(cfg.model.MLmodel == "bilstm")
        bilstm()
    elseif(cfg.model.MLmodel == "cnn")
        cnn()
    elseif(cfg.model.MLmodel == "svm")
        svm()
    elseif(cfg.model.MLmodel == "naiveBayes")
        naiveBayes()
    else
        fprintf('Wrong model configuration. Please check model name in configuration again. \n');
    end
 elseif cfg.execMode == "test"
     if(cfg.model.MLmodel == "lstm")
        %Load pretrained model
        model = load(cfg.model.pretrainedLSTM,'lstmModel');
        %predict
        predict(model.lstmModel,cfg,clean_text_test,clean_text_train,Y_test)
    elseif(cfg.model.MLmodel == "bilstm")
        %Load pretrained model
        model = load(cfg.model.pretrainedBiLSTM,'bilstmModel');
        %predict
        predict(model.bilstmModel,cfg,clean_text_test,clean_text_train,Y_test)
    elseif(cfg.model.MLmodel == "cnn")
        %Load pretrained model
        model = load(cfg.model.pretrainedCNN,'cnnModel');
        %predict
        predictCNN(model.cnnModel,cfg,clean_text_test,clean_text_train,Y_test)
    else
        fprintf('This model does not have pretrained version. \n');
     end       
 end
 
 %Print result
 function predict(model,cfg,clean_text_test,clean_text_train,Y_test)
 
    fprintf('Testing  %s ...: \n',upper(cfg.model.MLmodel));
    % create dictionary
    if (cfg.model.fastTextWordEmbedding)
        enc = fastTextWordEmbedding;
    else
        enc = wordEncoding(clean_text_train);
    end
    X_test = doc2sequence(enc,clean_text_test,'Length',cfg.dataset.sequenceLength);
    fprintf('Result of %s on test set: \n',upper(cfg.model.MLmodel));
    Y_pred_ls = classify(model,X_test);
    lstm_acc = model_Acc(categorical(Y_test),Y_pred_ls);
    fprintf('Accuracy for %s: %f\n',upper(cfg.model.MLmodel),lstm_acc*100);

    [lsPre, lsRe, lsFS,lsFB,lsAUC] = model_FScore(categorical(Y_test),Y_pred_ls);
    fprintf('Precision for %s: %f\n',upper(cfg.model.MLmodel),lsPre);
    fprintf('Recal for %s: %f\n',upper(cfg.model.MLmodel),lsRe);
    fprintf('FScore for %s: %f\n',upper(cfg.model.MLmodel),lsFS);
    fprintf('FP for %s: %f\n',upper(cfg.model.MLmodel),lsFB);
    fprintf('AUC for %s: %f\n',upper(cfg.model.MLmodel),lsAUC);
    
 end
 
 %Print result
 function predictCNN(model,cfg,clean_text_test,clean_text_train,Y_test)
 
    fprintf('Testing  %s ...: \n',upper(cfg.model.MLmodel));
    % create dictionary
    if (cfg.model.fastTextWordEmbedding)
        enc = fastTextWordEmbedding;
    else
        enc = trainWordEmbedding(clean_text_train);
 
    end
    X_test = doc2sequence(enc,clean_text_test,'Length',cfg.dataset.sequenceLength);
    fprintf('Result of %s on test set: \n',upper(cfg.model.MLmodel));
    
    predictorsTest = cellfun(@(X) permute(X,[3 2 1]),X_test,'UniformOutput',false);
    responsesTest = categorical(Y_test,unique(Y_test));
    dataTransformedTest = table(predictorsTest,responsesTest);
    Y_pred_ls = classify(model,dataTransformedTest);
    
    lstm_acc = model_Acc(categorical(Y_test),Y_pred_ls);
    fprintf('Accuracy for %s: %f\n',upper(cfg.model.MLmodel),lstm_acc*100);

    [lsPre, lsRe, lsFS,lsFB,lsAUC] = model_FScore(categorical(Y_test),Y_pred_ls);
    fprintf('Precision for %s: %f\n',upper(cfg.model.MLmodel),lsPre);
    fprintf('Recal for %s: %f\n',upper(cfg.model.MLmodel),lsRe);
    fprintf('FScore for %s: %f\n',upper(cfg.model.MLmodel),lsFS);
    fprintf('FP for %s: %f\n',upper(cfg.model.MLmodel),lsFB);
    fprintf('AUC for %s: %f\n',upper(cfg.model.MLmodel),lsAUC);
    
 end