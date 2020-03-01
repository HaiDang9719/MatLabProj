function cfg = modelConfig()
    %% Dataset:
    % Dataset path:
    cfg.dataset.path = "dataset2Lable.csv";
    % Split ratio for train set and test set, ex: 80% train set, 20% test
    % set
    cfg.dataset.splitRatio = 0.2;
    %kford
    cfg.dataset.kfoldvalidation = true;
    cfg.dataset.kfold = 10;
    
    
    % Model:
    %model (lstm, bilstm, cnn)
    cfg.model.MLmodel = "cnn";
    %pretrained wordembedding layer with fastTextWordEmbedding
    cfg.model.fastTextWordEmbedding = false;
    %save lstm path
    cfg.model.pretrainedLSTM = "save_models/lstmModel";
    %save bilstm path
    cfg.model.pretrainedBiLSTM = "save_models/bilstmModel";
    %save cnn path
    cfg.model.pretrainedCNN = "save_models/cnnModel";
    
    % Execution mode:("train/test")
    cfg.execMode = "train";
    

    
end
