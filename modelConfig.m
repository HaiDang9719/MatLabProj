function cfg = modelConfig()
    %% Dataset:
    % Dataset path:
    cfg.dataset.path = "dataset2LabelARFOUR.csv";
    % Dataset sequence length, 16 - dataset2Label.csv, 150 - AdHocAnnoucements.csv 
    cfg.dataset.sequenceLength = 16;
    % Split ratio for train set and test set, ex: 80% train set, 20% test
    % set
    cfg.dataset.splitRatio = 0.2;
    %kford
    cfg.dataset.kfoldvalidation = false;
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
    cfg.model.pretrainedCNN = "pretrained/cnnModel";
    
    % Execution mode:("train/test")
    cfg.execMode = "test";
    

    
end
