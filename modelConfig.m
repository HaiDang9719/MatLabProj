function cfg = modelConfig()
    %% Dataset:
    % Dataset path:("datasets/dataset2Label.csv",
    % "datasets/AdHocAnnouncements.csv")
    cfg.dataset.path = "datasets/dataset2Label.csv";
    % Dataset average sequence length, 16 - dataset2Label.csv, 150 - AdHocAnnouncements.csv 
    cfg.dataset.sequenceLength = 16;
    % Number of class label
    cfg.dataset.numClass = 2;
    % Split ratio for train set and test set, ex: 80% train set, 20% test
    % set
    cfg.dataset.splitRatio = 0.2;
    % Config for kford validation
    cfg.dataset.kfoldvalidation = false;
    cfg.dataset.kfold = 10;
    
    
    % Model:
    %model (lstm, bilstm, cnn,svm,naiveBayes)
    cfg.model.MLmodel = "lstm";
    %pretrained wordembedding layer with fastTextWordEmbedding(only for
    %deep learning models)
    cfg.model.fastTextWordEmbedding = false;
    %save lstm path
    cfg.model.pretrainedLSTM = "pretrained/lstmModel";
    %save bilstm path
    cfg.model.pretrainedBiLSTM = "pretrained/bilstmModel";
    %save cnn path
    cfg.model.pretrainedCNN = "pretrained/cnnModel";
    
    % Execution mode:("train/test")
    cfg.execMode = "train";
    

    
end
