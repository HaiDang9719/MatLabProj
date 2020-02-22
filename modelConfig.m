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
    %model
    cfg.model.MLmodel = "lstm";
    %pretrained wordembedding layer with fastTextWordEmbedding
    cfg.model.fastTextWordEmbedding = false;
    
    

    
end
