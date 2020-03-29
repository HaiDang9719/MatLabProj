if (~cfg.dataset.kfoldvalidation)
    % Create a bag-of-words model from the tokenized documents
    bagWord = bagOfWords(clean_text_train);

    % Remove words from the bag-of-words model that do not appear more than two times in total. 
    % Remove any documents containing no words from the bag-of-words model, and remove the corresponding entries in labels.
    bagWord = removeInfrequentWords(bagWord,2);
    [bagWord,idx] = removeEmptyDocuments(bagWord);
    Y_train1 = Y_train;
    Y_train1(idx) = [];

    % Train supervised classification models (naive bayes, svm, random forest)
    % using bag-of-word
%     X_train = tfidf(bagWord,'Normalized',true,'TFWeight','log','IDFWeight','smooth');
%     X_test = tfidf(bagWord,clean_text_test,'Normalized',true,'TFWeight','log','IDFWeight','smooth');
    X_train = bagWord.Counts;
    X_test = encode(bagWord,clean_text_test);
    BuildandTrainNaivsBayes(X_train,Y_train1,X_test,Y_test,cfg)
else
    naiveBayes_acc=zeros(cfg.dataset.kfold,1); 
    naiveBayesPre=zeros(cfg.dataset.kfold,1); 
    naiveBayesRe=zeros(cfg.dataset.kfold,1); 
    naiveBayesFS=zeros(cfg.dataset.kfold,1); 
    naiveBayesFB=zeros(cfg.dataset.kfold,1); 
    naiveBayesAUC=zeros(cfg.dataset.kfold,1); 
    for i=1:cfg.dataset.kfold
        % Create a bag-of-words model from the tokenized documents
    bagWord = bagOfWords(clean_text_train{i});

    % Remove words from the bag-of-words model that do not appear more than two times in total. 
    % Remove any documents containing no words from the bag-of-words model, and remove the corresponding entries in labels.
    bagWord = removeInfrequentWords(bagWord,2);
    [bagWord,idx] = removeEmptyDocuments(bagWord);
    Y_train1 = Y_train{i};
    Y_train1(idx) = [];

    % Train supervised classification models (naive bayes, svm, random forest)
    % using bag-of-word
%     X_train = tfidf(bagWord,'Normalized',true,'TFWeight','log','IDFWeight','smooth');
%     X_test = tfidf(bagWord,clean_text_test{i},'Normalized',true,'TFWeight','log','IDFWeight','smooth');
    X_train = bagWord.Counts;
    X_test = encode(bagWord,clean_text_test{i});
    
    [naiveBayes_acc,naiveBayesPre, naiveBayesRe, naiveBayesFS,naiveBayesFB,naiveBayesAUC] = BuildandTrainNaivsBayes(X_train,Y_train1,X_test,Y_test{i},cfg);
        
    end
%     %save model
%     saveModel(lstmModel);
    
    %print result
    fprintf('Result of NAIVEBAYES on test set with kfold cross validation: \n');
    fprintf('Average Accuracy for NAIVEBAYES: %f\n',mean(naiveBayes_acc)*100);
    fprintf('Average Precision for NAIVEBAYES: %f\n',mean(naiveBayesPre));
    fprintf('Average Recal for NAIVEBAYES: %f\n',mean(naiveBayesRe));
    fprintf('Average FScore for NAIVEBAYES: %f\n',mean(naiveBayesFS));
    fprintf('Average FP for NAIVEBAYES: %f\n',mean(naiveBayesFB));
    fprintf('Average AUC for NAIVEBAYES: %f\n',mean(naiveBayesAUC));
end

%% naiveBays model
function[naiveBayes_acc,naiveBayesPre, naiveBayesRe, naiveBayesFS,naiveBayesFB,naiveBayesAUC] = BuildandTrainNaivsBayes(X_train,Y_train1,X_test,Y_test,cfg)
    
    t = templateNaiveBayes('DistributionNames','mvmn');
    predictors = X_train;
    response = Y_train1;
    naiveBayesModel = fitcecoc(predictors,response,'Learners',t,'FitPosterior',true,'Coding','onevsone','ResponseName','response');
    %testing
    fprintf('Result of SVM on test set: \n');
    Y_pred_ls = predict(naiveBayesModel,X_test);
    naiveBayes_acc = model_Acc(Y_test,Y_pred_ls);
    fprintf('Accuracy for %s: %f\n',upper(cfg.model.MLmodel),naiveBayes_acc*100);

    [naiveBayesPre, naiveBayesRe, naiveBayesFS,naiveBayesFB,naiveBayesAUC] = model_FScore(Y_test,Y_pred_ls);
    fprintf('Precision for %s: %f\n',upper(cfg.model.MLmodel),naiveBayesPre);
    fprintf('Recal for %s: %f\n',upper(cfg.model.MLmodel)',naiveBayesRe);
    fprintf('FScore for %s: %f\n',upper(cfg.model.MLmodel),naiveBayesFS);
    fprintf('FP for %s: %f\n',upper(cfg.model.MLmodel),naiveBayesFB);
    fprintf('AUC for %s: %f\n',upper(cfg.model.MLmodel),naiveBayesAUC);
end