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
    X_train = bagWord.Counts;
    X_test = encode(bagWord,clean_text_test);
    BuildandTrainSVM(X_train,Y_train1,X_test,Y_test)
else
    svm_acc=zeros(cfg.dataset.kfold,1); 
    svmPre=zeros(cfg.dataset.kfold,1); 
    svmRe=zeros(cfg.dataset.kfold,1); 
    svmFS=zeros(cfg.dataset.kfold,1); 
    svmFB=zeros(cfg.dataset.kfold,1); 
    svmAUC=zeros(cfg.dataset.kfold,1); 
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
    X_train = bagWord.Counts;
    X_test = encode(bagWord,clean_text_test{i});
    
    [svm_acc,svmPre, svmRe, svmFS,svmFB,svmAUC] = BuildandTrainSVM(X_train,Y_train1,X_test,Y_test{i});
        
    end
%     %save model
%     saveModel(lstmModel);
    
    %print result
    fprintf('Result of SVM on test set with kfold cross validation: \n');
    fprintf('Average Accuracy for SVM: %f\n',mean(svm_acc)*100);
    fprintf('Average Precision for SVM: %f\n',mean(svmPre));
    fprintf('Average Recal for SVM: %f\n',mean(svmRe));
    fprintf('Average FScore for SVM: %f\n',mean(svmFS));
    fprintf('Average FP for SVM: %f\n',mean(svmFB));
    fprintf('Average AUC for SVM: %f\n',mean(svmAUC));
end

%% SVM model
function[svm_acc,svmPre, svmRe, svmFS,svmFB,svmAUC] = BuildandTrainSVM(X_train,Y_train1,X_test,Y_test)
    
    svmModel = fitcecoc(X_train,Y_train1,'Learners','linear');
    %testing
    fprintf('Result of SVM on test set: \n');
    Y_pred_ls = predict(svmModel,X_test);
    svm_acc = model_Acc(Y_test,Y_pred_ls);
    fprintf('Accuracy for SVM: %f\n',svm_acc*100);

    [svmPre, svmRe, svmFS,svmFB,svmAUC] = model_FScore(Y_test,Y_pred_ls);
    fprintf('Precision for SVM: %f\n',svmPre);
    fprintf('Recal for SVM: %f\n',svmRe);
    fprintf('FScore for SVM: %f\n',svmFS);
    fprintf('FP for SVM: %f\n',svmFB);
    fprintf('AUC for SVM: %f\n',svmAUC);
end