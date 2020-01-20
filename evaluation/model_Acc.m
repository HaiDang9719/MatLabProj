function [accuracy] = model_Acc(Y_test,Y_pred)
%MODEL_ACC Summary of this function goes here
%   Calculate the accuracy on test set of the model, note: bote input value
%   should be in categrical form
accuracy = sum(Y_pred == Y_test)/numel(Y_pred);
end

