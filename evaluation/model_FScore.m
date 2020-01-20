function [biPre, biRe, biFS] = model_FScore(Y_test,Y_pred)
%MODEL_FSCORE Summary of this function goes here
%   Detailed explanation goes here
conTable = confusionmat(Y_test,Y_pred);
biPre =  conTable(2,2)/(conTable(2,2)+conTable(1,2));
biRe =  conTable(2,2)/(conTable(2,2)+conTable(2,1));
biFS = 2*biPre*biRe/(biPre+biRe);
end

