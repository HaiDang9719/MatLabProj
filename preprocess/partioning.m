function [train_set,eval_set,test_set] = partioning(data)
%PARTIONING Summary of this function goes here
%   Detailed explanation goes here

%divide dataset into train set(80%) and dev set(20%)
cvp = cvpartition(data.label, 'HoldOut',0.2);
train_set = data(training(cvp),:);
dev_set = data(test(cvp),:);

%then, divide dev set into two sets: evaluation set(50%) and test set(50%)
cvp = cvpartition(dev_set.label,'HoldOut',0.5);
eval_set = dev_set(training(cvp),:);
test_set = dev_set(test(cvp),:);
end

