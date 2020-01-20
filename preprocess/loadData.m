function [dataset] = loadData(csvPath)
    
    % Load csv file
    dataset = readtable(csvPath);  
    header={'label','text'};
    dataset.Properties.VariableNames{8} = 'text';
    dataset.Properties.VariableNames{9} = 'label';
%     label = dataset(:,1);
%     text = dataset(:,2);
    
end