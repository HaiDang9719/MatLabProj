function [dataset] = loadData(csvPath)
    
    if csvPath=="datasets/dataset2Lable.csv" || csvPath=="datasets/dataset2LabelARFOUR.csv"
        
        % Load csv file
        dataset = readtable(csvPath);  
        header={'label','text'};
        dataset.Properties.VariableNames{8} = 'text';
        dataset.Properties.VariableNames{9} = 'label';
        
    elseif csvPath=="datasets/AdHocAnnouncements.csv"
        % Load csv file
        dataset = readtable(csvPath);  
        header={'label','text'};
        dataset.Properties.VariableNames{2} = 'text';
        dataset.Properties.VariableNames{6} = 'label';
    end
%     dataset.Properties.VariableNames{2} = 'text';
%     dataset.Properties.VariableNames{6} = 'label';
%     label = dataset(:,1);
%     text = dataset(:,2);
    
end