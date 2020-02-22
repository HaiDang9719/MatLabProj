function [dataset] = loadData(csvPath)
    
    if csvPath=="dataset2Lable.csv"
        
        % Load csv file
        dataset = readtable(csvPath);  
        header={'label','text'};
        dataset.Properties.VariableNames{8} = 'text';
        dataset.Properties.VariableNames{9} = 'label';
        
    elseif csvPath=="AdHocAnnouncements.csv"
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