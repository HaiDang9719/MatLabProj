function [dataset] = loadData(csvPath)
    
    % Load csv file
    dataset = readtable(csvPath, "Delimiter",",",'TextType','String');  
    header={'label','text'};
    dataset.Properties.VariableNames = header;
%     label = dataset(:,1);
%     text = dataset(:,2);
    
end