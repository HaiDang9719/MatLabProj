% Load csv file
dataset = readtable("dataset2LabelARFOUR.csv");  
header={'label','text'};
dataset.Properties.VariableNames{8} = 'text';
dataset.Properties.VariableNames{9} = 'label';

car = readtable("Events_CAR.xlsx");





% Calculate frequency of labels
f = tabulate(dataset.label);