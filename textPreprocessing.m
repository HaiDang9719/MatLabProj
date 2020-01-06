% T = readtable('./data/dgap_adhoc_2010.csv',"Format","%s%s%s%s%s%s");
% headline = T.("headline");
% fprintf('%s',headline(1,1));
% documents = tokenizedDocument(headline(1));
% function [XTrain, YTrain] = prepareTrainData(trainset, k)
%     %% Fold the original dataset into chunks of size lag
%     chunkCount = ceil(size(trainset, 1) / k);
%     XTrain = {};%zeros(chunkCount, size(trainset, 2), k);
%     YTrain = zeros(chunkCount, size(trainset, 2));
%     for i = 1:(size(trainset, 1) - k)
%         tmpX = trainset(i:(i + k - 1), :);
%         tmpY = trainset((i + k), :);
%         XTrain{i} = tmpX.';
%         YTrain(i, :) = tmpY;
%     end
%     XTrain = XTrain.';
% end

function [label,text] = textPreprocessing()
    [label, text] = loadDataset('./data/train.csv');
    tok_doc = tokenizationFunc(text);
    rsw_doc = removeStopWordFunc(tok_doc);
    rp_doc = removePunctFunc(rsw_doc);
    text = normalizaTextFunc(rp_doc);
end

function [label, text] = loadDataset(csvPath)
    
    % Load csv file
    dataset = readtable(csvPath, "Delimiter",",",'TextType','String');   
    label = dataset(:,1);
    text = dataset(:,2);
    
end

%tokenization string text
function [tok_doc] = tokenizationFunc(textData)

    strArr = table2array(textData);
    strArr = removeSpecialCharFunc(strArr);
    tok_doc = tokenizedDocument(strArr);  
    
end

%remove stop words from text
function [cl_tok_doc] = removeStopWordFunc(doc)

    cl_tok_doc = removeStopWords(doc);
    
end

%remove punctuation fromtext
function [cl_tok_doc] = removePunctFunc(doc)

    cl_tok_doc = erasePunctuation(doc);
    
end

%remove special character
function [strArr] = removeSpecialCharFunc(doc)

    strArr = replace(doc,"\n"," ");
    
end

%normalize text, convert to base form of words
function [cl_doc] = normalizaTextFunc(doc)
    
    cl_doc = normalizeWords(doc,'Style','lemma');
    
end