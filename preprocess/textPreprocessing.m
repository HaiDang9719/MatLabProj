function [clean_textset] = textPreprocessing(dataset)

    clean_textset = tokenizationFunc(dataset);
    clean_textset = removeStopWordFunc(clean_textset);
    clean_textset = removePunctFunc(clean_textset);
    clean_textset = normalizaTextFunc(clean_textset);
    
end


%tokenization string text
function [tok_doc] = tokenizationFunc(textData)

%     strArr = table2array(textData);
    strArr = removeSpecialCharFunc(textData);
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