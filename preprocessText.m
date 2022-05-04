function documents = preprocessText(textData)
%remove duplicate entries (usually a result of retweets
%documents = unique(textData,'rows');
% Tokenize the text.
documents = tokenizedDocument(textData);

% Erase punctuation.
documents = erasePunctuation(documents);

% Remove a list of stop words.
documents = removeStopWords(documents);

% Convert to lowercase.
documents = lower(documents);

end