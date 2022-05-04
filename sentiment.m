clear
emb = fastTextWordEmbedding;

data = readLexicon;

idx = ~isVocabularyWord(emb,data.Word);
data(idx,:) = [];

numWords = size(data,1);
cvp = cvpartition(numWords,'HoldOut',0.1);
dataTrain = data(training(cvp),:);
dataTest = data(test(cvp),:);

wordsTrain = dataTrain.Word;
XTrain = word2vec(emb,wordsTrain);
YTrain = dataTrain.Label;

mdl = fitcsvm(XTrain,YTrain);

wordsTest = dataTest.Word;
XTest = word2vec(emb,wordsTest);
YTest = dataTest.Label;

[YPred,scores] = predict(mdl,XTest);

filename = "Query 1.csv";
tbl = unique(readtable(filename,'TextType','string'));
textData = tbl.TweetText;
textTime = tbl.Time;
documents = preprocessText(textData);

%change the twitter datetime format into something matlab can understand
textDateTime = datetime(textTime,'TimeZone','local','InputFormat','yyyy-MM-dd''T''HH:mm:ss.SSSZ');
disp(textDateTime);
disp(textDateTime(3));


idx = ~isVocabularyWord(emb,documents.Vocabulary);
documents = removeWords(documents,idx);

words = documents.Vocabulary;
words(ismember(words,wordsTrain)) = [];

vec = word2vec(emb,words);
[YPred,scores] = predict(mdl,vec);

figure
subplot(1,2,1)
idx = YPred == "Positive";
wordcloud(words(idx),scores(idx,1));
title("Predicted Positive Sentiment")

subplot(1,2,2)
wordcloud(words(~idx),scores(~idx,2));
title("Predicted Negative Sentiment")

for i = 1:numel(documents)
    words = string(documents(i));
    vec = word2vec(emb,words);
    [~,scores] = predict(mdl,vec);
    sentimentScore(i) = mean(scores(:,1));
end

figure(2)
s = scatter(textDateTime,sentimentScore');
m = s.Marker;
s.Marker = '*';
figure(3)

disp(table(sentimentScore', textData));
t = table(sentimentScore', textData);
vars = {'Sentiment Score', 'TextData'};
fig = uifigure;
uitable(fig,'Data',t);