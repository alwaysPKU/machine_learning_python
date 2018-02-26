import nltk


sent1 = 'The cat is walking in the bedroom.'
sent2 = 'A dog was running across the kitchen.'

# 对句子进行词汇分割和正则化，有些情况如aren't分割为are和n't；I'm分割成I和'm。
token_1 = nltk.word_tokenize(sent1)
print(token_1)
token_2 = nltk.word_tokenize(sent2)
# 整理两句的词表，按照ASCII的排序输出
vocab_1 = sorted(set(token_1))
vocab_2 = sorted(set(token_2))
# 初始化stemmer寻找哥哥词汇最原始的词根
stemmer = nltk.stem.PorterStemmer()
stem_1 = [stemmer.stem(t) for t in token_1]
print(stem_1)
stem_2 = [stemmer.stem(t) for t in token_2]
# 初始化词性标注器对每个词汇进行标注
pos_tag_1 = nltk.tag.pos_tag(token_1)
print(pos_tag_1)
pos_tag_2 = nltk.tag.pos_tag(token_2)

