from sklearn.feature_extraction import DictVectorizer

measurements = [{'city': 'Dubai', 'temperature': 33.}, {'city': 'London', 'temperature': 12.},
                {'city': 'San Fransisco', 'temperature': 18.}]

# 初始化特征抽取器
vec = DictVectorizer()
vec_feature = vec.fit_transform(measurements)
print(vec_feature)
print(type(vec_feature))
print(vec_feature.toarray())
print(vec.feature_names_)
