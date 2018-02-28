from sklearn.feature_extraction.text import CountVectorizer


train_set = ("The sky is blue.", "The sun is bright.")
test_set = ("The sun in the sky is bright.","We can see the shining sun, the bright sun.")


vectorizer = CountVectorizer(vocabulary=['sky', 'sun', 'bright'])
print vectorizer
vectorizer.fit(train_set)
vectorizer.fit_transform(train_set)
vectorizer.get_feature_names()
print vectorizer.vocabulary
smatrix = vectorizer.transform(test_set)
print smatrix
print smatrix.todense()

d4: We can see the shining sun, the bright sun.

Train Document Set:
d1: The sky is blue.
d2: The sun is bright.
Test Document Set:
d3: The sun in the sky is bright.
d4: We can see the shining sun, the bright sun.