import graphlab

sents = graphlab.SFrame(sys.argv[1])
test_data = graphlab.SFrame(sys.argv[2])

sents['word_count'] = graphlab.text_analytics.count_words(sents['text'])

test_data['word_count'] = graphlab.text_analytics.count_words(test_data['text'])

docs = sents['word_count']
sents['word_count'] = docs.dict_trim_by_keys(graphlab.text_analytics.stopwords(), exclude=True)

test_docs = test_data['word_count']
test_data['word_count'] = test_docs.dict_trim_by_keys(graphlab.text_analytics.stopwords(), exclude=True)

sents.head()

test_data.head()

model = graphlab.logistic_classifier.create(sents, target='sentiment', features=['word_count'], validation_set=None)

predictions = model.classify(test_data)

pred_class = model.predict(test_data, output_type = "class")
#pred_class
ids = list(range(1455))
#len(list(pred_class))

import csv
import sys
from itertools import izip
"""
argv[0]: program name
argv[1]: train file
argv[2]: test file
argv[3]: result file
"""

with open(sys.argv[3], 'wb') as f:
    writer = csv.writer(f)
    writer.writerows(izip(ids, list(pred_class)))

