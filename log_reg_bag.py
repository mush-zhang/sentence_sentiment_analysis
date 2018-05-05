import graphlab
import csv
import operator
import pandas as pd 
from itertools import izip


def procedure(train_data, test_data):

    sents = train_data

    sents['word_count'] = graphlab.text_analytics.count_words(sents['text'])

    test_data['word_count'] = graphlab.text_analytics.count_words(test_data['text'])

    docs = sents['word_count']
    sents['word_count'] = docs.dict_trim_by_keys(graphlab.text_analytics.stopwords(), exclude=True)

    test_docs = test_data['word_count']
    test_data['word_count'] = test_docs.dict_trim_by_keys(graphlab.text_analytics.stopwords(), exclude=True)

    sents.head()

    test_data.head()

    model = graphlab.logistic_classifier.create(sents, target='sentiment', features=['word_count'], convergence_threshold=0.01, validation_set=None)

    #predictions = model.classify(test_data)

    pred_class = model.predict(test_data, output_type = "class")
    
    return list(pred_class)

def record(agg, single):

    for idx in range(len(single)):
        agg[idx][int(single[idx])] += 1
    
    return agg

def aggregate_results(agg):
    final = [6] * len(agg)
    for idx in range(len(agg)):
        final[idx] = max(agg[idx].iteritems(), key=operator.itemgetter(1))[0]
    return final

def main():

    #train_data = graphlab.SFrame('clean_train2.csv')
    test_data = graphlab.SFrame('clean_test2.csv')

    # record results: list of dicts
    aggregate = [{0: 0, 1: 0, 2: 0, 3: 0, 4: 0}]*test_data.num_rows()
    print aggregate[0]

    for idx in range(100):
        
        data = pd.read_csv('clean_train2.csv', sep=',', header=0, engine='python')
        sample_data = data.sample(frac=1, replace=True)
        sample_data.to_csv('./temp_train/train.csv', sep=',', index=False)
        train_data = graphlab.SFrame('./temp_train/train.csv')
        result_t = procedure(train_data, test_data)
        record(aggregate, result_t)
    
    final_results = aggregate_results(aggregate)
    

    ids = list(range(1455))
    with open('bagg_pred.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(izip(ids, final_results))

if __name__ == "__main__": main()