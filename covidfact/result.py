from fairseq.models.roberta import RobertaModel
import os
import numpy as np
from sklearn.metrics import classification_report

roberta = RobertaModel.from_pretrained(
    './covidfact-roberta/',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='../RTE-covidfact-bin'
)

label_fn = lambda label: roberta.task.label_dictionary.string(
    [label + roberta.task.label_dictionary.nspecial]
)
roberta.cuda()
roberta.eval()
m = {'not_entailment': 0, 'entailment' : 1}

files = ['data/test1_baseline.tsv', 'data/test1_advadd.tsv']
for file in files:
    f = open(file.replace("data/test1_", "").replace(".tsv", "") + '_predictions.txt','w')
    ncorrect, nsamples = 0, 0
    pred = []
    gold = []
    with open(file) as fin:
        fin.readline()
        for index, line in enumerate(fin):
            tokens = line.strip().split('\t')
            sent1, sent2 = tokens[1], tokens[2]
            target = "not_entailment"
            tokens = roberta.encode(sent1, sent2)
            if len(tokens) > 512:
                print("Truncate!")
                j = 0
                while len(tokens) > 512:
                    evis = sent1.split(". ")
                    evi = evis.index(max(evis, key=lambda x: len(x.split())))
                    sent1 = ". ".join([" ".join(item.split()[:-j]) if i == evi else item for i, item in enumerate(sent1.split(". "))])
                    tokens = roberta.encode(sent1, sent2)
                    j += 1
            prediction = roberta.predict('sentence_classification_head', tokens).argmax().item()
            prediction_label = label_fn(prediction)
            f.write(prediction_label+'\t'+target+'\n')
            pred.append(m[prediction_label])
            gold.append(m[target])
            ncorrect += int(prediction_label == target)
            nsamples += 1
    print('| Accuracy: ', float(ncorrect)/float(nsamples))

