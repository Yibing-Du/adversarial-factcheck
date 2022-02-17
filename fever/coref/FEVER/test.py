import random, os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.autograd import Variable
from transformers import BertTokenizer, BertForSequenceEncoder, RobertaTokenizer, RobertaForSequenceEncoder
from models import inference_model
from data_loader import DataLoader
from torch.nn import NLLLoss
import logging


from data_loader import DataLoaderTest
import logging
import json
import random, os
import argparse
import numpy as np
from models import inference_model

logger = logging.getLogger(__name__)



def eval_model(model, label_list, validset_reader, outdir, name):
    outpath = outdir + name
    with torch.no_grad():
        with open(outpath, "w") as f:
            for index, data in enumerate(validset_reader):
                inputs, ids = data
                logits = model(inputs)
                preds = logits.max(1)[1].tolist()
                assert len(preds) == len(ids)
                for step in range(len(preds)):
                    instance = {"id": ids[step], "predicted_label": label_list[preds[step]]}
                    f.write(json.dumps(instance) + "\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="bert", required=True)
    parser.add_argument('--test_path', help='train path')
    parser.add_argument('--name', help='train path')
    parser.add_argument('--test_origin_path', help='train path')
    parser.add_argument("--batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument('--outdir', required=True, help='path to output directory')
    parser.add_argument('--bert_pretrain', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout.')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    # parser.add_argument("--bert_hidden_dim", default=1024, type=int, help="Total batch size for training.")
    parser.add_argument("--layer", type=int, default=1, help='Graph Layer.')
    parser.add_argument("--num_labels", type=int, default=3)
    parser.add_argument("--kernel", type=int, default=21, help='Evidence num.')
    parser.add_argument("--evi_num", type=int, default=5, help='Evidence num.')
    parser.add_argument("--threshold", type=float, default=0.0, help='Evidence num.')
    parser.add_argument("--max_len", default=130, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")




    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.DEBUG,
                        datefmt='%d-%m-%Y %H:%M:%S')
    logger.info(args)
    logger.info('Start testing!')

    label_map = {'SUPPORTS': 0, 'REFUTES': 1, 'NOT ENOUGH INFO': 2}
    label_list = ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']
    args.num_labels = len(label_map)
    if args.model_type=='bert':
        tokenizer = BertTokenizer.from_pretrained(args.bert_pretrain, do_lower_case=False)
        bert_model = BertForSequenceEncoder.from_pretrained(args.bert_pretrain)
    elif args.model_type=='roberta':
        tokenizer = RobertaTokenizer.from_pretrained(args.bert_pretrain, do_lower_case=False)
        bert_model = RobertaForSequenceEncoder.from_pretrained(args.bert_pretrain)
    else:
        assert(False)
    args.bert_hidden_dim = bert_model.hidden_size

    logger.info("loading validation set")
    validset_reader = DataLoaderTest(args.test_path, label_map, tokenizer, args, batch_size=args.batch_size)
    logger.info('initializing estimator model')


    model = inference_model(bert_model, args)
    model.load_state_dict(torch.load(args.checkpoint)['model'])

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model = model.cuda()
    model.eval()
    eval_model(model, label_list, validset_reader, args.outdir, args.name)


