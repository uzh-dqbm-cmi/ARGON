import json
import re,os
from collections import Counter
from transformers import BertTokenizer
import numpy as np
class Tokenizer(object):
    def __init__(self, args):
        self.args = args
        self.ann_path = args.ann_path_tokenizer
        self.threshold = args.threshold
        self.folds = args.folds
        self.dataset_name = args.dataset_name
        # if self.dataset_name == 'iu_xray':
        #     self.clean_report = self.clean_report_iu_xray
        # else:
        self.clean_report = self.clean_report_mimic_cxr
        self.ann = json.loads(open(self.ann_path, 'r').read())

        if self.folds:
            self.train = [x for x in self.ann if str(x['fold']) in self.folds.split(",")]
        else:
            self.train = self.ann['train']

        self.token2idx, self.idx2token = self.create_vocabulary()



    def do_filter(self, tokens):
        NUMBER_TOKEN = '<NUM>'
        ALPHANUM_PATTERN = re.compile('^[a-zA-Z\\\.]+$')
        NUM_PATTERN = re.compile('^[0-9]*$')
        new_tokens = []
        for token in tokens:
            if token != '.':
                if ALPHANUM_PATTERN.search(token) is not None:
                    new_tokens.append(token)
                elif NUM_PATTERN.search(token) is not None:
                    new_tokens.append(NUMBER_TOKEN)
        return new_tokens

    def create_vocabulary(self):

        total_tokens = []

        for example in self.train:
            tokens = self.clean_report(example[self.args.report_mode]).split()
            for token in tokens:
                total_tokens.append(token)

        counter = Counter(total_tokens)
        vocab = [k for k, v in counter.items() if v >= self.threshold] + ['<unk>']
        vocab.sort()
        token2idx, idx2token = {}, {}
        for idx, token in enumerate(vocab):
            token2idx[token] = idx + 1
            idx2token[idx + 1] = token
        return token2idx, idx2token

    def clean_report_iu_xray(self, report):
        report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                        replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(" ".join(self.do_filter(sent.split()))) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    def clean_report_mimic_cxr(self, report):
        report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
            .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
            .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
            .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
            .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                        .replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(" ".join(self.do_filter(sent.split()))) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    def get_token_by_id(self, id):
        return self.idx2token[id]

    def get_id_by_token(self, token):
        if token not in self.token2idx:
            return self.token2idx['<unk>']
        return self.token2idx[token]

    def get_vocab_size(self):
        return len(self.token2idx)

    def __call__(self, report):
        tokens = self.clean_report(report).split()
        ids = []
        for token in tokens:
            ids.append(self.get_id_by_token(token))
        ids = [0] + ids + [0]
        return ids

    def decode(self, ids):
        txt = ''
        for i, idx in enumerate(ids):
            if idx > 0:
                if i >= 1:
                    txt += ' '
                txt += self.idx2token[idx]
            else:
                break
        return txt

    def decode_batch(self, ids_batch):
        out = []
        for ids in ids_batch:
            out.append(self.decode(ids))
        return out

class Tokenizer_t2t(object):
    def __init__(self, args):
        self.report_mode= args.report_mode
        self.ann_path = args.ann_path
        self.threshold = args.threshold
        self.dataset_name = args.dataset_name
        if self.dataset_name == 'iu_xray':
            self.clean_report = self.clean_report_iu_xray
        else:
            self.clean_report = self.clean_report_mimic_cxr
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.src_token2idx, self.src_idx2token = self.create_vocabulary(key_for_report= args.report_mode)
        self.tgt_token2idx, self.tgt_idx2token = self.create_vocabulary(key_for_report='report')

    def create_vocabulary(self, key_for_report):
        total_tokens = []
        for example in self.ann['train']:
            example_report = example[key_for_report]

            tokens = self.clean_report(example_report).split()
            for token in tokens:
                total_tokens.append(token)

        counter = Counter(total_tokens)
        vocab = [k for k, v in counter.items() if v >= self.threshold] + ['<unk>']
        vocab.sort()
        token2idx, idx2token = {}, {}
        for idx, token in enumerate(vocab):
            token2idx[token] = idx + 1
            idx2token[idx + 1] = token
        return token2idx, idx2token

    def clean_report_iu_xray(self, report):
        report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                        replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    def clean_report_mimic_cxr(self, report):
        report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
            .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
            .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
            .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
            .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                        .replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    def get_tgt_token_by_id(self, id):
        return self.tgt_idx2token[id]

    def get_tgt_id_by_token(self, token):
        if token not in self.tgt_token2idx:
            return self.tgt_token2idx['<unk>']
        return self.tgt_token2idx[token]


    def get_src_token_by_id(self, id):
        return self.src_idx2token[id]

    def get_src_id_by_token(self, token):
        if token not in self.src_token2idx:
            return self.src_token2idx['<unk>']
        return self.src_token2idx[token]

    def get_tgt_vocab_size(self):
        return len(self.tgt_token2idx)

    def get_src_vocab_size(self):
        return len(self.src_token2idx)

    def __call__(self, src, tgt):
        src_tokens = self.clean_report(src).split()
        src_ids = []
        for token in src_tokens:
            src_ids.append(self.get_src_id_by_token(token))
        src_ids = [0] + src_ids + [0]

        tgt_tokens = self.clean_report(tgt).split()
        tgt_ids = []
        for token in tgt_tokens:
            tgt_ids.append(self.get_tgt_id_by_token(token))
        tgt_ids = [0] + tgt_ids + [0]
        return src_ids, tgt_ids

    def decode(self, ids):
        txt = ''
        for i, idx in enumerate(ids):
            if idx > 0:
                if i >= 1:
                    txt += ' '
                txt += self.tgt_idx2token[idx]
            else:
                break
        return txt

    def decode_batch(self, ids_batch):
        out = []
        for ids in ids_batch:
            out.append(self.decode(ids))
        return out

class Bert_Tokenizer(object):
    def __init__(self, args):
        self.args = args
        self.ann_path = args.ann_path_tokenizer
        self.threshold = args.threshold
        self.folds = args.folds
        self.dataset_name = args.dataset_name

        if self.dataset_name == 'iu_xray':
            self.clean_report = self.clean_report_iu_xray
        else:
             self.clean_report = self.clean_report_mimic_cxr

        model_name_or_path = os.path.join(f"/cluster/home/fnooralahzad/models/Bio_ClinicalBERT")
        if os.path.exists(model_name_or_path):
            self.tokenizer  = BertTokenizer.from_pretrained(model_name_or_path)
        else:
            self.tokenizer  = BertTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")



    def clean_report_iu_xray(self, report):
        report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                        replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    def clean_report_mimic_cxr(self, report):
        report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
            .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
            .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
            .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
            .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                        .replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    def get_vocab_size(self):
        return self.tokenizer.vocab_size

    def __call__(self, report):
        clean_report = self.clean_report(report)
        ids = self.tokenizer.encode(clean_report, add_special_tokens=False)
        ids = [self.tokenizer.cls_token_id] + ids + [0]
        return ids

    def decode(self, ids):
        txt = self.tokenizer.decode(ids, skip_special_tokens=True)
        return txt

    def decode_batch(self, ids_batch):
        out = []
        for ids in ids_batch:
            out.append(self.decode(ids))
        return out

class Tokenizer_w2vEmb(object):
    def __init__(self, args):
        self.args = args
        self.ann_path = args.ann_path_tokenizer
        self.threshold = args.threshold
        self.folds = args.folds
        self.dataset_name = args.dataset_name
        self.clean_report = self.clean_report_mimic_cxr

        model_name_or_path = os.path.join(f"/cluster/home/fnooralahzad/models/w2v/glove")
        if os.path.exists(model_name_or_path):
            self.token2idx = np.load(os.path.join(model_name_or_path, 'word2ind.npy'), allow_pickle=True).item()
            self.idx2token = np.load(os.path.join(model_name_or_path, 'ind2word.npy'), allow_pickle=True).item()
        else:
           print("you should have w2v folder")


    def clean_report_iu_xray(self, report):
        report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                        replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(" ".join(self.do_filter(sent.split()))) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    def clean_report_mimic_cxr(self, report):
        report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
            .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
            .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
            .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
            .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                        .replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(" ".join(self.do_filter(sent.split()))) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report


    def get_token_by_id(self, id):
        return self.idx2token[id]

    def get_id_by_token(self, token):
        if token not in self.token2idx:
            return self.token2idx['<unk>']
        return self.token2idx[token]

    def get_vocab_size(self):
        return len(self.token2idx)

    def __call__(self, report):
        tokens = self.clean_report(report).split()
        ids = []
        for token in tokens:
            ids.append(self.get_id_by_token(token))
        ids = [0] + ids + [0]
        return ids

    def decode(self, ids):
        txt = ''
        for i, idx in enumerate(ids):
            if idx > 0:
                if i >= 1:
                    txt += ' '
                txt += self.idx2token[idx]
            else:
                break
        return txt

    def decode_batch(self, ids_batch):
        out = []
        for ids in ids_batch:
            out.append(self.decode(ids))
        return out

    def do_filter(self, tokens):
        NUMBER_TOKEN = '<NUM>'
        ALPHANUM_PATTERN = re.compile('^[a-zA-Z\\\.]+$')
        NUM_PATTERN = re.compile('^[0-9]*$')
        new_tokens = []
        for token in tokens:
            if token != '.':
                if ALPHANUM_PATTERN.search(token) is not None:
                    new_tokens.append(token)
                elif NUM_PATTERN.search(token) is not None:
                    new_tokens.append(NUMBER_TOKEN)
        return new_tokens

class SentenceTokenizer(object):
    def __init__(self, args):
        self.args = args
        self.ann_path = args.ann_path_tokenizer
        self.threshold = args.threshold
        self.folds = args.folds
        self.dataset_name = args.dataset_name
        self.clean_report = self.clean_report_mimic_cxr
        self.ann = json.loads(open(self.ann_path, 'r').read())

        if self.folds:
            self.train = [x for x in self.ann if str(x['fold']) in self.folds.split(",")]
        else:
            self.train = [ex for ex in self.ann['train'] ] + [ex for ex in self.ann['val']] + [ex for ex in self.ann['test'] ]

        self.token2idx, self.idx2token = self.create_vocabulary()

    def do_filter(self, tokens):
        NUMBER_TOKEN = '__NUM__'
        ALPHANUM_PATTERN = re.compile('^[a-zA-Z\\\.]+$')
        NUM_PATTERN = re.compile('^[0-9]*$')
        new_tokens = []
        for token in tokens:
            if token != '.':
                if ALPHANUM_PATTERN.search(token) is not None:
                    new_tokens.append(token)
                elif NUM_PATTERN.search(token) is not None:
                    new_tokens.append(NUMBER_TOKEN)
        return new_tokens

    def create_vocabulary(self):

        total_tokens = []

        for example in self.train:
            tokens = self.clean_report(example[self.args.report_mode]).split(".")
            for token in tokens:
                total_tokens.append(token+".")

        counter = Counter(total_tokens)
        vocab = [k for k, v in counter.items() if v >= self.threshold] + ['<unk>']
        vocab.sort()
        token2idx, idx2token = {}, {}
        for idx, token in enumerate(vocab):
            token2idx[token] = idx + 1
            idx2token[idx + 1] = token
        return token2idx, idx2token

    def clean_report_iu_xray(self, report):
        report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                        replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(" ".join(self.do_filter(sent.split()))) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    def clean_report_mimic_cxr(self, report):
        report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
            .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
            .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
            .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
            .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                        .replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(" ".join(self.do_filter(sent.split()))) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    def get_token_by_id(self, id):
        return self.idx2token[id]

    def get_id_by_token(self, token):
        if token not in self.token2idx:
            return self.token2idx['<unk>']
        return self.token2idx[token]

    def get_vocab_size(self):
        return len(self.token2idx)

    def __call__(self, report):
        tokens = self.clean_report(report).split(".")
        ids = []
        for token in tokens:
            ids.append(self.get_id_by_token(token+"."))
        ids = [0] + ids + [0]
        return ids

    def decode(self, ids):
        txt = ''
        for i, idx in enumerate(ids):
            if idx > 0:
                if i >= 1:
                    txt += ' '
                txt += self.idx2token[idx]
            else:
                break
        return txt

    def decode_batch(self, ids_batch):
        out = []
        for ids in ids_batch:
            out.append(self.decode(ids))
        return out
