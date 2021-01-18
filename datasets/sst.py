'''
The majority of the code comes from https://pytorch.org/text/stable/_modules/torchtext/datasets/sst.html
Small alterations were added to incorporate BERT and project structure.
Most of this will most likely be deprecated in the near future, but other than providing an ad hoc solution,
this seems most elegant.
'''

import os

import torch
import torchtext
import torchtext.data as data
from torchtext.data import Dataset, Field

MAX_SEQ_LEN = 82 # Tested on training dataset
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
INIT_TOKEN = '<s>'
EOS_TOKEN = '<\s>'

class SST(data.Dataset):

    urls = ['http://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip']
    dirname = 'trees'
    name = 'SST'

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path, text_field, label_field, subtrees=False,
                 fine_grained=False, **kwargs):
        """Create an SST dataset instance given a path and fields.

        Arguments:
            path: Path to the data file
            text_field: The field that will be used for text data.
            label_field: The field that will be used for label data.
            subtrees: Whether to include sentiment-tagged subphrases
                in addition to complete examples. Default: False.
            fine_grained: Whether to use 5-class instead of 3-class
                labeling. Default: False.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """

        fields = [('text', text_field), ('label', label_field)]

        def get_label_str(label):
            pre = 'very ' if fine_grained else ''
            return {'0': pre + 'negative', '1': 'negative', '2': 'neutral',
                    '3': 'positive', '4': pre + 'positive', None: None}[label]

        label_field.preprocessing = data.Pipeline(get_label_str)
        with open(os.path.expanduser(path)) as f:
            if subtrees:
                examples = [ex for line in f for ex in
                            data.Example.fromtree(line, fields, True)]
            else:
                examples = [data.Example.fromtree(line, fields) for line in f]

        if fine_grained:
            filter_pred = lambda ex: ex.label in ['negative', 'positive']
        else:
            filter_pred = None

        super(SST, self).__init__(examples, fields, filter_pred)

    @ classmethod
    def splits(cls, text_field, label_field, root='./datasets',
               train='train.txt', validation='dev.txt', test='test.txt',
               train_subtrees=False, **kwargs):
        """Create dataset objects for splits of the SST dataset.

            Arguments:
                text_field: The field that will be used for the sentence.
                label_field: The field that will be used for label data.
                root: The root directory that the dataset's zip archive will be
                    expanded into; therefore the directory in whose trees
                    subdirectory the data files will be stored.
                train: The filename of the train data. Default: 'train.txt'.
                validation: The filename of the validation data, or None to not
                    load the validation set. Default: 'dev.txt'.
                test: The filename of the test data, or None to not load the test
                    set. Default: 'test.txt'.
                train_subtrees: Whether to use all subtrees in the training set.
                    Default: False.
                Remaining keyword arguments: Passed to the splits method of
                    Dataset.
            """
        path = os.path.join(root, cls.name, cls.dirname)
        _ = cls.download(root, check=path)

        train_data = None if train is None else cls(
            os.path.join(path, train), text_field, label_field, subtrees=train_subtrees,
            **kwargs)
        val_data = None if validation is None else cls(
            os.path.join(path, validation), text_field, label_field, **kwargs)
        test_data = None if test is None else cls(
            os.path.join(path, test), text_field, label_field, **kwargs)

        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)

    @ classmethod
    def iters(cls, batch_size=32, device=None, root='./datasets', repeat=True, vectors=None, **kwargs):
        """Create iterator objects for splits of the SST dataset.

            Arguments:
                batch_size: Batch_size
                device: Device to create batches on. Use - 1 for CPU and None for
                    the currently active GPU device.
                root: The root directory that the dataset's zip archive will be
                    expanded into; therefore the directory in whose trees
                    subdirectory the data files will be stored.
                vectors: one of the available pretrained vectors or a list with each
                    element one of the available pretrained vectors (see Vocab.load_vectors)
                Remaining keyword arguments: Passed to the splits method.
            """

        LABEL = Field(sequential=False, batch_first=False, is_target=True)
        TEXT = Field(use_vocab=True, lower=True, include_lengths=False,
                     batch_first=False, fix_length=MAX_SEQ_LEN,
                     pad_token=PAD_TOKEN, unk_token=UNK_TOKEN,
                     init_token=INIT_TOKEN, eos_token=EOS_TOKEN)

        train, val, test = cls.splits(TEXT, LABEL, root=root, **kwargs)

        LABEL.build_vocab(train, specials_first=False)
        if vectors != None:
            TEXT.build_vocab(train, specials_first=True, vectors=vectors)
        else:
            TEXT.build_vocab(train, specials_first=True)

        vocab = TEXT.vocab

        return (data.BucketIterator.splits((train, val, test), batch_size=batch_size,
                                           repeat=repeat, device=device),
                (vocab, train))
