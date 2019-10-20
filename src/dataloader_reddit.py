# -*- coding: utf-8 -*-
"""Wrapper for the reddit loaders of dbiscorpora."""
from dbispipeline.base import TrainTestLoader, Loader
from sklearn.preprocessing import LabelBinarizer
import os
from glob import glob

def _get_all_categories(directory): 
    return set([os.path.basename(x) for x in glob(f'{directory}/*/*')])


def load_reddit_single(directory, category): 
    return load_reddit(directory, [category], [category])

def load_reddit(directory, train_categories, test_categories):
    if not os.path.isdir(directory):
        raise Exception('dataset not available: %s' % directory)
    xtrain, ytrain, xtest, ytest = [], [], [], []

    all_c = _get_all_categories(directory)
    test_c = set(test_categories)
    train_c = set(train_categories)

    single_category = train_c == test_c

    if train_c - all_c: 
        raise Exception(f'requested training category ({train_c - all_c}) '
                f'is not available from the categories {all_c}')
    if test_c - all_c:
        raise Excgveption(f'requested testing category ({test_c - all_c}) is not available from the categories {all_c}.')
    if len(test_c & train_c) == 1 and len(train_c) > 1:
        raise Exception(f'using multiple categories with one of them overlapping for training and testing ({test_c & train_c}) is not supported')
    if len(test_c & train_c) > 1:
        raise Exception(f'using more than one overlapping category for training and testing ({test_c & train_c}) is not supported')


    for i, author in enumerate(sorted(os.listdir(directory))):
        author_dir = os.path.join(directory, author)
        if not os.path.isdir(author_dir):
            continue

        for category in all_c:
            category_dir = os.path.join(author_dir, category)
            if not os.path.isdir(category_dir): 
                continue
            if category not in train_c and category not in test_c:
                continue
            data = sorted(glob(f'{category_dir}/*'))
            data = [x for x in data if '.json' in x]
            labels = [author] * len(data)

            if single_category or category in train_c:
                xtrain += data
                ytrain += labels
            elif category in test_c:
                xtest += data
                ytest += labels

    if single_category:
        return xtrain, ytrain, train_categories
    else:
        return xtrain, ytrain, xtest, ytest, train_categories, test_categories


class RedditLoader(TrainTestLoader): 

    def __init__(self, corpus_path, train_categories, test_categories, load_classes_one_hot=False):
        data = load_reddit(corpus_path, train_categories, test_categories)
        xtrain, ytrain, xtest, ytest, train_categories, test_categories = data
        if load_classes_one_hot:
            lb = LabelBinarizer()
            lb.fit(ytrain)
            ytrain = lb.transform(ytrain)
            ytest = lb.transform(ytest)
        self.train = (xtrain, ytrain)
        self.corpus_path = corpus_path
        self.test = (xtest, ytest)
        self.train_categories = train_categories
        self.test_categories = test_categories

    def load_train(self):
        """Returns the train data."""
        return self.train

    def load_test(self):
        """Returns the test data."""
        return self.test

    def load_run_id(self):
        return f"{self.corpus_path} - training with: {self.train_categories} - testing with: {self.test_categories}"

    @property
    def configuration(self):
        """Returns a dict-like representation of the configuration."""
        return {
            'corpus_path': self.corpus_path,
            'train_categories': self.train_categories,
            'test_categories': self.test_categories,
        }


class RedditCrossCategoryLoader(RedditLoader):
    """Wrapper class for the reddit cross category loader of dbiscorpora."""
    def __init__(self, corpus_path, train_category, test_category):
        super().__init__(corpus_path, [train_category], [test_category])

class RedditSingleCategoryLoader(Loader):

    def __init__(self, corpus_path, category):
        self.corpus_path = corpus_path
        self.data, self.labels, self.category = load_reddit_single(corpus_path, category)
    def load(self): 
        return self.data, self.labels

    @property
    def configuration(self):
        """Returns a dict-like representation of the configuration."""
        return {
            'corpus_path': self.corpus_path,
            'category': self.category,
        }

