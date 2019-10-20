# -*- coding: utf-8 -*-
"""Wrapper for the reddit loaders of dbiscorpora."""
import os
from dbispipeline.base import Loader
from sklearn.model_selection import PredefinedSplit, BaseCrossValidator, LeaveOneOut
from glob import glob
import itertools
import re
from collections import defaultdict
import numpy as np
import logging
logging.basicConfig(level='INFO', format='%(asctime)s %(levelname)s: %(message)s')

DEFAULT_LLORENS_PATH = '../data/llorens'
DEFAULT_LLORENS_PATTERN = r'.*/(?P<author>\w+)/(?P<language>\w+)/(?P<title>.*).txt'

DEFAULT_BOGDANOVA_PATH = '../data/bogdanova'
DEFAULT_BOGDANOVA_PATTERN_CHUNKS = r'.*/(?P<author>\w+)/(?P<language>\w+)/chunks/\d\d_(?P<title>.*)_\d\d\d.txt'

class NovelsCrossValidator(BaseCrossValidator):

    def __init__(self, pattern, use_chunks, strict_languages=True, strict_titles=True):
        self.pattern = pattern
        self.use_chunks = use_chunks
        self.strict_titles = strict_titles
        self.strict_languages = strict_languages

    def get_n_splits(self, X, y, groups=None): 
        return len(list(self.split(X, y)))

    def split(self, X, y, groups=None):
        files = [re.match(self.pattern, x) for x in X]
        splits = []
        if self.use_chunks: 
            tested_titles = set()
            for i, test_file in enumerate(files): 
                test_title = test_file['title']
                test_lang = test_file['language']
                test_string = f'{test_title}_{test_file["author"]}_{test_lang}'
                if test_string in tested_titles: 
                    continue
                tested_titles.add(test_string)
                test_indices = [i]
                train_indices = []
                train_authors = set()

                for j, other_file in enumerate(files): 
                    other_lang = other_file['language']
                    other_title = other_file['title']
                    other_string = f'{other_title}_{other_file["author"]}_{other_lang}'

                    if i == j:
                        continue
                    
                    # a chunk of the same title can be used for testing 
                    # simultaneously
                    if other_string == test_string: 
                        test_indices.append(j)
                        continue 
                    if self.strict_titles and test_file['title'] == other_file['title']:
                        continue
                    if self.strict_languages and test_file['language'] == other_file['language']:
                        continue
                    train_indices.append(j)
                    train_authors.add(other_file['author'])

                if test_file['author'] not in train_authors:
                    continue
                splits.append((train_indices, test_indices))
            return splits

        else:
            for i, test_file in enumerate(files): 
                test_indices = [i]
                train_indices = []
                train_authors = set()
                for j, train_file in enumerate(files): 
                    if i == j:
                        continue
                    if self.strict_titles and test_file['title'] == train_file['title']:
                        continue
                    if self.strict_languages and test_file['language'] == train_file['language']:
                        continue
                    train_indices.append(j)
                    train_authors.add(train_file['author'])

                if test_file['author'] not in train_authors:
                    continue
                splits.append((train_indices, test_indices))
            return splits


class NovelsLoader(Loader):  

    def __init__(self, basedir, pattern, use_chunks=False, strict_titles=True, strict_languages=True):
        self.basedir = basedir
        self.pattern = pattern
        self.use_chunks = use_chunks
        self.strict_titles = strict_titles
        self.strict_languages = strict_languages

    def load(self):
        if self.use_chunks:
            all_textfiles = glob(f'{self.basedir}/*/*/chunks/*.txt')
        else:
            all_textfiles = glob(f'{self.basedir}/*/*/*.txt')
        self.titles = []
        for t in all_textfiles:
            m = re.match(self.pattern, t)
            if not m: 
                raise ValueError(f'document: {t} doesn\'t match pattern: {self.pattern}')
            self.titles.append(m)
        
        data = np.array([x.string for x in self.titles])
        labels = np.array([x['author'] for x in self.titles])
        cv = NovelsCrossValidator(self.pattern, self.use_chunks, self.strict_languages, self.strict_titles)
        return data, labels, cv

    @property
    def configuration(self):
        return {
            'basedir': self.basedir,
            'pattern': self.pattern,
            'use_chunks': self.use_chunks,
            'strict_titles': self.strict_titles,
            'strict_languages': self.strict_languages,
        }


class LlorensLoader(NovelsLoader):
    def __init__(self, 
            path=DEFAULT_LLORENS_PATH, 
            pattern=DEFAULT_LLORENS_PATTERN):
        super().__init__(path, pattern, False, True, False)


class BogdanovaLoader(NovelsLoader): 
    def __init__(self, 
            path=DEFAULT_BOGDANOVA_PATH,
            pattern=DEFAULT_BOGDANOVA_PATTERN_CHUNKS):
        super().__init__(path, pattern, True, True, False)


class LlorensSingleLoader(NovelsLoader): 

    def __init__(self,
            path=DEFAULT_LLORENS_PATH,
            pattern=DEFAULT_LLORENS_PATTERN): 
        super().__init__(path, pattern, False, True, False)
        self.native_languages = {
            'blascoibanez': 'es',
            'dickens': 'en',
            'dumas': 'fr',
            'gerstacker': 'de',
            'goethe': 'de',
            'haggard': 'en',
            'verne': 'fr'
        }

    def load(self): 
        data, labels, cv = super().load()
        valid_indices = []
        for i, (filename, author) in enumerate(zip(data, labels)): 
            if author not in self.native_languages:
                continue
            native_language = self.native_languages[author]
            if re.match(self.pattern, filename)['language'] == native_language:
                valid_indices.append(i)
        return np.array(data)[valid_indices], np.array(labels)[valid_indices], cv


