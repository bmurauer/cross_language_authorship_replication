# -*- coding: utf-8 -*-
# This transformers are a re-implementation of the feature extraction method
# presented in Llorens(2016).

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from collections import defaultdict
import numpy as np
import random

class LifeVectorizer(BaseEstimator, TransformerMixin):

    def __init__(self, fragment_sizes=[200, 500, 800, 1000, 1500, 2000, 3000, 4000], samples=200, sample_type='bow', force=True):
        valid_sample_types = ['bow', 'fragment', 'both']
        if sample_type not in valid_sample_types:
            raise ValueError(f'unknown sample type: {sample_type}. valid values: {valid_sample_types}')
        self.fragment_sizes = fragment_sizes
        self.samples = samples
        self.sample_type = sample_type
        self.force = force

    def fit(self, X, y=None):
        return self

    def sample(self, words, fragment_size, method):
        ret = []
        wordcount = len(words)
        if wordcount < fragment_size: 
            if self.force:
                return [words] * self.samples
            else:
                raise ValueError(f'fragment size ({fragment_size}) is larger than document size ({wordcount}) for document starting with: \n\n{document[:250]}\n\n')
        for i in range(self.samples): 
            if method == 'fragment': 
                left = random.randint(0, wordcount - fragment_size)
                right = left + fragment_size
                ret.append(words[left:right])
            if method == 'bow': 
                ret.append(random.sample(words, fragment_size))
        return ret

    def get_features_for_sample(self, sample):
        counts = defaultdict(int)
        for word in sample: 
            counts[word] += 1
        v0 = len(counts.keys())
        v1, v2, v3 = 0, 0, 0
        for word, occurrances in counts.items():
            if occurrances <= 1:
                v1 += 1
            elif occurrances <= 4:
                v2 += 1
            elif occurrances <= 10:
                v3 += 1
        
        return [v0, v1, v2, v3]


    def get_features(self, document, sample_size): 
        if self.sample_type == 'both': 
            return np.concatenate([
                self._get_features(document, sample_size, 'bow'),
                self._get_features(document, sample_size, 'fragment'),
            ])
        else:
            return self._get_features(document, sample_size, self.sample_type)


    def _get_features(self, document, fragment_size, method):
        samples = self.sample(document, fragment_size, method)
        features = []
        for sample in samples:
            features.append(self.get_features_for_sample(sample))
        features = np.array(features)
        means = np.mean(features, axis=0)
        stds = np.std(features, axis=0)
        return np.concatenate([
            means, 
            np.divide(means, stds, out=np.zeros_like(means), where=stds!=0)
        ])


    def transform(self, X, y=None):
        ret = []
        for document in X:
            doc = [self.get_features(document, size) for size in self.fragment_sizes]
            ret.append(np.concatenate(doc))
        return ret
