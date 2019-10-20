# -*- coding: utf-8 -*-
from sklearn.base import BaseEstimator, TransformerMixin
import os
import json

class CustomCallbackTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, callback, per_document=True):
        self.callback = callback
        self.per_document = per_document

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if not self.per_document:
            return self.callback(X)
        ret = []
        for document in X:
            ret.append(self.callback(document))
        return ret


# -*- coding: utf-8 -*-

class FileReader(BaseEstimator, TransformerMixin):

    """
    Transforms filenames into their content.
    By default, this transformer will open files in text mode.

    input: list of paths
    output: list of contents of the files specified by the paths
    """

    def __init__(self, mode='r'):
        self.mode = mode

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        ret = []
        for document in X:
            if not os.path.isfile(document):
                raise Exception('NO SUCH FILE: ', document)
            with open(document, self.mode) as i_f:
                ret.append(i_f.read())
        return ret

class DictFieldTransformer(BaseEstimator, TransformerMixin):

    """
    extracts a field from a json object.
    This transformer expects json strings as input, not parsed dictionaries.
    """

    def __init__(self, field_name):

        """
        :param field_name: the field to extract from the string
        """
        self.field_name = field_name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        ret = []
        for document in X:
            ret.append(document[self.field_name])
        return ret

    
class JsonTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        ret = []
        for document in X:
            ret.append(json.loads(document))
            
        return ret
