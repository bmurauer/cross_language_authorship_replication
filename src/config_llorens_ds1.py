from dataloader import LlorensSingleLoader
from feature_extraction import LifeVectorizer
from pipeline_tools import CustomCallbackTransformer, FileReader

from dbispipeline import result_handlers
from dbispipeline.base import MultiLoaderGenerator
from dbispipeline.evaluators import CustomCvGridEvaluator

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

dataloader = LlorensSingleLoader()

pipeline = Pipeline([
    ('reader', FileReader()),
    ('splitter', CustomCallbackTransformer(lambda x: x.split())),
    ('life', LifeVectorizer(force=True)),
    ('rf', RandomForestClassifier(n_estimators=10, n_jobs=1)),
])

model_params = {
    'life__sample_type': ['fragment', 'bow', 'both'],
    'life__fragment_sizes': [
        [200, 500, 800, 1000, 1500, 2000, 3000, 4000],
        [200, 500, 800, 1000, 1500, 2000, 3000],
        [200, 500, 800, 1000, 1500, 2000],
        [200, 500, 800, 1000, 1500],
        [200, 500, 800, 1000],
        [200, 500, 800],
        [200, 500],
        [200],
    ],
}

grid_params ={
    'n_jobs': -1,
    'verbose': 1,
    'iid': False,
    'return_train_score': False,
    'refit': False,
}

evaluator = CustomCvGridEvaluator(model_params, grid_params)

result_handlers = []
