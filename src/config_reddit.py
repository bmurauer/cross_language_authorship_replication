from dataloader_reddit import RedditCrossCategoryLoader
from pipeline_tools import FileReader, JsonTransformer, DictFieldTransformer
from feature_extraction import LifeVectorizer

from dbispipeline.base import MultiLoaderGenerator
from dbispipeline.evaluators import FixedSplitGridEvaluator
from dbispipeline import result_handlers

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

base = '/home/benjamin/local_datasets/reddit'

loader_parameters = [
    (f'{base}/R-CL1', 'en', 'de'),
    (f'{base}/R-CL1', 'de', 'en'),
    (f'{base}/R-CL2', 'en', 'es'),
    (f'{base}/R-CL2', 'es', 'en'),
    (f'{base}/R-CL3', 'en', 'pt'),
    (f'{base}/R-CL3', 'pt', 'en'),
    (f'{base}/R-CL4', 'en', 'nl'),
    (f'{base}/R-CL4', 'nl', 'en'),
    (f'{base}/R-CL5', 'en', 'fr'),
    (f'{base}/R-CL5', 'fr', 'en'),
    (f'{base}/R-CL6', 'en', 'ar'),
    (f'{base}/R-CL6', 'ar', 'en'),
]

dataloader = MultiLoaderGenerator(RedditCrossCategoryLoader, loader_parameters)

pipeline = Pipeline([
    ("filereader", FileReader()),
    ("json", JsonTransformer()),
    ("text", DictFieldTransformer('body')),
    ('life', LifeVectorizer(samples=20, force=True)),
    ('rf', RandomForestClassifier(n_estimators=10, n_jobs=1)),
])

evaluator = FixedSplitGridEvaluator (
    {
        'life__fragment_sizes': [[50], [50, 100], [50, 100, 200]],
    }, {
        'n_jobs': -1,
        'scoring': 'accuracy',
        'verbose': 1,
    }
)

result_handlers = []
