from pathlib import Path

ROOT_PATH = '/mas/u/echu/projects/research/real-text/'

DATA_PATH = Path('data/')

REAL_TEXT_PATH = DATA_PATH / 'gpt-2-output-dataset/data/'
GENERATED_TEXT_PATH = DATA_PATH / 'generations/'
PREPPED_REALGEN_TEXT_PATH = DATA_PATH / 'realgen_prepped/'

# REALNEWS_REAL_PATH = DATA_PATH / 'realnews' / 'realnews.jsonl'
REALNEWS_PREPPED_PATH = DATA_PATH / 'realnews' / 'generator=mega~dataset=p0.94.jsonl'
# REALNEWS_PREPPED_PATH = DATA_PATH / 'realnews' / 'prepped/'

ETS_NONNATIVE_PATH = DATA_PATH / 'ets_nonnative/data/text'


SCHOOL_REVIEWS_PATH = DATA_PATH / 'school_reviews'
SCHOOL_REVIEWS_DATA = SCHOOL_REVIEWS_PATH / 'Parent_gs_comments_by_school_with_covars.csv'
SCHOOL_REVIEWS_TRAINGPT2_PATH = SCHOOL_REVIEWS_PATH / 'train_gpt2'