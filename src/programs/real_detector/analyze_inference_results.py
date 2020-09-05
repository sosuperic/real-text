
"""
Analyze outputs

PYTHONPATH=. python src/programs/real_detector/analyze_inference_results.py
"""
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, chisquare, fisher_exact 

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html


from config import ETS_NONNATIVE_PATH, SCHOOL_REVIEWS_DATA
from src.utils import load_file, save_file




def analyze_ets():
    fp = ETS_NONNATIVE_PATH / 'index.csv'
    df = pd.read_csv(fp)

    #####################################################
    # Loading inference results

    ###### First pass with trainer (model was trained on gpt2 gen text)
    # correctness = load_file('outputs/gpt2gens_detector/ets/evalETS_v0.0.json')
    # correctness = load_file('outputs/gpt2gens_detector/ets/evalETS_v0.1.json')
    ######

    ###### OpenAI detector on ets / school reviews
    correctness = {}
    data = load_file('outputs/openai_detector/ets/results.json')
    # breakpoint()

    # correctness = {fn: d['correct'] for fn, d in data.items()}
    # The default 'correct' value is based on a 0.5 threshold. Set a new threshold here
    threshold = 0.94
    print(threshold)
    for fn, d in data.items():
        correct = d['real'] > threshold
        correctness[fn] = correct
    ######

    ####################################################
    df['correct'] = np.NaN
    # df.correct = df.correct.astype('bool')
    df.set_index('Filename', inplace=True)

    for fn, correct in correctness.items():
        correct = 1 if correct else 0
        df.set_value(fn, 'correct', correct)

    df = df[df.correct.notnull()]

    print('-' * 100)
    print('Counts')
    print(df.groupby('Score Level').correct.count())
    print('-' * 100)
    print('Mean')
    print(df.groupby('Score Level').correct.mean())
    # print('-' * 100)
    # print('Std')
    # print(df.groupby('Score Level').std())

    print('-' * 100)
    print('Fisher tabs')
    # fisher exact
    # https://medium.com/@robertmckee/statistical-analysis-hypothesis-testing-of-binary-data-b0dce43306
    # https://en.wikipedia.org/wiki/Fisher%27s_exact_test
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html
    print('high-medium')
    tab = pd.crosstab(df[df['Score Level'].isin(['high', 'medium'])]['Score Level'], df.correct)
    print(fisher_exact(tab))
    print('medium-low')
    tab = pd.crosstab(df[df['Score Level'].isin(['medium', 'low'])]['Score Level'], df.correct)
    print(fisher_exact(tab))
    print('high-low')
    tab = pd.crosstab(df[df['Score Level'].isin(['high', 'low'])]['Score Level'], df.correct)
    print(fisher_exact(tab))


    breakpoint()
    # print(ttest_ind(df[df['Score Level'] == 'high'].correct, df[df['Score Level'] == 'medium'].correct))
    # print(ttest_ind(df[df['Score Level'] == 'medium'].correct, df[df['Score Level'] == 'low'].correct))
    # print(ttest_ind(df[df['Score Level'] == 'high'].correct, df[df['Score Level'] == 'low'].correct))


    print('=' * 100)
    print(df.groupby('Language').count())
    print('-' * 100)
    print(df.groupby('Language').mean())


    breakpoint()


def analyze_school_reviews():
    df = pd.read_csv(SCHOOL_REVIEWS_DATA)


    ###### OpenAI detector on ets / school reviews
    correctness = {}
    data = load_file('outputs/openai_detector/school_reviews/results.json')

    # correctness = {fn: d['correct'] for fn, d in data.items()}
    # The default 'correct' value is based on a 0.5 threshold. Set a new threshold here
    threshold = 0.5
    print(threshold)
    for fn, d in data.items():
        correct = d['real'] > threshold
        correctness[fn] = correct
    ######

    # Index(['url', 'review_text', 'mn_grd_eb', 'mn_avg_eb', 'top_level', 'perwht',
    #    'perfrl', 'totenrl', 'gifted_tot', 'lep', 'disab_tot_idea', 'disab_tot',
    #    'perind', 'perasn', 'perhsp', 'perblk', 'perfl', 'perrl',
    #    'nonwhite_share2010', 'med_hhinc2016', 'mail_return_rate2010',
    #    'traveltime15_2010', 'poor_share2010', 'frac_coll_plus2010',
    #    'jobs_total_5mi_2015', 'jobs_highpay_5mi_2015',
    #    'ann_avg_job_growth_2004_2013', 'singleparent_share2010',
    #    'popdensity2010', 'urbanicity'],
    #   dtype='object')

    ####################################################
    df['correct'] = np.NaN
    # df.correct = df.correct.astype('bool')
    df.set_index('url', inplace=True)

    for fn, correct in correctness.items():
        correct = 1 if correct else 0
        df.set_value(fn, 'correct', correct)
    df = df[df.correct.notnull()]

    # create some categorical (binary) buckets out of continous variables
    df['singleparent_share2010_aboveavg'] = df.singleparent_share2010 > df.singleparent_share2010.mean()
    df['perwht_aboveavg'] = df.perwht > df.perwht.mean()

    # breakpoint()

    print('-' * 100)
    print('Mean')
    print(df.groupby('urbanicity').correct.mean())
    print('-' * 100)
    print('Mean')
    print(df.groupby('singleparent_share2010_aboveavg').correct.mean())
    print('-' * 100)
    print('Mean')
    print(df.groupby('perwht_aboveavg').correct.mean())

    breakpoint()

if __name__ == '__main__':
    # analyze_ets()
    analyze_school_reviews()