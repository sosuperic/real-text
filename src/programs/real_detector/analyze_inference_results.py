
"""
Analyze outputs

PYTHONPATH=. python src/programs/real_detector/analyze_inference_results.py
"""
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, chisquare, fisher_exact 

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html


from config import ETS_NONNATIVE_PATH
from src.utils import load_file, save_file

def analyze():
    fp = ETS_NONNATIVE_PATH / 'index.csv'
    df = pd.read_csv(fp)

    # correctness = load_file('outputs/gpt2gens_detector/ets/evalETS_v0.0.json')
    # correctness = load_file('outputs/gpt2gens_detector/ets/evalETS_v0.1.json')

    correctness = {}
    data = load_file('outputs/openai_detector/ets/ets_results.json')
    # correctness = {fn: d['correct'] for fn, d in data.items()}
    # The default 'correct' value is based on a 0.5 threshold. Set a new threshold here
    threshold = 0.99
    print(threshold)
    for fn, d in data.items():
        correct = d['real'] > 0.9
        correctness[fn] = correct

    df['correct'] = np.NaN
    # df.correct = df.correct.astype('bool')
    df.set_index('Filename', inplace=True)

    for fn, correct in correctness.items():
        correct = 1 if correct else 0
        df.set_value(fn, 'correct', correct)

    df = df[df.correct.notnull()]

    print('-' * 100)
    print('Counts')
    print(df.groupby('Score Level').count())
    print('-' * 100)
    print('Mean')
    print(df.groupby('Score Level').mean())
    # print('-' * 100)
    # print('Std')
    # print(df.groupby('Score Level').std())

    print('-' * 100)
    print('Fisher tabs')
    # fisher exact
    # https://medium.com/@robertmckee/statistical-analysis-hypothesis-testing-of-binary-data-b0dce43306
    # https://en.wikipedia.org/wiki/Fisher%27s_exact_test
    tab = pd.crosstab(df[df['Score Level'].isin(['high', 'medium'])]['Score Level'], df.correct)
    print(fisher_exact(tab))
    tab = pd.crosstab(df[df['Score Level'].isin(['medium', 'low'])]['Score Level'], df.correct)
    print(fisher_exact(tab))
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


if __name__ == '__main__':
    analyze()