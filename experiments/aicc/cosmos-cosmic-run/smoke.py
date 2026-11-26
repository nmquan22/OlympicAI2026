# OlympicAI2026 | AICC post-contest practice smoke test
# Fork lineage: Anindyo, notebookca40e6396b (Apache 2.0).
# Independent implementation below; no external pretrained model.
# CPU, seed 42. Saves predictions, validation splits, per-race metrics and JSON report.

# Post-contest practice. No manually labelled validation/test records.
# Baseline idea: https://www.kaggle.com/code/anindyo237/notebookca40e6396b
# This implementation adds leakage checks, biome holdout, ablations and audit files.
import json, hashlib, platform, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import kendalltau
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor

SEED = 42
ITERATIONS = 600
STATS = ['thermal_shield', 'cryo_insulation', 'wide_treads', 'hull_hardness', 'compressor']
VARIANTS = ['numeric_catboost', 'categorical_biome_catboost', 'char_interaction_ridge']
OUT = Path('/kaggle/working')
ROOTS = [Path('/kaggle/input/competitions/cosmos-cosmic-run-aicc-round-11'), Path('/kaggle/input/cosmos-cosmic-run-aicc-round-11')]
DATA = next((p for p in ROOTS if (p / 'train.csv').is_file()), None)
assert DATA is not None, 'Attach Cosmos Cosmic Run competition data first'
train = pd.read_csv(DATA / 'train.csv').reset_index(drop=True)
test = pd.read_csv(DATA / 'test.csv').reset_index(drop=True)
for frame in [train, test]:
    assert set(STATS + ['race_id', 'biome']).issubset(frame.columns)
    frame['biome'] = frame['biome'].fillna('UNKNOWN').astype(str)
    assert frame.groupby('race_id')['biome'].nunique().max() == 1
assert 'id' in test and test['id'].is_unique
assert 'finish_time' in train and np.isfinite(train['finish_time']).all()
assert not set(train.race_id) & set(test.race_id)


def features(frame, medians):
    result = pd.DataFrame(index=frame.index)
    result['biome'] = frame['biome']
    for col in STATS:
        values = pd.to_numeric(frame[col], errors='coerce').fillna(medians[col])
        result[col] = values
        result[col + '_rank'] = values.groupby(frame['race_id']).rank(pct=True)
        mean = values.groupby(frame['race_id']).transform('mean')
        std = values.groupby(frame['race_id']).transform('std').replace(0, np.nan)
        result[col + '_relative'] = ((values - mean) / std).fillna(0)
    return result


def target(frame):
    rank = frame.groupby('race_id')['finish_time'].rank(method='average')
    n = frame.groupby('race_id')['finish_time'].transform('size')
    return (rank - 1) / (n - 1).clip(lower=1)


def fit_model(frame, variant):
    medians = frame[STATS].apply(pd.to_numeric, errors='coerce').median().fillna(0)
    x = features(frame, medians)
    y = target(frame)
    artifact = {'variant': variant, 'medians': medians}
    if variant.endswith('catboost'):
        cols = list(x.columns) if variant.startswith('categorical') else list(x.columns[1:])
        model = CatBoostRegressor(iterations=ITERATIONS, depth=6, learning_rate=0.05,
                                  loss_function='RMSE', random_seed=SEED, thread_count=2,
                                  verbose=False, allow_writing_files=False)
        model.fit(x[cols], y, cat_features=['biome'] if 'biome' in cols else [])
        artifact.update(model=model, cols=cols)
    else:
        vect = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=1500)
        vect.fit(frame['biome'].drop_duplicates())
        z = x.drop(columns='biome').to_numpy(dtype=float)
        center, scale = z.mean(0), np.maximum(z.std(0), 1e-6)
        artifact.update(vect=vect, center=center, scale=scale)
        model = Ridge(alpha=10.0, solver='lsqr', tol=1e-6)
        artifact['model'] = model
        model.fit(design(frame, artifact), y)
    return artifact


def design(frame, artifact):
    z = features(frame, artifact['medians']).drop(columns='biome').to_numpy(dtype=float)
    z = (z - artifact['center']) / artifact['scale']
    text = artifact['vect'].transform(frame['biome'])
    return sparse.hstack([sparse.csr_matrix(z)] + [text.multiply(z[:, j:j+1]) for j in range(z.shape[1])], format='csr')


def predict(frame, artifact):
    if artifact['variant'].endswith('catboost'):
        x = features(frame, artifact['medians'])[artifact['cols']]
    else:
        x = design(frame, artifact)
    return artifact['model'].predict(x)


def evaluate(frame, scores):
    observed = frame[['race_id', 'finish_time']].copy()
    observed['score'] = scores
    rows = []
    for race, group in observed.groupby('race_id', sort=False):
        assert len(group) >= 2
        tau = float(kendalltau(group['finish_time'], group['score']).statistic)
        assert np.isfinite(tau), 'Undefined Kendall tau; inspect tied predictions/labels'
        rows.append({'race_id': race, 'kendall_tau': tau, 'ships': len(group)})
    per_race = pd.DataFrame(rows)
    raw = float(per_race.kendall_tau.mean())
    return raw, float(np.clip(raw, 0, 1)), per_race


started = time.perf_counter()
records = []
for split_name, group_col in [('race_holdout', 'race_id'), ('unseen_biome_holdout', 'biome')]:
    assert train[group_col].nunique() >= 2
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    tr_idx, va_idx = next(splitter.split(train, groups=train[group_col]))
    tr, va = train.iloc[tr_idx].copy(), train.iloc[va_idx].copy()
    assert not set(tr.race_id) & set(va.race_id)
    overlap = sorted(set(tr.biome) & set(va.biome))
    if group_col == 'biome':
        assert not overlap
    pd.DataFrame({'row_index': tr_idx, 'partition': 'train'}).to_csv(OUT / (split_name + '_train_rows.csv'), index=False)
    pd.DataFrame({'row_index': va_idx, 'partition': 'validation'}).to_csv(OUT / (split_name + '_validation_rows.csv'), index=False)
    for variant in VARIANTS:
        tick = time.perf_counter()
        model = fit_model(tr, variant)
        scores = predict(va, model)
        raw, metric, per_race = evaluate(va, scores)
        record = dict(split=split_name, variant=variant, raw_mean_kendall=raw, score=metric,
                      train_rows=len(tr), validation_rows=len(va), validation_races=va.race_id.nunique(),
                      overlapping_biomes=len(overlap), seed=SEED, seconds=time.perf_counter()-tick)
        records.append(record)
        per_race.to_csv(OUT / (split_name + '_' + variant + '_per_race.csv'), index=False)
        pd.DataFrame({'row_index': va.index, 'race_id': va.race_id, 'finish_time': va.finish_time, 'score': scores}).to_csv(OUT / (split_name + '_' + variant + '_predictions.csv'), index=False)
        print('EXPERIMENT ' + json.dumps(record), flush=True)

results = pd.DataFrame(records)
results.to_csv(OUT / 'experiments.csv', index=False)
selection = results[results.split == 'unseen_biome_holdout'].sort_values('raw_mean_kendall', ascending=False, kind='stable').iloc[0]
best_variant = selection['variant']
checks = []
for variant, filename in [(best_variant, 'submission.csv'), ('categorical_biome_catboost', 'submission_baseline.csv')]:
    full_model = fit_model(train, variant)
    scores = predict(test, full_model)
    ranks = pd.Series(scores, index=test.index).groupby(test['race_id']).rank(method='first', ascending=True).astype(int)
    submission = pd.DataFrame({'id': test['id'], 'predicted_rank': ranks})
    assert len(submission) == len(test) and submission.id.is_unique
    assert submission.id.tolist() == test.id.tolist()
    for _, idx in test.groupby('race_id').groups.items():
        assert sorted(submission.loc[idx, 'predicted_rank']) == list(range(1, len(idx)+1))
    sample = DATA / 'sample_submission.csv'
    if sample.exists():
        sample_df = pd.read_csv(sample)
        assert list(sample_df.columns) == list(submission.columns)
        assert set(sample_df.id) == set(submission.id)
    submission.to_csv(OUT / filename, index=False)
    checks.append({'file': filename, 'variant': variant, 'rows': len(submission), 'races': int(test.race_id.nunique()), 'valid_permutations': True})
    if filename == 'submission.csv':
        import joblib
        joblib.dump(full_model, OUT / 'selected_model.joblib')

summary = dict(status='completed', mode='post_contest_practice', competition='cosmos-cosmic-run-aicc-round-11',
               seed=SEED, iterations=ITERATIONS, selected_variant=best_variant, selection_split='unseen_biome_holdout',
               train_shape=list(train.shape), test_shape=list(test.shape), train_biomes=int(train.biome.nunique()),
               test_biomes=int(test.biome.nunique()), unseen_test_biomes=len(set(test.biome)-set(train.biome)),
               experiments=records, submission_checks=checks, python=platform.python_version(),
               input_sha256={name: hashlib.sha256((DATA/name).read_bytes()).hexdigest() for name in ['train.csv', 'test.csv']},
               seconds=time.perf_counter()-started, leaderboard_score=None,
               limitations=['One fixed holdout per split; no claim of statistical significance.',
                            'No semantic pretrained embeddings; unseen synonyms may remain difficult.',
                            'A saved notebook version is not evidence of a scored competition submission.'])
(OUT / 'smoke_report.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print('SMOKE_RESULT_JSON ' + json.dumps(summary), flush=True)
print('SMOKE_TEST_PASSED: 6 validation runs and 2 checked submissions', flush=True)

