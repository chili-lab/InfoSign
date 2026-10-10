"""Detailed diagnostics and compact concept-recovery reports."""
from pathlib import Path

import numpy as np
import pandas as pd

from infosign.common import TASK_LABELS, read_json
from infosign.stats import auroc_selection, bh_adjust
from infosign.report import md
from infosign.cbm.diagnostics import VERSION


def ci(row, metric, signed=False):
    value = row.get(metric,np.nan)
    if pd.isna(value): return 'NA'
    fmt = '+.4f' if signed else '.4f'
    return f'{value:{fmt}} [{row[metric+"_ci_lower"]:{fmt}}, {row[metric+"_ci_upper"]:{fmt}}]'


def number(x): return 'NA' if pd.isna(x) else f'{x:.4g}'


def make_diagnostics_report(cfg):
    root = Path(cfg['output_root']); out = root/'tables/concept_diagnostics'
    out.mkdir(parents=True,exist_ok=True)
    tasks = [t for t in cfg['tasks'] if (root/t/'concept_diagnostics/DONE.json').exists()
             and read_json(root/t/'concept_diagnostics/DONE.json').get('version')==VERSION]
    if not tasks: return
    frames = {name:pd.concat([pd.read_csv(root/t/'concept_diagnostics'/(name+'.csv'),dtype={'concept_id':str})
                             for t in tasks],ignore_index=True)
              for name in ('quality','performance','comparisons','restoration')}
    r = frames['restoration']
    for metric in ('f1','ba','brier'):
        r[metric+'_q_value'] = np.nan
        eligible = r.learnable & (r.status=='evaluated') & r[metric+'_p_value'].notna()
        for _,idx in r[eligible].groupby(['split','source','population']).groups.items():
            r.loc[idx,metric+'_q_value'] = bh_adjust(r.loc[idx,metric+'_p_value'])
    # F1/BA remain reported outcomes. Brier ranks are supplementary/exploratory.
    eligible = r.learnable & (r.status=='evaluated') & r.brier_gain.notna()
    r['brier_rank_within_split'] = np.nan
    r.loc[eligible,'brier_rank_within_split'] = r[eligible].groupby(['task','split','source','population']).brier_gain.rank(method='min',ascending=False)
    for _, idx in r.groupby(['task','split','source','population']).groups.items():
        group = r.loc[idx]
        # Predicted restoration tests every single ID, including zero effects
        # from constant channels. Measured candidates need a usable target.
        evaluable = (group.source == 'full_input_prediction') | (group.status == 'evaluated')
        for key, values in auroc_selection(group.auroc_gain, evaluable).items():
            if key not in r: r[key] = False if key == 'auroc_selected' else np.nan
            r.loc[idx, key] = values
    rank_column = 'auroc_rank_within_split'
    frames['restoration'] = r.sort_values(['task','split','source','population',rank_column,'concept_id'],na_position='last')
    for split in ('val','test'):
        q,p,c,s = [frames[name].query('split == @split') for name in ('quality','performance','comparisons','restoration')]
        bank = 'without Observation' if cfg.get('exclude_observation') else 'with the four approved Observation surveys'
        lines = [f'# Mixed CBM concept diagnostics — {split}', '',
                 f'Concept bank: **{bank}**. Completed tasks: {len(tasks)}/{len(cfg["tasks"])}. '
                 f'Seed {cfg["seed"]}; {cfg["bootstrap_samples"]:,} paired, outcome-stratified bootstrap resamples.', '',
                 'No retraining, preprocessing, concept-bank changes, or threshold optimization was performed. '
                 'Mixed full, masked and supplemented predictions use the saved masked-validation threshold. '
                 'The original model uses its own saved threshold. Nonbasic clinical inputs are removed; '
                 'Basic Clinical and the other eight modalities remain available.', '',
                 '## Downstream probability and classification performance', '',
                 'Brier is the mean squared error of the positive-class probability (lower is better). '
                 'AUROC evaluates discrimination independently of a classification threshold. '
                 'F1 and BA retain their existing definitions and thresholds.', '',
                 md(['Task','Model/input','AUROC [95% CI]','Brier [95% CI]','F1 [95% CI]','BA [95% CI]'],
                    [[TASK_LABELS[x.task],x.condition,ci(x._asdict(),'auroc'),ci(x._asdict(),'brier'),ci(x._asdict(),'f1'),ci(x._asdict(),'ba')] for x in p.itertuples()]),
                 '## Information loss after removing nonbasic clinical inputs', '',
                 'Positive values mean removal worsened performance: Brier(masked) − Brier(full); '
                 'AUROC/F1/BA(full) − AUROC/F1/BA(masked).', '',
                 md(['Task','Model','AUROC drop [95% CI]','Brier increase [95% CI]','F1 drop [95% CI]','BA drop [95% CI]'],
                    [[TASK_LABELS[x.task],x.model,ci(x._asdict(),'auroc_gain',True),ci(x._asdict(),'brier_gain',True),ci(x._asdict(),'f1_gain',True),
                      ci(x._asdict(),'ba_gain',True)] for x in c.itertuples()]),
                 '## Concept prediction quality', '',
                 'Quality is evaluated per source/qualifier/unit item, because one Concept ID may have multiple measurements. '
                 'Only recorded, encodable labels are evaluated. The CSV includes missing/unknown-category counts. '
                 'Continuous items report MAE/RMSE in their recorded units (days for date-encoded items) and MSE in training-standardized units. '
                 'Categorical items report accuracy, macro recall over observed categories, log loss, and multiclass Brier '
                 '(sum over categories, range 0–2). Reference predictors use the training mean or training category frequencies. '
                 'Constants and items without training labels are retained in the CSV but excluded from this summary.', '']
        quality_summary = []
        for task in tasks:
            z=q[(q.task==task)&q.learnable&(q.n_evaluable>0)]
            continuous=z[z.type=='continuous'];categorical=z[z.type=='categorical']
            def count(df,column):
                return int((df.get(column,pd.Series(dtype=float))>0).sum())
            cont_better=int((continuous.get('masked_standardized_mse',pd.Series(dtype=float)) < continuous.get('train_mean_standardized_mse',pd.Series(dtype=float))).sum())
            cat_better=int((categorical.get('masked_multiclass_brier',pd.Series(dtype=float)) < categorical.get('train_prior_multiclass_brier',pd.Series(dtype=float))).sum())
            quality_summary.append([TASK_LABELS[task],len(continuous),cont_better,count(continuous,'masking_standardized_mse_increase'),
                                    len(categorical),cat_better,count(categorical,'masking_multiclass_brier_increase')])
        lines += [md(['Task','Continuous items','Masked better than training mean','Continuous worse after masking',
                      'Categorical items','Masked better than training prior','Categorical worse after masking'],quality_summary),
                  'Counts are descriptive, not significance tests; see CONCEPT_QUALITY for individual values and sample sizes.', '',
                  '## Single-ID supplementation', '',
                  'Each ID is changed separately from the same masked baseline. Full-input-prediction restoration replaces '
                  'all channels of that ID with full-input predicted concepts. Measured supplementation replaces only observed, '
                  'encodable, learnable items with their recorded targets; missing items and unseen categories stay predicted. '
                  'Continuous targets use saved training normalization; categorical targets use the saved category order and one-hot encoding. '
                  'No absence is interpreted as a negative response. Constant/untrained channels are not supplemented.', '',
                  'All-participant effects include people without available measurements (unchanged predictions). '
                  'Observed-participant effects compare exactly the same eligible people before/after supplementation; '
                  'eligible means at least one learnable item of the ID is available, so some IDs are only partially supplemented. '
                  'n_complete in the CSV counts people with all learnable items of the ID available. '
                  'Observed cohorts can differ between IDs; their gains should not be compared as if they came from one common population. '
                  'BA and AUROC are NA for cohorts containing only one outcome class.', '',
                  'Brier gain = baseline Brier − supplemented Brier; AUROC/F1/BA gain = supplemented − baseline. '
                  'Positive values are improvements. Test rankings are exploratory, and their intervals do not account '
                  'for selecting the largest test effects.', '']
        lines += [f'Tables report single Concept IDs with {split} ΔAUROC > μ + 2σ, ranked directly on {split}. '
                  'μ and σ are computed separately for each task and intervention source across all evaluated individual IDs; '
                  'σ is the population standard deviation (ddof=0). Predicted restoration includes constant IDs with zero gain. '
                  'Measured supplementation requires usable measured targets. Joint/all-ID restoration does not enter the distribution. '
                  'This is a descriptive outlier rule, not a significance test. No fixed top-10 padding or extra positivity filter is applied; '
                  'a selected negative gain is still a deterioration. Test candidates are not selected using validation ranks. '
                  'The CSV retains every ID with its AUROC rank, selection flag, μ, σ and cutoff; unevaluable IDs appear last. '
                  'AUROC has paired bootstrap intervals without permutation p/q values.', '']
        selection_rows = []
        for (task,source), group in s[s.population=='all_participants'].groupby(['task','source'],sort=False):
            first = group.iloc[0]
            selection_rows.append([TASK_LABELS[task],source,int(first.auroc_selection_n),
                                   number(first.auroc_selection_mean),number(first.auroc_selection_std),
                                   number(first.auroc_selection_cutoff),int(group.auroc_selected.sum())])
        lines += [md(['Task','Intervention source','Evaluated IDs','Mean ΔAUROC (μ)','SD (σ)','μ + 2σ','Selected IDs'],selection_rows)]
        for source,title in [('full_input_prediction','Full-input predicted concept restoration'),('measured','Measured concept supplementation')]:
            lines += [f'### {title} — all participants', '']
            for task in tasks:
                selected = s.auroc_selected
                z=s[(s.task==task)&(s.source==source)&(s.population=='all_participants')&selected]
                if source=='measured':z=z[z.n_eligible>0]
                z=z.sort_values([rank_column,'concept_id'])
                lines += [f'#### {TASK_LABELS[task]}', '']
                if z.empty:
                    lines += [f'No single Concept ID exceeded the {split} μ + 2σ cutoff.', ''];continue
                lines += [md([split.title()+' AUROC rank','Concept ID','Concept','Source fields','Eligible N / total',
                              'ΔAUROC [95% CI]','ΔBrier [95% CI]','Brier q','ΔF1 [95% CI]','ΔBA [95% CI]','Corrected / newly wrong'],
                    [[int(getattr(x,rank_column)),x.concept_id,x.concept_name,x.source_fields,f'{x.n_eligible}/{x.n_total}',
                      ci(x._asdict(),'auroc_gain',True),ci(x._asdict(),'brier_gain',True),number(x.brier_q_value),ci(x._asdict(),'f1_gain',True),
                      ci(x._asdict(),'ba_gain',True),f'{int(x.corrected)}/{int(x.newly_wrong)}'] for x in z.itertuples()])]
                if source=='measured':
                    obs=s[(s.task==task)&(s.source==source)&(s.population=='observed_participants')&s.concept_id.isin(z.concept_id)].sort_values([rank_column,'concept_id'])
                    lines += [f'The same candidates, evaluated only among participants with usable measured labels, sorted by their {split} '+'AUROC'+' gain in that subset:', '',
                              md(['Concept ID','Observed N','Positive / negative','ΔAUROC [95% CI]','ΔBrier [95% CI]','ΔF1 [95% CI]','ΔBA [95% CI]'],
                                 [[x.concept_id,x.n,f'{x.positive}/{x.negative}',ci(x._asdict(),'auroc_gain',True),ci(x._asdict(),'brier_gain',True),
                                   ci(x._asdict(),'f1_gain',True),ci(x._asdict(),'ba_gain',True)] for x in obs.itertuples()])]
        lines += ['## Interpretation and exports', '',
                  'Measured-value supplementation is an additional intervention experiment; benefit is not guaranteed. '
                  'A classifier trained on predicted concepts may respond differently to measured targets. '
                  'Small observed cohorts, outcome imbalance, multiple testing and a single training seed limit certainty. '
                  'No concept was added or removed according to these results.', '',
                  'F1, BA and Brier restoration q values use BH correction over learnable, evaluated IDs across completed tasks within '
                  'each split, source, population and metric. Full-input-prediction restoration and measured supplementation '
                  'are separate families; all IDs, not only displayed candidates, enter correction. '
                  'P values use paired prediction swaps. Resampling retains the observed outcome counts in each evaluated cohort. '
                  'Final q values require all configured tasks; intervals do not account for selection or training-seed variability.', '',
                  f'CSV: DOWNSTREAM_BRIER_{split}.csv; INPUT_REMOVAL_BRIER_{split}.csv; CONCEPT_QUALITY_{split}.csv; '
                  f'PREDICTED_RESTORATION_BRIER_{split}.csv; MEASURED_CONCEPT_SUPPLEMENTATION_{split}.csv. '
                  'Per-patient measured probabilities and availability are saved under each task/concept_diagnostics/.', '']
        for name,df in [('DOWNSTREAM_BRIER',p),('INPUT_REMOVAL_BRIER',c),('CONCEPT_QUALITY',q),
                        ('PREDICTED_RESTORATION_BRIER',s[s.source=='full_input_prediction']),
                        ('MEASURED_CONCEPT_SUPPLEMENTATION',s[s.source=='measured'])]:
            df.to_csv(out/f'{name}_{split}.csv',index=False)
        (out/f'SUMMARY_CONCEPT_DIAGNOSTICS_{split}.md').write_text('\n'.join(lines))
    print(f'[diagnostic tables] {len(tasks)}/{len(cfg["tasks"])} tasks: {out}',flush=True)


SOURCES = (
    ('full_input_prediction', 'PREDICTED', 'Full-input predicted concepts',
     'PREDICTED_RESTORATION_BRIER'),
    ('measured', 'MEASURED', 'Measured concept targets',
     'MEASURED_CONCEPT_SUPPLEMENTATION'),
)
CONDITIONS = {
    'full_original': 'Original model / full inputs',
    'full_original_masked': 'Original model / masked inputs',
    'mixed_full': 'Mixed CBM / full inputs',
    'mixed_masked': 'Mixed CBM / masked inputs',
}


def _label(task):
    return TASK_LABELS.get(task, task)


def _number(value):
    return 'NA' if pd.isna(value) else f'{value:+.6f}'


def _read(path):
    return pd.read_csv(path, dtype={'concept_id': str}, float_precision='round_trip')


def _selection(frame):
    """Use the existing diagnostic decision without another ranking/filter step."""
    values = frame['auroc_selected']
    if values.dtype == bool:
        return values
    return values.astype(str).str.lower().eq('true')


def make_recovery_report(cfg):
    """Call after make_diagnostics_report; never fit models or recompute effects.

    The main exports include every single-ID all-participant row. The existing
    measured diagnostic export retains the separate observed-participant rows.
    """
    root = Path(cfg['output_root'])
    out = root/'tables'
    diagnostics = out/'concept_diagnostics'
    tasks = list(cfg['tasks'])
    bank = ('without Observation' if cfg.get('exclude_observation') else
            'with the four approved Observation surveys')
    for split in ('val', 'test'):
        performance_path = diagnostics/f'DOWNSTREAM_BRIER_{split}.csv'
        if not performance_path.exists():
            continue
        performance = _read(performance_path)
        performance = performance[performance.task.isin(tasks)]
        completed = [task for task in tasks if task in set(performance.task)]
        if not completed:
            continue
        recovery = {}
        for source, prefix, _, legacy in SOURCES:
            frame = _read(diagnostics/f'{legacy}_{split}.csv')
            frame = frame[(frame.task.isin(completed)) &
                          (frame.source == source) &
                          (frame.population == 'all_participants')].copy()
            # Keep configured task order and descending AUROC gain within task.
            frame['_task_order'] = frame.task.map({t: i for i, t in enumerate(tasks)})
            frame = frame.sort_values(
                ['_task_order', 'auroc_rank_within_split', 'concept_id'],
                kind='stable', na_position='last').drop(columns='_task_order')
            selected = frame[_selection(frame)]
            for suffix, data in [('ALL', frame), ('SELECTED', selected)]:
                data.to_csv(out/f'{prefix}_CONCEPT_RECOVERY_{suffix}_{split}.csv', index=False)
            recovery[source] = frame

        lines = [f'# Single-concept recovery — {split}', '',
                 f'Concept bank: **{bank}**. Completed tasks: {len(completed)}/{len(tasks)}. '
                 f'Seed {cfg["seed"]}; {cfg["bootstrap_samples"]:,} paired, outcome-stratified bootstrap resamples.', '',
                 'Each Concept ID is restored separately in the saved Mixed CBM, starting from the same '
                 'masked-input baseline. Masking removes nonbasic clinical inputs; Basic Clinical and '
                 'the other eight modalities remain available. Models, normalization and decision thresholds '
                 'are reused without retraining or threshold optimization.', '',
                 '**Predicted recovery** replaces all channels of one ID with the same participant’s '
                 'full-input predicted concepts. **Measured recovery** replaces observed, encodable, learnable '
                 'items of that ID with their recorded targets using the saved training encoding; '
                 'missing items and unseen categories retain masked predictions.', '',
                 f'Within each task and intervention, select **{split} ΔAUROC > μ + 2σ** among individual IDs '
                 'evaluated on all participants, with population SD (ddof=0). Predicted recovery includes '
                 'constant IDs with zero effect; measured recovery requires usable measured targets. '
                 'Joint/all-ID recovery is excluded. There is no top-10 padding or extra positivity filter. '
                 'This is a descriptive outlier rule, not a significance test. Test selection is exploratory; '
                 'the confidence intervals do not adjust for selection on test or training-seed variability.', '',
                 '## Baseline performance', '',
                 'All entries are point estimates [95% CI]. The original model uses its saved threshold; '
                 'all Mixed CBM conditions use the saved masked-validation threshold.', '']
        performance_rows = []
        for task in completed:
            for condition, title in CONDITIONS.items():
                for row in performance[(performance.task == task) &
                                       (performance.condition == condition)].to_dict('records'):
                    performance_rows.append([_label(task), title, int(row['n']),
                                             *[ci(row, metric) for metric in ('auroc', 'f1', 'ba', 'brier')]])
        lines += [md(['Task', 'Model / input', 'N', 'AUROC', 'F1', 'BA', 'Brier'], performance_rows),
                  '## Selection cutoffs', '',
                  'Statistics and flags below are copied from the diagnostic exports. '
                  'Each intervention has its own distribution; observed-only cohorts do not enter these rankings.', '']
        selection_rows = []
        for task in completed:
            for source, _, title, _ in SOURCES:
                group = recovery[source][recovery[source].task == task]
                if group.empty:
                    continue
                first = group.iloc[0]
                selection_rows.append([_label(task), title, int(first.auroc_selection_n),
                                       _number(first.auroc_selection_mean),
                                       _number(first.auroc_selection_std),
                                       _number(first.auroc_selection_cutoff), int(_selection(group).sum())])
        lines += [md(['Task', 'Intervention', 'Evaluated IDs', 'μ', 'σ', 'μ + 2σ', 'Selected IDs'],
                     selection_rows),
                  '## Selected single IDs', '',
                  'Effects use all participants, including those whose predictions remain unchanged because '
                  'measured targets are unavailable. “Labels any / complete / total” counts participants with '
                  'at least one / all learnable items available, followed by cohort size; this availability '
                  'does not limit predicted recovery. AUROC after recovery is absolute. ΔAUROC, ΔF1 and ΔBA '
                  'are recovery minus masked baseline; Brier improvement is baseline minus recovery. '
                  'Positive changes indicate improvement. All metric entries include 95% CI.', '']
        for task in completed:
            lines += [f'### {_label(task)}', '']
            for source, _, title, _ in SOURCES:
                group = recovery[source][recovery[source].task == task]
                selected = group[_selection(group)]
                lines += [f'**{title}**', '']
                if selected.empty:
                    lines += [f'No single ID exceeded the {split} μ + 2σ cutoff.', '']
                    continue
                rows = []
                for row in selected.to_dict('records'):
                    rows.append([int(row['auroc_rank_within_split']), row['concept_id'],
                                 row['concept_name'],
                                 f'{int(row["n_eligible"])} / {int(row["n_complete"])} / {int(row["n_total"])}',
                                 ci(row, 'restored_auroc'),
                                 *[ci(row, metric+'_gain', True) for metric in ('auroc', 'f1', 'ba', 'brier')]])
                lines += [md(['Rank', 'Concept ID', 'Concept', 'Labels any / complete / total',
                              'AUROC after recovery', 'ΔAUROC', 'ΔF1', 'ΔBA', 'Brier improvement'], rows)]
        lines += ['## Exported results', '',
                  'ALL files retain every single-ID all-participant row, including unavailable IDs, '
                  'absolute metrics, paired effects, confidence intervals, availability, ranks and selection statistics. '
                  'SELECTED files contain exactly the flagged rows. Numerical values are copied from the diagnostic '
                  'exports; the Markdown display is rounded.', '']
        for _, prefix, title, _ in SOURCES:
            filenames = [f'{prefix}_CONCEPT_RECOVERY_{kind}_{split}.csv' for kind in ('ALL', 'SELECTED')]
            lines += [f'- {title}: ' + '; '.join(f'[{name}]({name})' for name in filenames) + '.']
        lines += [f'- Supplementary measured observed-participant effects: '
                  f'[MEASURED_CONCEPT_SUPPLEMENTATION_{split}.csv]'
                  f'(concept_diagnostics/MEASURED_CONCEPT_SUPPLEMENTATION_{split}.csv). '
                  'Observed cohorts may differ between IDs; their effects are not used for main selection.',
                  f'- Full diagnostics, including concept prediction quality and input-removal comparisons: '
                  f'[SUMMARY_CONCEPT_DIAGNOSTICS_{split}.md]'
                  f'(concept_diagnostics/SUMMARY_CONCEPT_DIAGNOSTICS_{split}.md).',
                  '- Run settings: [runtime.yaml](../runtime.yaml).', '']
        (out/f'SUMMARY_CONCEPT_RECOVERY_{split}.md').write_text('\n'.join(lines))
    print(f'[recovery tables] {out}', flush=True)
