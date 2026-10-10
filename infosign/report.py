"""English F1 and balanced-accuracy tables for coauthors, with Mixed-only concept restoration."""
from pathlib import Path
import numpy as np
import pandas as pd
from infosign.common import TASK_LABELS, INPUT_LABELS, read_json
from infosign.stats import bh_adjust, auroc_selection

MODEL_NAMES = {'full':'Full Clinical model', 'full_basic_inference':'Full model: Basic Clinical at inference',
               'basic':'Basic Clinical model',
               'mixed_full':'Mixed CBM: full clinical inputs', 'mixed_basic':'Mixed CBM: basic clinical inputs'}
COMPARISON_NAMES = {'full_input_loss':'Full model: loss after removing nonbasic clinical inputs',
    'basic_vs_full':'Basic minus Full model', 'retraining_gain':'Basic retraining gain over Full model with basic-only clinical inputs',
    'mixed_cbm_gain':'Mixed CBM (basic inputs) minus Full model (basic inputs)',
    'mixed_input_loss':'Mixed CBM: loss after removing nonbasic clinical inputs'}


def md(headers, rows):
    def cell(value): return str(value).replace('|',' / ').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',
                      *['| '+' | '.join(cell(v) for v in row)+' |' for row in rows]])+'\n'


def interval(row, signed=False, prefix=''):
    fmt = '+.4f' if signed else '.4f'
    return f'{row[prefix+"value"]:{fmt}} [{row[prefix+"ci_lower"]:{fmt}}, {row[prefix+"ci_upper"]:{fmt}}]'


def prob(value): return f'{value:.4g}'


def input_label(source, model):
    if source=='clinical_data' and model=='full': return 'All clinical information'
    return INPUT_LABELS.get(source, 'Nonbasic clinical inputs')


def auroc_concept_table(rows, task, split):
    """Display every single ID strictly above its task/split mean + 2 SD."""
    ids = rows[(rows.task==task)&(rows.intervention=='individual')]
    z = ids[ids.auroc_selected].sort_values(['auroc_rank_within_split','concept_id'])
    lines = [f'#### {TASK_LABELS[task]} — nonbasic clinical inputs removed', '']
    if ids.empty:
        return lines + ['No evaluated Concept IDs.', '']
    first = ids.iloc[0]
    lines += [f'{int(first.auroc_selection_n)} evaluated individual IDs; '
              f'mean ΔAUROC = {first.auroc_selection_mean:+.6f}, '
              f'population SD = {first.auroc_selection_std:.6f}, '
              f'cutoff μ + 2σ = {first.auroc_selection_cutoff:+.6f}. '
              f'{len(z)} IDs exceed the cutoff. Masked CBM AUROC: {first.masked_auroc:.4f}.', '']
    if z.empty:
        return lines + ['No Concept ID exceeds μ + 2σ; no fixed Top10 is substituted.', '']
    return lines + [md([split.title()+' ΔAUROC rank','Concept ID','Concept (recorded terminology)',
            'Source fields restored together','Restored AUROC','ΔAUROC [95% CI]',
            'ΔF1 [95% CI]','ΔBA [95% CI]','Changed classifications (N)'],
        [[int(x.auroc_rank_within_split),x.concept_id,x.concept_name,x.source_fields,
          f'{x.restored_auroc:.4f}',interval(x._asdict(),True,'auroc_'),
          interval(x._asdict(),True),interval(x._asdict(),True,'ba_'),int(x.n_prediction_changes)]
         for x in z.itertuples()])]


def make_report(cfg):
    from infosign.analysis import ANALYSIS_VERSION
    root = Path(cfg['output_root']); out = root/'tables'; out.mkdir(exist_ok=True)
    completed = [t for t in cfg['tasks'] if (root/t/'analysis/DONE.json').exists()
                 and read_json(root/t/'analysis/DONE.json').get('analysis_version')==ANALYSIS_VERSION]
    if not completed: return
    frames = {}
    for name in ('classification','comparisons','ablation','restoration','cohorts'):
        df = pd.concat([pd.read_csv(root/t/'analysis'/(name+'.csv'), dtype={'concept_id':str}) for t in completed], ignore_index=True)
        if 'p_value' in df:
            for prefix in ('','ba_'):
                df[prefix+'q_value'] = np.nan
                for _, idx in df.groupby(['split','family']).groups.items():
                    df.loc[idx,prefix+'q_value'] = bh_adjust(df.loc[idx,prefix+'p_value'])
        if name=='restoration':
            single = df.intervention=='individual'
            df['auroc_selected'] = False
            for _, idx in df[single].groupby(['task','split','mode','removed_input']).groups.items():
                for column, values in auroc_selection(df.loc[idx,'auroc_value'].to_numpy()).items():
                    df.loc[idx,column] = values
            df['f1_rank'] = df['rank']
            df['f1_tied_concepts'] = df['tied_concepts']
            df['rank'] = df['auroc_rank_within_split']
            df.loc[single,'tied_concepts'] = df[single].groupby(
                ['task','split','mode','removed_input','auroc_value']).auroc_value.transform('size')
            df['ranking_metric'] = 'delta_auroc'
        frames[name] = df
        df.to_csv(out/(name+'.csv'), index=False)
    for split in ('val','test'):
        c,e,a,r,co = [frames[name].query('split == @split') for name in ('classification','comparisons','ablation','restoration','cohorts')]
        all_rows = r[r.intervention=='all_concepts']
        def cell(task, model):
            row = c[(c.task==task)&(c.model==model)].iloc[0]
            return 'AUROC: '+interval(row,prefix='auroc_')+'<br>F1: '+interval(row)+'<br>BA: '+interval(row, prefix='ba_')
        def restored_cell(row):
            return (f'AUROC: {row.restored_auroc:.4f} [{row.restored_auroc_ci_lower:.4f}, {row.restored_auroc_ci_upper:.4f}]'
                    f'<br>F1: {row.restored_f1:.4f} [{row.restored_ci_lower:.4f}, {row.restored_ci_upper:.4f}]'
                    f'<br>BA: {row.restored_ba:.4f} [{row.restored_ba_ci_lower:.4f}, {row.restored_ba_ci_upper:.4f}]')
        lines = [f'# OT2 results — {split}', '',
            f'Completed tasks: {len(completed)}/{len(cfg["tasks"])}. Seed {cfg["seed"]}. '
            f'{cfg["bootstrap_samples"]:,} participant bootstrap resamples. AUROC, positive-class F1 and balanced accuracy (BA).', '',
            'Full and Basic model modality ablations are separate from the CBM experiment. '
            'Every task uses the same Mixed-CBM removal: nonbasic clinical inputs are withheld, while Basic Clinical and the other eight modalities remain available.', '',
            'Checkpoint selection and F1 thresholds use validation only. Modality top-three markers are validation-selected descriptive summaries; they do not select CBM removal conditions. '
            'Individual Concept IDs are ranked by ΔAUROC within each task/split; displayed IDs exceed the mean + 2 population SD of all finite evaluated individual-ID gains. '
            'Test concept ranks are exploratory. '
            'BA uses the same frozen threshold. Validation and test retain their natural class proportions; only training uses balanced oversampling.', '']
        if cfg.get('exclude_observation'):
            lines += ['Concept bank: all Observation-domain concepts excluded. Measurement, Condition, Procedure and Device concepts retain task-specific outcome exclusions.', '']
        else:
            lines += ['Concept bank: Observation is restricted to the food insecurity, neighborhood environment, '
                'racial/ethnic discrimination and combined SDOH surveys. Each task retains its own outcome exclusions. '
                'Measurement, Device, Condition and Procedure concepts keep their previous task-specific eligibility. '
                'The form restriction applies only to Observation: eligible Condition concepts from health_survey remain available. '
                'Survey-date/performed, metadata, start-timestamp and completion-timestamp IDs '
                '4160030, 2005200102, 2005200558 and 2005200559 are excluded.', '']
        lines += ['## Evaluation cohorts', '',
            md(['Task','N','Positive','Negative','Prevalence','All-positive baseline F1 (no model)'],
               [[TASK_LABELS[x.task],x.n,x.positive,x.negative,f'{x.prevalence:.3f}',f'{x.all_positive_f1:.4f}'] for x in co.itertuples()]),
            'All-positive baseline BA is 0.5000 for every task.', '',
            '## 1. Downstream classification', '', 'Cells show AUROC, F1 and BA with 95% CIs.', '',
            md(['Task',*map(MODEL_NAMES.get,('full','full_basic_inference','basic'))],
               [[TASK_LABELS[t],*[cell(t,m) for m in ('full','full_basic_inference','basic')]] for t in completed]),
            '## 2. Modality importance', '',
            '**Drop = original model performance − performance after removal.** '
            'The same 11 branch/family removals are evaluated for both models. Removing a nonclinical branch leaves the clinical inputs intact; '
            'related clinical measurements are not removed in tandem. Removing the clinical branch removes all clinical inputs of that model, '
            'which differs from the partial nonbasic-clinical removal in the Mixed-CBM experiment. '
            'Bold cells mark each model/task’s validation-selected top three; neither a positive drop nor statistical significance is guaranteed.', '']
        for variant in ('full','basic'):
            subset = a[a.model==variant]
            matrix = []
            for source in INPUT_LABELS:
                row = [input_label(source, variant)]
                for t in completed:
                    x = subset[(subset.task==t)&(subset.removed_input==source)].iloc[0]
                    value = 'AUROC: '+interval(x,True,'auroc_')+'<br>F1: '+interval(x,True)+'<br>BA: '+interval(x,True,'ba_')
                    row.append('**'+value+'**' if pd.notna(x.val_rank) else value)
                matrix.append(row)
            lines += [f'### {MODEL_NAMES[variant]}', '',
                      md(['Removed input',*[TASK_LABELS[t] for t in completed]], matrix)]
            subset.to_csv(out/f'MODALITY_ABLATION_{variant}_{split}.csv', index=False)
        lines += ['## 3. Mixed CBM: structured clinical information', '',
            'One shared concept predictor and classifier are trained on paired full/basic-only views of a frozen Full Clinical backbone. '
            'The threshold is selected on masked validation inputs and used for all three conditions below. '
            'There is no participant-level access to withheld clinical data during masked inference.', '',
            'Concept restoration is a separate diagnostic substitution: selected channels are replaced with the same participant’s full-input predicted concepts, '
            'not measured labels. Simultaneously restoring all channels reproduces this CBM’s full-input predictions by construction, '
            'because the classifier has no bypass. It is a reference endpoint, not autonomous recovery at deployment.', '',
            '### Downstream performance before and after all-concept restoration', '',
            md(['Task','Full clinical inputs','Basic-only clinical inputs','Basic-only inputs + all concepts restored'],
               [[TASK_LABELS[t],cell(t,'mixed_full'),cell(t,'mixed_basic'),restored_cell(all_rows[all_rows.task==t].iloc[0])] for t in completed]),
            '### Recovery from restoring all Concept IDs simultaneously', '',
            md(['Task','AUROC recovery [95% CI]','F1 recovery [95% CI]','F1 p','F1 q','BA recovery [95% CI]','BA p','BA q'],
               [[TASK_LABELS[x.task],interval(x._asdict(),True,'auroc_'),interval(x._asdict(),True),prob(x.p_value),prob(x.q_value),
                 interval(x._asdict(),True,'ba_'),prob(x.ba_p_value),prob(x.ba_q_value)] for x in all_rows.itertuples()]),
            '### Individual Concept-ID restoration', '',
            '**Recovery = restored − masked performance.** Each ID is restored separately from the same masked baseline; '
            'all source fields sharing an ID move together. Concepts are not restored cumulatively. '
            'All IDs with ΔAUROC > μ + 2σ are displayed, with σ calculated using ddof=0 across all finite evaluated single-ID gains in this task/split. '
            'The all-concepts-restored endpoint is excluded from that distribution. This describes unusually large effects within this bank; '
            'it is not a significance test or a two-standard-error interval. Selection uses unrounded values. No minimum count or Top10 padding is imposed. '
            'Equal F1/BA need not imply equal probabilities. The CSV includes probability changes, classification-change counts, '
            'confusion counts and exact probability/decision equivalence groups within each task/split/removal.', '']
        for t in completed:
            lines += auroc_concept_table(r,t,split)
        lines += ['## CBM selected fits', '',
            md(['Task','Mixed CBM selected fit'],
               [[TASK_LABELS[t],read_json(root/t/'mixed_cbm/DONE.json')['selection']['joint']['status']] for t in completed])]
        lines += [
            '`fitted` denotes a selected joint-tuning epoch; `sequential_fallback` retains the concept-then-head model. '
            'Joint tuning is selected only when validation F1/BCE improves while satisfying the concept-loss tolerance.', '',
            '## Paired model comparisons', '',
            md(['Task','Comparison','F1 difference [95% CI]','F1 p','F1 q','BA difference [95% CI]','BA p','BA q'],
               [[TASK_LABELS[x.task],COMPARISON_NAMES[x.comparison],interval(x._asdict(),True),prob(x.p_value),prob(x.q_value),
                 interval(x._asdict(),True,'ba_'),prob(x.ba_p_value),prob(x.ba_q_value)] for x in e.itertuples()]),
            '## Modality-ablation statistics', '',
            md(['Task','Model','Removed input','F1 drop [95% CI]','F1 p','F1 q','BA drop [95% CI]','BA p','BA q'],
               [[TASK_LABELS[x.task],MODEL_NAMES[x.model],input_label(x.removed_input,x.model),interval(x._asdict(),True),prob(x.p_value),prob(x.q_value),
                 interval(x._asdict(),True,'ba_'),prob(x.ba_p_value),prob(x.ba_q_value)] for x in a.itertuples()]),
            '## Thresholds', '',
            md(['Task','Model / input condition','Validation-selected threshold'],
               [[TASK_LABELS[x.task],MODEL_NAMES[x.model],f'{x.threshold:.6f}'] for x in c.itertuples()]),
            '## Statistical interpretation and exports', '',
            'Intervals are paired, outcome-stratified percentile 95% CIs. Two-sided p values use participant-level prediction swaps. '
            'BH q values are computed across completed tasks within each split, metric and analysis family: downstream comparisons, CBM comparisons, '
            'Full-model ablations, Basic-model ablations, individual Mixed-concept restoration, and joint all-concept restoration. '
            'Every tested ID enters the F1/BA individual-restoration correction, not only displayed IDs. Final q values require all configured tasks. '
            'AUROC is reported with paired bootstrap CIs; AUROC permutation p/q values are not computed. '
            'Test ranks are exploratory; intervals do not adjust for selecting the largest test effects. '
            'A single seed does not measure retraining variability, and previously inspected test data are not a new confirmatory cohort.', '',
            f'CSV tables: `MODALITY_ABLATION_full_{split}.csv`, `MODALITY_ABLATION_basic_{split}.csv`, '
            f'`MIXED_CBM_PERFORMANCE_{split}.csv`, `CONCEPT_RESTORATION_ALL_{split}.csv`, `ALL_CONCEPTS_RESTORED_{split}.csv`. '
            'The last two contain only Mixed-CBM restoration under nonbasic-clinical removal. '
            'Individual probability_group/decision_group numbers identify exactly equal probability/decision vectors within a condition; '
            'they are diagnostic labels, not concept clusters. Source definitions and supervision remain in the task/CBM directories. '
            'Task outcome exclusions and the four previously excluded administrative IDs remain in effect. '
            'Participant-level predictions remain in the local result directory.', '']
        individual = r[r.intervention=='individual'].sort_values(['task','auroc_rank_within_split','concept_id'],na_position='last')
        individual.to_csv(out/f'CONCEPT_RESTORATION_ALL_{split}.csv',index=False)
        individual[individual.auroc_selected].to_csv(out/f'CONCEPT_RESTORATION_SELECTED_AUROC_{split}.csv',index=False)
        all_rows.to_csv(out/f'ALL_CONCEPTS_RESTORED_{split}.csv',index=False)
        mixed = c[c.model.isin(['mixed_full','mixed_basic'])].copy()
        additional = [dict(task=x.task,split=split,model='mixed_all_restored',threshold=x.threshold,n=x.n,
            value=x.restored_f1,ci_lower=x.restored_ci_lower,ci_upper=x.restored_ci_upper,
            auroc_value=x.restored_auroc,auroc_ci_lower=x.restored_auroc_ci_lower,auroc_ci_upper=x.restored_auroc_ci_upper,
            ba_value=x.restored_ba,ba_ci_lower=x.restored_ba_ci_lower,ba_ci_upper=x.restored_ba_ci_upper) for x in all_rows.itertuples()]
        pd.concat([mixed,pd.DataFrame(additional)],ignore_index=True).sort_values(['task','model']).to_csv(out/f'MIXED_CBM_PERFORMANCE_{split}.csv',index=False)
        (out/f'SUMMARY_{split}.md').write_text('\n'.join(lines))
    print(f'[tables] {len(completed)}/{len(cfg["tasks"])} tasks: {out}',flush=True)
