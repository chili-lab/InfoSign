"""Select Concept IDs on validation, then evaluate single and joint recovery.

This analysis reads frozen Mixed CBMs and cached representations. It never fits
a model, changes a threshold, or sends concepts back through the input encoder.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from infosign.common import TASK_LABELS, backbone_dir, write_json
from infosign.train import load
from infosign.stats import PairedF1, auroc, auroc_selection
from infosign.report import md
from infosign.cbm.model import concept_id_groups, predict, restore, restoration_batch
from infosign.cbm.diagnostics import aligned_labels, measured_targets, metric_draws, interval_fields


SOURCES = {'full_input_prediction': 'Predicted recovery', 'measured': 'Measured recovery'}
CONDITIONS = {
    'full': 'Full Clinical inputs',
    'masked': 'Basic Clinical inputs',
    'predicted_joint': 'Basic inputs + validation-selected predicted concepts',
    'measured_joint': 'Basic inputs + validation-selected measured concepts',
    'predicted_all': 'Basic inputs + all full-input predicted concepts',
    'measured_all': 'Basic inputs + all available measured concepts',
}
METRICS = ('auroc', 'f1', 'ba', 'brier')


def select_ids(y, masked_probability, probabilities, groups, availability, sd=2.0):
    """Called on validation only; keep the original per-source eligibility rule."""
    rows = []
    baseline = auroc(y, masked_probability)
    for source in SOURCES:
        gains = auroc(y, probabilities[source]) - baseline
        eligible = np.ones(len(groups), bool) if source == 'full_input_prediction' else np.array(
            [g['learnable'] and availability[cid].any() for cid, g in groups.items()])
        selection = auroc_selection(gains, eligible, sd=sd)
        for i, (cid, g) in enumerate(groups.items()):
            rows.append(dict(
                source=source, concept_id=cid, concept_name=g['concept_name'],
                source_fields=' | '.join(g['source_fields']), channels=len(g['columns']),
                learnable=g['learnable'], selection_split='val', val_n=len(y),
                val_eligible=bool(eligible[i]), val_labels_available=int(availability[cid].sum()),
                val_baseline_auroc=baseline, val_restored_auroc=baseline+gains[i],
                val_auroc_gain=gains[i], val_rank=selection['auroc_rank_within_split'][i],
                selection_n=selection['auroc_selection_n'][i],
                selection_mean=selection['auroc_selection_mean'][i],
                selection_std=selection['auroc_selection_std'][i],
                selection_cutoff=selection['auroc_selection_cutoff'][i],
                sd_multiplier=sd, selected=bool(selection['auroc_selected'][i])))
    return pd.DataFrame(rows).sort_values(['source', 'val_rank', 'concept_id'], kind='stable')


def recover(model, masked, reference, columns):
    # An empty selection is exactly the masked baseline; do not pad with top IDs.
    if not columns:
        return model.task_head(masked).squeeze(-1).sigmoid().cpu().numpy()
    return restoration_batch(model, masked, reference, [columns])[0]


def statistics(y, probabilities, cut, cfg, device):
    """All conditions share draws. Rows 0 and 1 are full and masked references."""
    boot = PairedF1(y, cfg['bootstrap_samples'], cfg['seed'], device)
    rows = [dict(n=len(y), positive=int(np.sum(y)), negative=int((y == 0).sum()), threshold=cut)
            for _ in probabilities]
    for metric, draws in metric_draws(boot, probabilities, cut).items():
        sign = -1 if metric == 'brier' else 1
        absolute = boot.intervals(draws)
        gain = boot.intervals(sign * (draws - draws[1]))
        gap = boot.intervals(sign * (draws[0] - draws))
        for row, perf, delta, remaining in zip(rows, absolute, gain, gap):
            row.update(**interval_fields(metric, perf), **interval_fields(metric+'_gain', delta),
                       **interval_fields(metric+'_gap_to_full', remaining))
    before = probabilities[1] >= cut
    for row, p in zip(rows, probabilities):
        after = p >= cut
        row.update(changed=int(np.sum(before != after)),
                   corrected=int(np.sum((before != y) & (after == y))),
                   newly_wrong=int(np.sum((before == y) & (after != y))))
    return rows


@torch.no_grad()
def run_validation_recovery(task, cfg, device):
    source = Path(cfg.get('concept_sources', {}).get(task, cfg['output_root']))
    source_cfg = cfg
    if cfg.get('concept_sources'):
        settings = yaml.safe_load((source/'runtime.yaml').read_text())
        source_cfg = {**settings.get('study', settings), 'output_root': str(source)}
    out = Path(cfg['output_root'])/'validation_selected'/task
    out.mkdir(parents=True, exist_ok=True)
    (out/'DONE.json').unlink(missing_ok=True)
    artifact = load(source/task/'mixed_cbm/model.pt')
    model = restore(artifact, device)
    torch.set_float32_matmul_precision('highest')
    cut = artifact['threshold']
    backbone = backbone_dir(task, source_cfg)/'full'
    frame = pd.read_csv(source/'labels'/task/'labels.csv', dtype=str,
                        keep_default_na=False).set_index('person_id')
    groups = concept_id_groups(model.spec['items'])
    item_indices = {cid: [j for j, c in enumerate(model.spec['items'])
                         if str(c['concept_id']) == cid and c['learnable']] for cid in groups}
    # Finish and save validation selection before loading any test representations.
    for split in ('val', 'test'):
        cache = load(backbone/f'features_{split}.pt')
        y = np.asarray(cache['y'], dtype=int)
        full, fp = predict(model, cache['embedding'])
        masked, mp = predict(model, cache['masked_embedding'])
        labels = aligned_labels(frame, cache['person_ids'])
        measured, available_items = measured_targets(model, labels, masked)
        references = {'full_input_prediction': full, 'measured': measured}
        available = {cid: available_items[:, js].any(1) for cid, js in item_indices.items()}
        if split == 'val':
            columns = [g['columns'] for g in groups.values()]
            individual = {name: np.concatenate([
                restoration_batch(model, masked, ref, columns[s:s+32])
                for s in range(0, len(columns), 32)]) for name, ref in references.items()}
            selection = select_ids(y, mp, individual, groups, available, sd=cfg.get('selection_sd', 2.0))
            selection.insert(0, 'task', task)
            selection['cbm_source_root'] = str(source)
            selection.to_csv(out/'CONCEPT_SELECTION_val.csv', index=False)
            selected = selection[selection.selected].to_dict('records')
            print(f'[validation selection {task}] ' + '; '.join(
                f'{label}: {sum(r["source"] == name for r in selected)} IDs'
                for name, label in SOURCES.items()), flush=True)

        probabilities = [fp, mp]
        records = [dict(kind='reference', condition=k, source='', concept_id='', selected_ids='',
                        n_selected_ids=0, n_channels=0, n_available=len(y)) for k in ('full', 'masked')]
        # Keep validation order, including negative or zero test effects.
        for row in selected:
            cid, name = row['concept_id'], row['source']
            probabilities.append(recover(model, masked, references[name], groups[cid]['columns']))
            records.append(dict(**row, kind='single', condition=name+'__'+cid, selected_ids=cid,
                                n_selected_ids=1, n_channels=len(groups[cid]['columns']),
                                n_available=len(y) if name == 'full_input_prediction' else int(available[cid].sum())))
        for name, ref in references.items():
            ids = [r['concept_id'] for r in selected if r['source'] == name]
            columns = sorted({col for cid in ids for col in groups[cid]['columns']})
            probabilities.append(recover(model, masked, ref, columns))
            js = [j for cid in ids for j in item_indices[cid]]
            any_label = available_items[:, js].any(1)
            complete = available_items[:, js].all(1) if js else np.zeros(len(y), bool)
            records.append(dict(kind='joint', condition='predicted_joint' if name == 'full_input_prediction' else 'measured_joint',
                                source=name, concept_id='', selected_ids=' | '.join(ids),
                                n_selected_ids=len(ids), n_channels=len(columns),
                                n_available=(len(y) if ids else 0) if name == 'full_input_prediction' else int(any_label.sum()),
                                n_measured_complete=int(complete.sum()),
                                status='evaluated' if ids else 'no_validation_selected_ids'))
        if cfg.get('restore_all_concepts'):
            for name, ref in references.items():
                columns = sorted({col for g in groups.values() for col in g['columns']})
                probabilities.append(recover(model, masked, ref, columns))
                records.append(dict(kind='all_concepts', condition='predicted_all' if name=='full_input_prediction' else 'measured_all',
                                    source=name, concept_id='', selected_ids=' | '.join(groups),
                                    n_selected_ids=len(groups), n_channels=len(columns),
                                    n_available=len(y) if name=='full_input_prediction' else int(available_items.any(1).sum()),
                                    status='evaluated'))
        probabilities = np.asarray(probabilities)
        stats = statistics(y, probabilities, cut, cfg, device)
        results = pd.DataFrame([{**r, **s, 'task': task, 'split': split, 'selection_split': 'val',
                                 'cbm_source_root': str(source), 'feature_directory': str(backbone)}
                                for r, s in zip(records, stats)])
        results.to_csv(out/f'RECOVERY_{split}.csv', index=False)
        predictions = pd.DataFrame(dict(person_id=cache['person_ids'], y=y))
        predictions = pd.concat([predictions, pd.DataFrame(probabilities.T,
                                   columns=[r['condition'] for r in records])], axis=1)
        predictions.to_csv(out/f'PREDICTIONS_{split}.csv', index=False)
        print(f'[validation-selected recovery {task}/{split}] single IDs and both joint sets complete', flush=True)
    write_json(out/'DONE.json', dict(task=task, selection_split='val', seed=cfg['seed'],
               bootstrap_samples=cfg['bootstrap_samples'], threshold=cut, training=False,
               cbm_source_root=str(source), feature_directory=str(backbone),
               selection_sd=cfg.get('selection_sd', 2.0), restore_all_concepts=bool(cfg.get('restore_all_concepts'))))


def ci(row, key, signed=False):
    fmt = '+.4f' if signed else '.4f'
    return f'{row[key]:{fmt}} [{row[key+"_ci_lower"]:{fmt}}, {row[key+"_ci_upper"]:{fmt}}]'


def make_validation_report(cfg):
    sd = cfg.get('selection_sd', 2.0)
    root = Path(cfg['output_root'])/'validation_selected'
    tasks = [t for t in cfg['tasks'] if (root/t/'DONE.json').exists()]
    if not tasks:
        return
    out = root/'tables'
    out.mkdir(parents=True, exist_ok=True)
    selection = pd.concat([pd.read_csv(root/t/'CONCEPT_SELECTION_val.csv', dtype={'concept_id': str})
                           for t in tasks], ignore_index=True)
    selection.to_csv(out/'CONCEPT_SELECTION_val.csv', index=False)
    for split in ('val', 'test'):
        results = pd.concat([pd.read_csv(root/t/f'RECOVERY_{split}.csv', dtype={'concept_id': str})
                             for t in tasks], ignore_index=True)
        single = results[results.kind == 'single']
        joint = results[results.kind != 'single']
        single.to_csv(out/f'SELECTED_CONCEPT_RECOVERY_{split}.csv', index=False)
        joint.to_csv(out/f'JOINT_CONCEPT_RECOVERY_{split}.csv', index=False)
        lines = [f'# Validation-selected concept recovery — {split}', '',
                 f'Completed tasks: {len(tasks)}/{len(cfg["tasks"])}. Seed {cfg["seed"]}; '
                 f'{cfg["bootstrap_samples"]:,} paired, outcome-stratified participant bootstrap resamples.', '',
                 'All conditions use the same frozen Mixed CBM and its saved masked-validation classification threshold. '
                 'Basic Clinical inputs retain basic clinical information and the other eight modalities; '
                 'only nonbasic clinical values and their availability indicators are masked.', '',
                 'For each task, predicted and measured interventions have separate candidate lists: '
                 f'**validation single-ID ΔAUROC > validation μ + {sd:g}σ**, using population SD (ddof=0). '
                 'Predicted selection includes constant IDs; measured selection requires at least one usable learnable '
                 'validation label. Test effects and test availability never change either list. '
                 'No top-k padding or additional positivity filter is used. This is a descriptive selection rule, '
                 'not a significance test. Validation results are selection-set descriptions. '
                 'Test results evaluate fixed validation lists, but do not erase prior exploratory use of this test set. '
                 'Intervals condition on the fitted model and selected lists; they exclude selection and training-seed uncertainty.', '',
                 'Predicted recovery uses the same participant’s full-input concept predictions as a diagnostic reference. '
                 'Measured recovery substitutes available, encodable, learnable recorded targets; missing or unseen '
                 'values retain masked predictions. Each joint condition simultaneously restores its own selected IDs '
                 'and reruns the CBM classification head. Single-ID effects are never added together.', '',
                 '## Validation selection', '']
        rows = []
        for task in tasks:
            for source, title in SOURCES.items():
                group = selection[(selection.task == task) & (selection.source == source)]
                first = group.iloc[0]
                ids = group[group.selected].concept_id.tolist()
                rows.append([TASK_LABELS[task], title, int(first.selection_n),
                             f'{first.selection_mean:+.6f}', f'{first.selection_std:.6f}',
                             f'{first.selection_cutoff:+.6f}', len(ids), ', '.join(ids) or 'None'])
        lines += [md(['Task', 'Intervention', 'Eligible IDs', 'Val μ', 'Val σ', f'Val μ + {sd:g}σ',
                      'Selected N', 'Selected IDs'], rows), '## Model sources', '',
                  md(['Task', 'CBM output root', 'Frozen feature directory'],
                     [[TASK_LABELS[r.task], r.cbm_source_root, r.feature_directory]
                      for r in joint[joint.condition == 'full'].itertuples()]),
                  '## Downstream performance', '']
        rows = [[TASK_LABELS[r['task']], CONDITIONS[r['condition']], int(r['n']),
                 int(r['n_selected_ids']), *[ci(r, m) for m in METRICS]] for r in joint.to_dict('records')]
        lines += [md(['Task', 'Condition', 'N', 'Restored IDs', 'AUROC [95% CI]', 'F1 [95% CI]',
                      'BA [95% CI]', 'Brier [95% CI]'], rows), '## Joint gain and remaining gap', '',
                  'Positive gain means improvement over Basic inputs. Positive gap means the restored condition '
                  'still performs worse than Full inputs. For AUROC/F1/BA, gain = restored − basic and '
                  'gap = full − restored; Brier uses the reverse signs because lower is better. '
                  'Negative gaps are retained. A CI containing zero does not establish equivalence or noninferiority.', '']
        rows = [[TASK_LABELS[r['task']], CONDITIONS[r['condition']], int(r['n_selected_ids']),
                 int(r['n_available']), *[ci(r, m+suffix, True) for m in METRICS
                                         for suffix in ('_gain', '_gap_to_full')]]
                for r in joint[joint.kind.isin(('joint','all_concepts'))].to_dict('records')]
        lines += [md(['Task', 'Joint recovery', 'IDs', 'N with any replacement',
                      *[f'{m.upper()} {what} [95% CI]' for m in METRICS for what in ('gain', 'gap to full')]], rows),
                  '## Selected individual IDs', '',
                  f'All values below are evaluated on **{split}**, with the original validation ranks. '
                  'Negative effects and IDs without usable test measurements remain reported.', '']
        for task in tasks:
            lines += [f'### {TASK_LABELS[task]}', '']
            for source, title in SOURCES.items():
                group = single[(single.task == task) & (single.source == source)]
                lines += [f'**{title}**', '']
                if group.empty:
                    lines += ['No ID exceeded the validation cutoff; joint recovery equals the Basic inputs baseline.', '']
                    continue
                rows = [[int(r['val_rank']), r['concept_id'], r['concept_name'], r['source_fields'],
                         f'{r["val_auroc_gain"]:+.6f}', f'{int(r["n_available"])}/{int(r["n"])}',
                         *[ci(r, m+'_gain', True) for m in METRICS]] for r in group.to_dict('records')]
                lines += [md(['Val rank', 'Concept ID', 'Concept', 'Source fields', 'Val ΔAUROC',
                              'N with replacement / total', 'ΔAUROC [95% CI]', 'ΔF1 [95% CI]',
                              'ΔBA [95% CI]', 'Brier improvement [95% CI]'], rows)]
        lines += ['## Files', '', '- `CONCEPT_SELECTION_val.csv`: every candidate, eligibility, validation gain, cutoff and selection flag.',
                  f'- `SELECTED_CONCEPT_RECOVERY_{split}.csv`: selected single-ID performance, gains and gaps.',
                  f'- `JOINT_CONCEPT_RECOVERY_{split}.csv`: Full, Basic and both joint-restoration conditions.',
                  f'- `../<task>/PREDICTIONS_{split}.csv`: paired participant-level probabilities.', '']
        (out/f'SUMMARY_VALIDATION_SELECTED_{split}.md').write_text('\n'.join(lines))
    print(f'[validation-selected report] {out}', flush=True)
