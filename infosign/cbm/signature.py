"""Validation-selected signature-only inference and mixed CBM retraining.

No raw data or encoder passes. Source models/diagnostics are read-only.
Run ``python -m infosign.cbm.signature --help`` from the repository root.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml

from infosign.common import TASK_LABELS, backbone_dir, read_json, write_json, save_torch
from infosign.train import load
from infosign.stats import PairedF1, auroc, auroc_selection, threshold
from infosign.cbm.model import concept_id_groups, fit, predict, restore, restoration_batch
from infosign.cbm.diagnostics import aligned_labels, concept_quality, metric_draws, interval_fields

REFERENCE = Path(__file__).with_name('signature_reference.json')
VIEWS = {'full': 'embedding', 'masked': 'masked_embedding'}
TRAIN_KEYS = ('seed', 'concept_epochs', 'head_epochs', 'joint_epochs', 'concept_patience',
              'cbm_learning_rate', 'cbm_weight_decay', 'concept_lambda', 'concept_tolerance')


def signature_name(sd):
    return f'signature_{sd:g}sd'


def source_config(task, cfg, roots):
    source = next((Path(r).resolve() for r in roots
                   if (Path(r)/task/'mixed_cbm/model.pt').is_file()), None)
    if source is None:
        raise FileNotFoundError(f'{task}: no mixed_cbm/model.pt in {roots}')
    settings = yaml.safe_load((source/'runtime.yaml').read_text())
    original = {**settings.get('study', settings), 'output_root': str(source)}
    original.update(original.get('task_settings', {}).get(task, {}))
    # Use the source experiment's exact CBM optimization settings by default.
    run_cfg = {**cfg, **{k: original[k] for k in TRAIN_KEYS if k in original}}
    run_cfg['cbm_checkpoint_metric'] = original.get('cbm_checkpoint_metric', 'f1')
    if cfg.get('signature_seed') is not None:
        run_cfg['seed'] = cfg['signature_seed']
    return source, original, run_cfg


@torch.no_grad()
def raw_outputs(model, embedding):
    x = (np.asarray(embedding)-np.asarray(model.spec['x_mean']))/np.asarray(model.spec['x_std'])
    return model.concept_layer(torch.as_tensor(x, dtype=torch.float32, device=model.active.device))


@torch.no_grad()
def signature_only(model, raw, reference, concept_ids):
    """Arbitrary ID list; neutralize other channels BEFORE categorical softmax."""
    groups = concept_id_groups(model.spec['items'])
    columns = [c for cid in concept_ids for c in groups[str(cid)]['columns']]
    changed = reference.expand_as(raw).clone()
    changed[:, columns] = raw[:, columns]
    z = model.raw_to_bottleneck(changed)
    return z, model.task_head(z).squeeze(-1).sigmoid().cpu().numpy()


@torch.no_grad()
def select_signature(task, model, val, sd, reference):
    groups = concept_id_groups(model.spec['items'])
    full, _ = predict(model, val['embedding'])
    masked, baseline = predict(model, val['masked_embedding'])
    columns = [g['columns'] for g in groups.values()]
    p = np.concatenate([restoration_batch(model, masked, full, columns[s:s+32])
                        for s in range(0, len(columns), 32)])
    gains = auroc(val['y'], p)-auroc(val['y'], baseline)
    if not np.isfinite(gains).all():
        raise ValueError(f'{task}: validation restoration AUROC is undefined')
    selected = auroc_selection(gains, sd=sd)
    old = auroc_selection(gains)  # The original default remains exactly μ + 2σ.
    old_ids = {cid for cid, flag in zip(groups, old['auroc_selected']) if flag}
    expected = reference['tasks'][task]
    if old_ids != set(expected['selected_ids']) or len(groups) != expected['eligible_ids']:
        raise ValueError(f'{task}: STOP: validation 2-SD reference mismatch. '
                         f'Computed {sorted(old_ids)} from {len(groups)} IDs; '
                         f'expected {expected}. No signature experiment was run.')
    chosen = [cid for cid, flag in zip(groups, selected['auroc_selected']) if flag]
    if sd <= 2 and not old_ids.issubset(chosen):
        raise AssertionError('2-SD IDs must be a subset of the less restrictive signature')
    rows = [dict(task=task, concept_id=cid, concept_name=g['concept_name'],
                 val_auroc_gain=float(gain), val_mean=selected['auroc_selection_mean'][i],
                 val_sd=selected['auroc_selection_std'][i], val_cutoff=selected['auroc_selection_cutoff'][i],
                 val_rank=selected['auroc_rank_within_split'][i], selected=bool(selected['auroc_selected'][i]),
                 selected_2sd=bool(old['auroc_selected'][i]), learnable=g['learnable'],
                 selection_split='val', population='all_concept_ids_including_constants', sd_multiplier=sd)
            for i, ((cid, g), gain) in enumerate(zip(groups.items(), gains))]
    return pd.DataFrame(rows).sort_values(['val_rank', 'concept_id']), chosen


def train_signature(train, val, labels, schema, cfg, device, concept_ids, seed):
    """The existing three-stage mixed training, restricted to arbitrary IDs."""
    selected = set(map(str, concept_ids))
    filtered = {**schema, 'concepts': [c for c in schema['concepts'] if str(c['concept_id']) in selected],
                'inventory_ids': len(selected)}
    if {str(c['concept_id']) for c in filtered['concepts']} != selected:
        raise ValueError('Signature IDs and source label schema disagree')
    artifact, history = fit(train, {**val, 'embedding': val['masked_embedding']}, labels,
                            filtered, {**cfg, 'seed': seed}, device, masked_train=train['masked_embedding'],
                            val_full=val['embedding'])
    return artifact, history


def check_signature(model, raw, reference, ids):
    groups = concept_id_groups(model.spec['items'])
    original = model.raw_to_bottleneck(raw)
    _, all_p = signature_only(model, raw, reference, list(groups))
    p = model.task_head(original).squeeze(-1).sigmoid().detach().cpu().numpy()
    error = float(np.max(np.abs(p-all_p)))
    if error >= 1e-6:
        raise AssertionError(f'All-ID identity failed: {error}')
    z, _ = signature_only(model, raw, reference, ids)
    cols = [c for cid in ids for c in groups[cid]['columns']]
    if not torch.allclose(z[:, cols], original[:, cols], atol=1e-7, rtol=0):
        raise AssertionError('Selected channels changed')
    for item in model.spec['items']:
        if item['type'] == 'categorical':
            # No-training-label items are explicitly zero in the existing model.
            total = z[:, item['start']:item['stop']].sum(1)
            expected = 1. if item['categories'] else 0.
            if not torch.allclose(total, torch.full_like(total, expected), atol=1e-6, rtol=0):
                raise AssertionError('Categorical item normalization changed')
    return dict(all_ids_max_probability_difference=error, selected_channels='pass',
                categorical_sums='pass (unobserved items remain zero)')


def original_conditions(task, source_cfg, cache):
    full = backbone_dir(task, source_cfg)/'full'
    cut = read_json(full/'TRAINING_DONE.json')['threshold']
    return {f'original_full_{view}': (cache[key].sigmoid().numpy(), cut)
            for view, key in (('full', 'original_logits'), ('masked', 'masked_logits'))}


def basic_condition(task, source_cfg, cache, split):
    root = backbone_dir(task, source_cfg)/'basic'
    if not (root/f'features_{split}.pt').is_file() or not (root/'TRAINING_DONE.json').is_file():
        return None, 'not_available: Basic Clinical cached predictions or threshold missing'
    basic = load(root/f'features_{split}.pt')
    ids, other = list(map(str, cache['person_ids'])), list(map(str, basic['person_ids']))
    if len(set(other)) != len(other) or set(ids) != set(other):
        return None, 'not_comparable: Basic Clinical participant sets differ'
    lookup = {pid: i for i, pid in enumerate(other)}
    ix = [lookup[pid] for pid in ids]
    if not np.array_equal(np.asarray(cache['y']), np.asarray(basic['y'])[ix]):
        return None, 'not_comparable: Basic Clinical outcome labels differ'
    return (basic['original_logits'].sigmoid().numpy()[ix],
            read_json(root/'TRAINING_DONE.json')['threshold']), 'included: aligned by person_id'


def evaluate(task, cache, conditions, comparisons, cfg, device, split):
    boot = PairedF1(cache['y'], cfg['bootstrap_samples'], cfg.get('bootstrap_seed', cfg['seed']), device)
    draws, performance, effects = {}, [], []
    predictions = pd.DataFrame({'person_id': cache['person_ids'], 'y': np.asarray(cache['y'], int)})
    for name, (p, cut) in conditions.items():
        predictions[name] = p
        predictions[name+'_decision'] = (p >= cut).astype(int)
        draws[name] = metric_draws(boot, p, cut)
        row = dict(task=task, split=split, condition=name, threshold=cut, n=len(p))
        for metric, d in draws[name].items():
            row.update(interval_fields(metric, boot.intervals(d)[0]))
        performance.append(row)
    for name, baseline in comparisons:
        row = dict(task=task, split=split, condition=name, reference=baseline,
                   difference='condition minus reference (negative Brier difference is better)')
        for metric in draws[name]:
            row.update(interval_fields(metric+'_difference', boot.intervals(draws[name][metric]-draws[baseline][metric])[0]))
        effects.append(row)
    return pd.DataFrame(performance), pd.DataFrame(effects), predictions


def experiment(task, which, out, model, artifact, ids, train, val, backbone, source,
               source_cfg, cfg, device, retrain_root):
    folder = out/which
    if (folder/'DONE.json').exists():
        print(f'[reuse] {task}/{which}', flush=True)
        return
    folder.mkdir(parents=True, exist_ok=True)
    checks = {}
    if which == 'exp1':
        if len(set(map(str, train['person_ids']))) != len(train['person_ids']):
            raise ValueError('Reference means require unique, non-oversampled training participants')
        references = {v: raw_outputs(model, train[key]).mean(0) for v, key in VIEWS.items()}
        save_torch(folder/'raw_training_means.pt', {v: x.cpu() for v, x in references.items()})
        for view, key in VIEWS.items():
            checks[view] = check_signature(model, raw_outputs(model, val[key]), references[view], ids)
        _, vp = signature_only(model, raw_outputs(model, val['masked_embedding']), references['masked'], ids)
        cut = threshold(val['y'], vp)
    else:
        if not ids:
            write_json(folder/'DONE.json', dict(status='not_run_empty_signature', selected_ids=[]))
            return
        labels = pd.read_csv(source/'labels'/task/'labels.csv', dtype=str, keep_default_na=False)
        schema = read_json(source/'labels'/task/'schema.json')
        retrain_root.mkdir(parents=True, exist_ok=True)
        if (retrain_root/'DONE.json').exists():
            if read_json(retrain_root/'DONE.json').get('checkpoint_metric','f1') != cfg.get('cbm_checkpoint_metric','f1'):
                raise ValueError(f'{retrain_root}: checkpoint metric changed; use a new output directory')
            fitted = load(retrain_root/'model.pt')
        else:
            fitted, history = train_signature(train, val, labels, schema, cfg, device, ids, cfg['seed'])
            save_torch(retrain_root/'model.pt', fitted)
            pd.DataFrame(history).to_csv(retrain_root/'training.csv', index=False)
            write_json(retrain_root/'concept_definitions.json', fitted['spec']['items'])
            write_json(retrain_root/'DONE.json', dict(selected_ids=ids, seed=cfg['seed'], threshold=fitted['threshold'],
                       selection=fitted['selection'], checkpoint_metric=fitted.get('checkpoint_metric','f1'),
                       training_views=['full', 'masked']))
        signature = restore(fitted, device)
        expected_width = sum(len(concept_id_groups(model.spec['items'])[cid]['columns']) for cid in ids)
        if set(concept_id_groups(fitted['spec']['items'])) != set(ids) or fitted['spec']['width'] != expected_width:
            raise AssertionError('Retrained signature spec/width differs from selected channels')
        checks['retained_ids_and_width'] = 'pass'
        cut = fitted['threshold']
        frame = labels.set_index('person_id')
        training_frame = aligned_labels(frame, train['person_ids'])
    quality, availability = [], {}
    # No test cache is read until the signature, model, and threshold are fixed.
    for split in ('val', 'test'):
        cache = val if split == 'val' else load(backbone/f'features_{split}.pt')
        conditions, comparisons, zs = {}, [], {}
        for view, key in VIEWS.items():
            _, p = predict(model, cache[key])
            conditions['mixed_'+view] = (p, artifact['threshold'])
            if which == 'exp1':
                raw = raw_outputs(model, cache[key])
                checks[split+'_'+view] = check_signature(model, raw, references[view], ids)
                _, sp = signature_only(model, raw, references[view], ids)
                name = 'signature_only_'+view
            else:
                z, sp = predict(signature, cache[key]); zs[view] = z.cpu().numpy()
                name = 'signature_cbm_'+view
            conditions[name] = (sp, cut)
            comparisons.append((name, 'mixed_'+view))
        if which == 'exp2':
            conditions.update(original_conditions(task, source_cfg, cache))
            comparisons.extend(('signature_cbm_'+v, 'original_full_'+v) for v in VIEWS)
            basic, availability[split] = basic_condition(task, source_cfg, cache, split)
            if basic is not None:
                conditions['basic_clinical'] = basic
                comparisons.append(('signature_cbm_masked', 'basic_clinical'))
            quality += [dict(task=task, split=split, **row) for row in concept_quality(
                signature.spec['items'], aligned_labels(frame, cache['person_ids']), training_frame, zs['full'], zs['masked'])]
        performance, effects, predictions = evaluate(task, cache, conditions, comparisons, cfg, device, split)
        performance.to_csv(folder/f'performance_{split}.csv', index=False)
        effects.to_csv(folder/f'comparisons_{split}.csv', index=False)
        predictions.to_csv(folder/f'predictions_{split}.csv', index=False)
    if quality:
        pd.DataFrame(quality).to_csv(folder/'concept_quality.csv', index=False)
    write_json(folder/'DONE.json', dict(status='complete', threshold=cut, threshold_source='signature masked validation F1',
               checks=checks, basic_comparator=availability, selected_ids=ids, seed=cfg['seed'],
               bootstrap_seed=cfg.get('bootstrap_seed', cfg['seed']), bootstrap_samples=cfg['bootstrap_samples']))


def run_task(task, cfg, roots, output, sd, stage, reference, device):
    source, source_cfg, cfg = source_config(task, cfg, roots)
    out = output/signature_name(sd)/task
    out.mkdir(parents=True, exist_ok=True)
    backbone = backbone_dir(task, source_cfg)/'full'
    contract = dict(source=str(source), backbone=str(backbone), sd=sd,
                    optimization={k: cfg[k] for k in TRAIN_KEYS}, bootstrap_samples=cfg['bootstrap_samples'],
                    reference=reference['tasks'][task])
    if cfg.get('cbm_checkpoint_metric', 'f1') != 'f1':
        contract['optimization']['cbm_checkpoint_metric'] = cfg['cbm_checkpoint_metric']
    if (out/'RUN.json').exists():
        saved = read_json(out/'RUN.json')
        saved.pop('source_hashes', None); saved.pop('label_files', None)
        if saved != contract:
            raise ValueError(f'{out}: settings changed; use a new --output-root')
    write_json(out/'RUN.json', contract)
    torch.set_float32_matmul_precision('highest')
    artifact = load(source/task/'mixed_cbm/model.pt'); model = restore(artifact, device)
    val = load(backbone/'features_val.pt')
    if not (out/'SELECTION_DONE.json').exists():
        table, ids = select_signature(task, model, val, sd, reference)
        table.to_csv(out/'selected_concepts.csv', index=False)
        write_json(out/'SELECTION_DONE.json', dict(selected_ids=ids, reference_2sd='pass',
                   subset_2sd='pass' if sd <= 2 else 'not_applicable', selection_split='val'))
    ids = read_json(out/'SELECTION_DONE.json')['selected_ids']
    print(f'[signature {task}] {len(ids)} IDs; 2-SD reference matched', flush=True)
    if stage != 'select':
        train = load(backbone/'features_train.pt')
        for which in ('exp1', 'exp2') if stage == 'all' else (stage,):
            experiment(task, which, out, model, artifact, ids, train, val, backbone, source, source_cfg,
                       cfg, device, output/task/('mixed_cbm_'+signature_name(sd)))


def aggregate(output, sd, tasks):
    root = output/signature_name(sd); root.mkdir(parents=True, exist_ok=True)
    for destination, relative in {
        'selected_concepts.csv': 'selected_concepts.csv',
        **{f'{e}_{kind}_test.csv': f'{e}/{kind}_test.csv' for e in ('exp1', 'exp2') for kind in ('performance', 'comparisons')},
        'exp2_concept_quality.csv': 'exp2/concept_quality.csv',
    }.items():
        marker = 'SELECTION_DONE.json' if relative == 'selected_concepts.csv' else str(Path(relative).parent/'DONE.json')
        frames = [pd.read_csv(root/t/relative, dtype={'concept_id': str}) for t in tasks
                  if (root/t/marker).exists() and (root/t/relative).exists()]
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(root/destination, index=False)
    lines = ['# Signature-only experiments — test', '',
             f'Selection: validation ΔAUROC > μ + {sd:g}σ; population SD; predicted restoration; all IDs including constants.',
             'AUROC is primary. Paired outcome-stratified bootstrap 95% CIs use the configured seed and resample count.',
             'Each condition retains its own masked-validation threshold, shared across its full/masked views. '
             'Brier differences are condition minus reference (negative is better). '
             'Intervals condition on the fitted models and selected signature; they exclude training and selection uncertainty. '
             'Validation-based selection does not erase earlier exploratory use of the test set.', '',
             '| Task | IDs | Mixed full | Mixed masked | Signature-only full | Signature-only masked | Retrained full | Retrained masked |',
             '|---|---:|---|---|---|---|---|---|']
    for task in tasks:
        folder = root/task
        if not (folder/'SELECTION_DONE.json').exists():
            continue
        rows = {}
        for e in ('exp1', 'exp2'):
            p = folder/e/'performance_test.csv'
            if p.exists() and (folder/e/'DONE.json').exists():
                rows.update({r['condition']: r for r in pd.read_csv(p).to_dict('records')})
        cells = []
        for name in ('mixed_full', 'mixed_masked', 'signature_only_full', 'signature_only_masked', 'signature_cbm_full', 'signature_cbm_masked'):
            r = rows.get(name)
            cells.append(f'{r["auroc"]:.4f} [{r["auroc_ci_lower"]:.4f}, {r["auroc_ci_upper"]:.4f}]' if r else 'Not run')
        n = len(read_json(folder/'SELECTION_DONE.json')['selected_ids'])
        lines.append('| '+' | '.join([TASK_LABELS.get(task, task), str(n), *cells])+' |')
    lines += ['', '## Checks and optional comparators', '']
    for task in tasks:
        folder = root/task
        for filename in ('SELECTION_DONE.json', 'exp1/DONE.json', 'exp2/DONE.json'):
            if (folder/filename).exists():
                lines.append(f'- {task} / {filename}: `{read_json(folder/filename)}`')
    (root/'SUMMARY_SIGNATURE_test.md').write_text('\n'.join(lines)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=Path('config.yaml'))
    p.add_argument('--cbm-roots', type=Path, nargs='+', help='First matching completed CBM root per task')
    p.add_argument('--output-root', type=Path, help='New outputs only; defaults to configured study output')
    p.add_argument('--sd', type=float, default=1.)
    p.add_argument('--stage', choices=('select', 'exp1', 'exp2', 'all'), default='all')
    p.add_argument('--tasks', nargs='+')
    p.add_argument('--devices', nargs='+', default=['cuda:0'], help='e.g. 0 1 2 3, or cpu')
    p.add_argument('--seed', type=int, help='Override signature training/bootstrap seed')
    p.add_argument('--bootstrap-samples', type=int)
    p.add_argument('--reference-selection', type=Path, default=REFERENCE,
                   help='Validation 2-SD reference IDs and eligible counts; mismatch stops execution')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    a = p.parse_args()
    if not np.isfinite(a.sd) or a.sd < 0:
        p.error('--sd must be finite and nonnegative')
    settings = yaml.safe_load(a.config.read_text()); cfg = settings.get('study', settings).copy()
    if a.bootstrap_samples is not None: cfg['bootstrap_samples'] = a.bootstrap_samples
    cfg['signature_seed'] = a.seed
    tasks = a.tasks or cfg['tasks']
    roots = a.cbm_roots or [Path(cfg['output_root'])]
    output = (a.output_root or Path(cfg['output_root'])).resolve()
    reference = read_json(a.reference_selection)
    if a.worker:
        torch.set_num_threads(cfg.get('cpu_threads', 2))
        run_task(tasks[0], cfg, roots, output, a.sd, a.stage, reference, torch.device(a.devices[0]))
        return
    # Complete selection for every task before any signature training/evaluation.
    phases = ['select'] + ([] if a.stage == 'select' else [a.stage])
    for phase in phases:
        running, waiting, free = {}, list(tasks), list(a.devices)
        try:
            while waiting or running:
                while waiting and free:
                    task, gpu = waiting.pop(0), free.pop(0)
                    env = os.environ.copy()
                    device = 'cpu' if gpu == 'cpu' else 'cuda:0'
                    if device != 'cpu':
                        index = int(gpu.removeprefix('cuda:'))
                        visible = env.get('CUDA_VISIBLE_DEVICES')
                        env['CUDA_VISIBLE_DEVICES'] = visible.split(',')[index] if visible else str(index)
                    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
                        env[key] = str(cfg.get('cpu_threads', 2))
                    cmd = [sys.executable, '-m', 'infosign.cbm.signature', *sys.argv[1:],
                           '--worker', '--stage', phase, '--tasks', task, '--devices', device]
                    running[subprocess.Popen(cmd, env=env)] = (task, gpu)
                    print(f'[start {phase}] {task} on {gpu}', flush=True)
                for job, (task, gpu) in list(running.items()):
                    code = job.poll()
                    if code is not None:
                        if code: raise RuntimeError(f'{task}/{phase} failed; inspect its error above')
                        del running[job]; free.append(gpu)
                if running: time.sleep(.2)
        finally:
            for job in running:
                if job.poll() is None: job.terminate(); job.wait()
            aggregate(output, a.sd, tasks)
    print('SIGNATURE_COMPLETE: '+str(output/signature_name(a.sd)), flush=True)


if __name__ == '__main__':
    main()
