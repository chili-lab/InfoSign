"""Repeat signature CBM training seeds while holding the saved signature fixed."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd
import torch
import yaml

from infosign.common import TASK_LABELS, read_json, write_json
from infosign.train import load
from infosign.cbm.model import restore
from infosign.cbm.signature import experiment


def selection_root(path):
    path = Path(path).expanduser().resolve()
    return path/'signature_1sd' if (path/'signature_1sd').is_dir() else path


def run_seed(task, seed, baseline, output, device):
    base = baseline/task
    original = read_json(base/'RUN.json')
    selection = read_json(base/'SELECTION_DONE.json')
    ids = selection['selected_ids']
    source, backbone = Path(original['source']), Path(original['backbone'])
    folder = output/f'seed_{seed}'/baseline.name/task
    folder.mkdir(parents=True, exist_ok=True)
    bootstrap_seed = original.get('bootstrap_seed', original['optimization']['seed'])
    contract = dict(training_seed=seed, bootstrap_seed=bootstrap_seed, baseline=str(base),
                    selected_ids=ids,
                    source_run=original)
    if (folder/'RUN.json').exists() and read_json(folder/'RUN.json') != contract:
        raise ValueError(f'{folder}: source selection or settings changed; use a new output directory')
    write_json(folder/'RUN.json', contract)
    if (folder/'exp2/DONE.json').exists():
        print(f'[reuse] {task}/seed_{seed}', flush=True)
        return
    cfg = {**original['optimization'], 'seed': seed, 'bootstrap_seed': bootstrap_seed,
           'bootstrap_samples': original['bootstrap_samples']}
    settings = yaml.safe_load((source/'runtime.yaml').read_text())
    source_cfg = {**settings.get('study', settings), 'output_root': str(source),
                  'backbone_root': str(backbone.parent.parent)}
    torch.set_float32_matmul_precision('highest')
    artifact = load(source/task/'mixed_cbm/model.pt')
    model = restore(artifact, device)
    train = load(backbone/'features_train.pt'); val = load(backbone/'features_val.pt')
    experiment(task, 'exp2', folder, model, artifact, ids, train, val, backbone, source, source_cfg,
               cfg, device, output/f'seed_{seed}'/task/('mixed_cbm_'+baseline.name))


def result_folder(task, seed, baseline, output):
    original_seed = read_json(baseline/task/'RUN.json')['optimization']['seed']
    if seed == original_seed:
        return baseline/task/'exp2'
    return output/f'seed_{seed}'/baseline.name/task/'exp2'


def summarize(baseline, output, tasks, seeds):
    output.mkdir(parents=True, exist_ok=True)
    status, records, comparisons = [], [], []
    for task in tasks:
        for seed in seeds:
            folder = result_folder(task, seed, baseline, output)
            done = read_json(folder/'DONE.json') if (folder/'DONE.json').exists() else {}
            complete = done.get('status') == 'complete'
            status.append(dict(task=task, seed=seed, status=done.get('status', 'pending'), result_directory=str(folder)))
            if not complete:
                continue
            if done['seed'] != seed:
                raise ValueError(f'{folder}: saved training seed does not match the requested seed')
            for split in ('val', 'test'):
                for target, name in ((records, 'performance'), (comparisons, 'comparisons')):
                    frame = pd.read_csv(folder/f'{name}_{split}.csv')
                    frame.insert(0, 'seed', seed)
                    target.append(frame)
    pd.DataFrame(status).to_csv(output/'SEED_STATUS.csv', index=False)
    if not records:
        (output/'SUMMARY_SIGNATURE_SEEDS_test.md').write_text(
            '# Signature CBM training-seed sensitivity\n\nNo completed performance tables yet. See SEED_STATUS.csv.\n')
        return
    all_performance = pd.concat(records, ignore_index=True)
    all_comparisons = pd.concat(comparisons, ignore_index=True)
    lines = ['# Signature CBM training-seed sensitivity', '',
             f'Requested seeds: {", ".join(map(str, seeds))}. Fixed signatures: `{baseline}`.',
             'Only the signature concept predictor and classification head are retrained. '
             'The Full Clinical backbone, data splits, selected Concept IDs and hyperparameters stay fixed. '
             'The original seed is read from its existing results, never retrained here. '
             'Signature-only inference (Experiment 1) is not repeated.', '',
             'Each seed inherits the baseline checkpoint criterion: mean full/masked validation AUROC for '
             '`mean_auroc`, or masked-validation AUROC/F1 for legacy runs. Masked-validation BCE breaks ties. '
             'Thresholds are selected separately by masked-validation F1. '
             'All models are evaluated on full and masked inputs. Bootstrap seed and resample count '
             'are held at the original experiment settings. Each seed retains its participant-bootstrap CIs.', '',
             '**Mean ± SD below describes variation across CBM training seeds (sample SD, ddof=1), '
             'not a confidence interval and not variation across backbone training runs.** '
             'No model is selected using test performance. Incomplete runs are listed in SEED_STATUS.csv.', '',
             '| Task | View | Completed seeds | Test AUROC | Test F1 | Test BA | Test Brier |',
             '|---|---|---:|---|---|---|---|']
    for split in ('val', 'test'):
        perf = all_performance[all_performance.split == split]
        comp = all_comparisons[all_comparisons.split == split]
        perf.to_csv(output/f'PER_SEED_PERFORMANCE_{split}.csv', index=False)
        comp.to_csv(output/f'PER_SEED_COMPARISONS_{split}.csv', index=False)
        metrics = ('auroc', 'f1', 'ba', 'brier')
        summaries = []
        signature = perf[perf.condition.isin(('signature_cbm_full', 'signature_cbm_masked'))]
        for (task, condition), group in signature.groupby(['task', 'condition'], sort=False):
            row = dict(task=task, condition=condition, completed_seeds=len(group), requested_seeds=len(seeds))
            for metric in metrics:
                row[metric+'_mean'] = group[metric].mean()
                row[metric+'_sd'] = group[metric].std(ddof=1)
            summaries.append(row)
            if split == 'test':
                cells = [f'{row[m+"_mean"]:.4f} ± {row[m+"_sd"]:.4f}' if len(group)>1
                         else f'{row[m+"_mean"]:.4f} (SD unavailable)' for m in metrics]
                lines.append('| '+' | '.join([TASK_LABELS.get(task,task), condition.removeprefix('signature_cbm_'),
                             f'{len(group)}/{len(seeds)}', *cells])+' |')
        pd.DataFrame(summaries).to_csv(output/f'ACROSS_SEEDS_{split}.csv', index=False)
    (output/'SUMMARY_SIGNATURE_SEEDS_test.md').write_text('\n'.join(lines)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-root', type=Path, required=True, help='Existing signature_1sd directory or its parent')
    p.add_argument('--output-root', type=Path, help='Default: sibling signature_retrain_seeds directory')
    p.add_argument('--seeds', type=int, nargs='+', default=[42,43,44,45,46])
    p.add_argument('--devices', nargs='+', default=['0'])
    p.add_argument('--tasks', nargs='+')
    p.add_argument('--cpu-threads', type=int, default=2)
    p.add_argument('--report-only', action='store_true')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    a = p.parse_args()
    baseline = selection_root(a.baseline_root)
    output = (a.output_root or baseline.parent/'signature_retrain_seeds').resolve()
    if output == baseline or output in baseline.parents:
        p.error('Use a separate output directory from the baseline selection')
    tasks = a.tasks or [t for t in TASK_LABELS if (baseline/t/'SELECTION_DONE.json').exists()]
    if not tasks: p.error('No completed baseline selections found')
    seeds = list(dict.fromkeys(a.seeds))
    for task in tasks:
        if read_json(baseline/task/'RUN.json')['sd'] != 1:
            p.error(f'{task}: expected the existing 1-SD signature')
    if a.worker:
        torch.set_num_threads(a.cpu_threads)
        run_seed(tasks[0], seeds[0], baseline, output, torch.device(a.devices[0]))
        return
    output.mkdir(parents=True, exist_ok=True)
    if a.report_only:
        summarize(baseline, output, tasks, seeds)
        return
    jobs = [(t,s) for t in tasks for s in seeds if s != read_json(baseline/t/'RUN.json')['optimization']['seed']]
    running, free, failed = {}, list(a.devices), []
    (output/'logs').mkdir(exist_ok=True)
    try:
        while jobs or running:
            while jobs and free:
                (task, seed), gpu = jobs.pop(0), free.pop(0)
                env = os.environ.copy(); device = 'cpu' if gpu == 'cpu' else 'cuda:0'
                if device != 'cpu':
                    index = int(gpu.removeprefix('cuda:')); visible = env.get('CUDA_VISIBLE_DEVICES')
                    env['CUDA_VISIBLE_DEVICES'] = visible.split(',')[index] if visible else str(index)
                for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
                    env[key] = str(a.cpu_threads)
                cmd = [sys.executable, '-u', '-m', 'infosign.cbm.signature_seeds', '--worker',
                       '--baseline-root', str(baseline), '--output-root', str(output), '--tasks', task,
                       '--seeds', str(seed), '--devices', device, '--cpu-threads', str(a.cpu_threads)]
                log = (output/'logs'/f'{task}_seed{seed}.log').open('w')
                running[subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)] = (task,seed,gpu,log)
                print(f'[start] {task} seed={seed} GPU={gpu}', flush=True)
            for process, (task,seed,gpu,log) in list(running.items()):
                code = process.poll()
                if code is not None:
                    log.close(); del running[process]; free.append(gpu)
                    if code: failed.append(f'{task}/seed{seed}')
                    print(f'[finished] {task} seed={seed} exit={code}', flush=True)
                    summarize(baseline, output, tasks, seeds)
            if running: time.sleep(.5)
    finally:
        for process, (_,_,_,log) in running.items():
            if process.poll() is None: process.terminate(); process.wait()
            log.close()
        summarize(baseline, output, tasks, seeds)
    if failed: raise RuntimeError('Failed runs: '+', '.join(failed)+'; see output logs and rerun to resume')
    print('SIGNATURE_SEEDS_COMPLETE: '+str(output/'SUMMARY_SIGNATURE_SEEDS_test.md'), flush=True)


if __name__ == '__main__':
    main()
