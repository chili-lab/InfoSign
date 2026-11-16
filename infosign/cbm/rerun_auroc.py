"""AUROC-selected CBM experiments, optionally refitting the Full/Basic backbones."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd
import torch
import yaml

from infosign.common import read_json, write_json, backbone_dir, TASK_LABELS
from infosign.stats import auroc_selection
from infosign.cbm.model import run_cbm
from infosign.cbm.diagnostics import run_diagnostics
from infosign.cbm.validation import run_validation_recovery, make_validation_report, ci
from infosign.cbm.signature import TRAIN_KEYS

BACKBONE_KEYS = ('source_root', 'prepared_inputs_root', 'clinical_dir', 'batch_size',
                 'num_workers', 'cpu_threads', 'max_epochs', 'patience', 'learning_rate', 'weight_decay')


def prepare_run(args):
    settings = yaml.safe_load(args.config.read_text())
    base = settings.get('study', settings)
    tasks = args.tasks or base['tasks']
    root = args.output_root.resolve()
    sources, overrides, links = {}, {}, {}
    for task in tasks:
        source = next((p.resolve() for p in args.cbm_roots if (p/task/'mixed_cbm/model.pt').is_file()), None)
        if source is None: raise FileNotFoundError(f'{task}: no source Mixed CBM in --cbm-roots')
        old = yaml.safe_load((source/'runtime.yaml').read_text()); old = old.get('study', old)
        old = {**old, **old.get('task_settings', {}).get(task, {}), 'output_root': str(source)}
        sources[task] = str(source)
        overrides[task] = {k: old.get(k, base.get(k)) for k in TRAIN_KEYS}
        overrides[task].update(seed=args.seed, cbm_checkpoint_metric='mean_auroc')
        links[root/'labels'/task] = source/'labels'/task
        original_backbone = backbone_dir(task, old)
        if args.retrain_backbones:
            # The source CBM may have reused an older multimodal study. Read
            # that backbone's training settings and existing input registry.
            training_path = original_backbone.parent/'runtime.yaml'
            training = yaml.safe_load(training_path.read_text())
            training = training.get('study', training)
            training = {**training, **training.get('task_settings', {}).get(task, {})}
            inherited = {**base, **old, **training}
            overrides[task].update({k: inherited[k] for k in BACKBONE_KEYS if k in inherited})
            overrides[task]['backbone_checkpoint_metric'] = 'auroc'
        else:
            links[root/'backbones'/task] = original_backbone
        schema = read_json(source/'labels'/task/'schema.json')
        if any(c.get('domain') == 'Observation' for c in schema['concepts']):
            raise ValueError(f'{task}: this rerun requires the agreed no-Observation concept bank')
    cfg = {**base, 'workflow': 'cbm_auroc_rerun', 'output_root': str(root),
           'backbone_root': str(root/'backbones'), 'task_settings': overrides, 'original_sources': sources,
           'tasks': tasks, 'seed': args.seed, 'cbm_checkpoint_metric': 'mean_auroc',
           'bootstrap_samples': args.bootstrap_samples, 'selection_sd': 1.0,
           'restore_all_concepts': True, 'exclude_observation': True}
    cfg.pop('concept_sources', None)
    if args.retrain_backbones:
        cfg.pop('backbone_root', None)
        cfg.update(workflow='full_pipeline_auroc_rerun', retrain_backbones=True,
                   backbone_checkpoint_metric='auroc', analysis_individual_restoration=False)
    else:
        cfg.pop('backbone_checkpoint_metric', None)
    if (root/'runtime.yaml').exists():
        existing = yaml.safe_load((root/'runtime.yaml').read_text())
        if existing != cfg:
            raise ValueError('Output configuration differs. Use a separate new --output-root.')
    elif root.exists() and any(root.iterdir()):
        raise ValueError('Choose an empty output directory; do not mix with earlier F1 results')
    root.mkdir(parents=True, exist_ok=True)
    for link, target in links.items():
        target = target.resolve(strict=True)
        if link.is_symlink():
            if link.resolve() != target: raise ValueError(f'{link}: source path changed')
        elif link.exists():
            raise ValueError(f'{link}: expected a read-only source link')
        else:
            link.parent.mkdir(parents=True, exist_ok=True); link.symlink_to(target, target_is_directory=True)
    (root/'runtime.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    return cfg


def worker(cfg, task, phase, device):
    cfg = {**cfg, **cfg['task_settings'][task]}
    torch.set_num_threads(cfg.get('cpu_threads', 2))
    if phase in ('full', 'basic'):
        from infosign.train import run_model
        if device.type == 'cuda': torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
        run_model(task, phase, cfg, device)
    elif phase == 'analyze':
        from infosign.analysis import run_analysis
        run_analysis(task, cfg, device)
    elif phase == 'train':
        run_cbm(task, cfg, device)
    else:
        done = read_json(Path(cfg['output_root'])/task/'mixed_cbm/DONE.json')
        if done.get('checkpoint_metric') != 'mean_auroc':
            raise ValueError('Recovery requires the new mean-AUROC-selected CBM')
        run_diagnostics(task, cfg, device)
        marker = Path(cfg['output_root'])/'validation_selected'/task/'DONE.json'
        if marker.exists():
            saved = read_json(marker)
            if saved.get('selection_sd') != 1 or not saved.get('restore_all_concepts'):
                raise ValueError('Existing recovery settings differ; use a new output root')
            print(f'[reuse recovery] {task}', flush=True)
        else:
            run_validation_recovery(task, cfg, device)


def task_pool(cfg, phase, devices):
    root = Path(cfg['output_root']); (root/'logs').mkdir(exist_ok=True)
    phases = ('full', 'basic') if phase == 'backbones' else (phase,)
    waiting = [(task, p) for task in cfg['tasks'] for p in phases]
    free, active, failed = list(devices), {}, []
    last_update = time.monotonic()
    try:
        while waiting or active:
            while waiting and free:
                (task, job_phase), gpu = waiting.pop(0), free.pop(0)
                job_cfg = {**cfg, **cfg['task_settings'][task]}
                env = os.environ.copy(); device = 'cpu' if gpu == 'cpu' else 'cuda:0'
                if device != 'cpu':
                    index = int(gpu.removeprefix('cuda:')); visible = env.get('CUDA_VISIBLE_DEVICES')
                    env['CUDA_VISIBLE_DEVICES'] = visible.split(',')[index] if visible else str(index)
                for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
                    env[key] = str(job_cfg.get('cpu_threads', 2))
                log = (root/'logs'/f'{task}_{job_phase}.log').open('a')
                cmd = [sys.executable, '-u', '-m', 'infosign.cbm.rerun_auroc', '--worker',
                       '--config', str(root/'runtime.yaml'), '--stage', job_phase, '--tasks', task, '--devices', device]
                active[subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)] = (task,job_phase,gpu,log)
                print(f'[start {job_phase}] {task} on {gpu}', flush=True)
            for process, (task,job_phase,gpu,log) in list(active.items()):
                if process.poll() is not None:
                    log.close(); del active[process]; free.append(gpu)
                    if process.returncode: failed.append(f'{task}/{job_phase}')
                    print(f'[finished {job_phase}] {task}, exit={process.returncode}', flush=True)
            if active and time.monotonic()-last_update > 30:
                names = ', '.join(f'{t}/{p}' for t,p,_,_ in active.values())
                print(f'[running] {names}; queued={len(waiting)}; epoch logs: {root/"logs"}', flush=True)
                last_update = time.monotonic()
            if active: time.sleep(.5)
    finally:
        for process, (_,_,_,log) in active.items():
            if process.poll() is None: process.terminate(); process.wait()
            log.close()
    if failed: raise RuntimeError(f'{phase} failed for {failed}; see {root/"logs"} and rerun to resume')


def new_reference(cfg):
    """2-SD consistency check derived ONLY from the new model's validation gains."""
    root = Path(cfg['output_root']); tasks = {}
    for task in cfg['tasks']:
        table = pd.read_csv(root/'validation_selected'/task/'CONCEPT_SELECTION_val.csv', dtype={'concept_id': str})
        group = table[table.source == 'full_input_prediction']
        flags = auroc_selection(group.val_auroc_gain.to_numpy())['auroc_selected']
        tasks[task] = dict(eligible_ids=len(group), selected_ids=group.loc[flags,'concept_id'].tolist())
    path = root/'signature_reference_auroc.json'
    write_json(path, dict(description='New AUROC-selected CBM: validation-only 2-SD reference; NOT the old F1 model lists.', tasks=tasks))
    return path


def reports(cfg):
    root = Path(cfg['output_root']); tables = root/'tables'; tables.mkdir(exist_ok=True)
    if cfg.get('retrain_backbones'):
        from infosign.report import make_report
        make_report(cfg)
    make_validation_report(cfg)
    for kind in ('quality','restoration','performance','comparisons'):
        frames = [pd.read_csv(root/t/'concept_diagnostics'/f'{kind}.csv', dtype={'concept_id':str})
                  for t in cfg['tasks'] if (root/t/'concept_diagnostics/DONE.json').exists()]
        if frames:
            frame = pd.concat(frames, ignore_index=True)
            for split in ('val','test'):
                frame[frame.split==split].to_csv(tables/f'ALL_CONCEPT_{kind.upper()}_{split}.csv', index=False)
    backbone_note = ('Full/Basic models were refitted from existing preprocessed inputs, selecting each model by '
                     'validation AUROC on its own input view, with validation BCE breaking ties. '
                     'All downstream features, CBMs and signatures were regenerated.' if cfg.get('retrain_backbones') else
                     'The original multimodal encoders/fusion and outcome models were not retrained.')
    lines = ['# AUROC-selected experiment rerun', '',
             backbone_note, '',
             'Concept pretraining: masked-validation concept loss. Head/joint checkpoint selection: '
             '(full-validation AUROC + masked-validation AUROC) / 2, with masked-validation BCE breaking ties. '
             'The 10% per-item masked-validation concept-loss gate and sequential fallback remain. '
             'CBM classification thresholds separately maximize masked-validation F1 and stay fixed across full/masked/intervened predictions.', '',
             f'Seed {cfg["seed"]}; {cfg["bootstrap_samples"]:,} paired outcome-stratified bootstrap resamples. '
             'Both predicted and measured recovery use new validation μ + 1σ selections; all-ID recovery is also reported.', '',
             '| Task | Head epoch | Joint epoch | Joint status | Val AUROC full | Val AUROC masked | Val AUROC mean | Test AUROC full [95% CI] | Test AUROC masked [95% CI] |',
             '|---|---:|---:|---|---:|---:|---:|---|---|']
    for task in cfg['tasks']:
        path = root/task/'mixed_cbm/DONE.json'
        metrics = root/task/'concept_diagnostics/performance.csv'
        if not path.exists() or not (metrics.parent/'DONE.json').exists(): continue
        done = read_json(path); df = pd.read_csv(metrics)
        r = {(x['split'],x['condition']):x for x in df.to_dict('records')}
        head,joint = done['selection']['head'],done['selection']['joint']
        cells = [TASK_LABELS.get(task,task),head['epoch'],joint['epoch'],joint['status'],
                 f'{r["val","mixed_full"]["auroc"]:.4f}',f'{r["val","mixed_masked"]["auroc"]:.4f}',
                 f'{(r["val","mixed_full"]["auroc"]+r["val","mixed_masked"]["auroc"])/2:.4f}',
                 ci(r['test','mixed_full'],'auroc'),ci(r['test','mixed_masked'],'auroc')]
        lines.append('| '+' | '.join(map(str,cells))+' |')
    lines += ['', '## Detailed outputs', '',
              '- `validation_selected/tables/SUMMARY_VALIDATION_SELECTED_test.md`: new validation-selected single/joint predicted and measured recovery, plus all-ID restoration.',
              '- `tables/ALL_CONCEPT_RESTORATION_{val,test}.csv`: every individual ID, without selecting concepts on test.',
              '- `signature_1sd/SUMMARY_SIGNATURE_test.md`: signature-only inference and new signature CBM (seed 42).',
              '- `signature_retrain_seeds/SUMMARY_SIGNATURE_SEEDS_test.md`: signature CBM seeds 42–46, with mean ± sample SD.',
              '- `logs/`: per-task training/recovery logs.', '',
              'All selections are validation-based; test data are used for evaluation. '
              'Earlier exploratory use of this test set is not erased by this rerun. '
              'Bootstrap CIs condition on the fitted model/signature; across-seed SD describes CBM readout variability only.', '']
    if cfg.get('retrain_backbones'):
        lines += ['## Full/Basic results', '',
                  '- `tables/SUMMARY_{val,test}.md`: new Full/Basic downstream performance, both modality-ablation panels and Mixed CBM performance.',
                  '- `tables/classification.csv`: all models and input conditions, validation/test metrics and CIs.',
                  '- `tables/MODALITY_ABLATION_{full,basic}_{val,test}.csv`: all 11 removals per model/task.',
                  '- `tables/BACKBONE_SELECTION.csv`: selected Full/Basic epochs, validation AUROC/F1 and thresholds.', '']
    (root/'SUMMARY_AUROC_RERUN.md').write_text('\n'.join(lines)+'\n')
    if cfg.get('retrain_backbones'):
        rows = []
        for task in cfg['tasks']:
            for variant in ('full', 'basic'):
                path = root/task/variant/'TRAINING_DONE.json'
                if path.exists():
                    d = read_json(path)
                    rows.append(dict(task=task, model=variant, checkpoint_metric=d['checkpoint_metric'],
                                     epoch=d['epoch'], val_auroc=d['val_auroc'], val_f1=d['val_f1'],
                                     val_bce=d['val_loss'], threshold=d['threshold']))
        pd.DataFrame(rows).to_csv(tables/'BACKBONE_SELECTION.csv', index=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=Path('config.yaml'))
    p.add_argument('--cbm-roots', type=Path, nargs='+')
    p.add_argument('--output-root', type=Path)
    p.add_argument('--devices', nargs='+', default=['0'])
    p.add_argument('--tasks', nargs='+')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--bootstrap-samples', type=int, default=10000)
    p.add_argument('--signature-seeds', type=int, nargs='+', default=[42,43,44,45,46])
    p.add_argument('--retrain-backbones', action='store_true',
                   help='Refit Full/Basic on existing processed inputs using validation AUROC, then regenerate CBM features.')
    p.add_argument('--stage', choices=('all','backbones','full','basic','train','analyze','recovery','application','seeds','report'), default='all')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.worker:
        cfg = yaml.safe_load(a.config.read_text())
        worker(cfg, a.tasks[0], a.stage, torch.device(a.devices[0])); return
    if not a.cbm_roots or not a.output_root:
        p.error('--cbm-roots and a new --output-root are required')
    cfg = prepare_run(a); root = Path(cfg['output_root'])
    if a.stage in ('backbones','full','basic','analyze') and not a.retrain_backbones:
        p.error('This stage requires --retrain-backbones')
    try:
        phases = ('backbones','train','analyze','recovery') if a.retrain_backbones else ('train','recovery')
        if a.stage in ('full','basic'): task_pool(cfg, a.stage, a.devices)
        for phase in phases:
            if a.stage in ('all',phase): task_pool(cfg, phase, a.devices)
        if a.stage in ('all','recovery','application'): reference = new_reference(cfg)
        if a.stage in ('all','application'):
            subprocess.run([sys.executable,'-u','-m','infosign.cbm.signature','--config',str(root/'runtime.yaml'),
                            '--cbm-roots',str(root),'--output-root',str(root),'--reference-selection',str(reference),
                            '--sd','1','--stage','all','--devices',*a.devices], check=True)
        if a.stage in ('all','seeds'):
            subprocess.run([sys.executable,'-u','-m','infosign.cbm.signature_seeds','--baseline-root',str(root/'signature_1sd'),
                            '--seeds',*map(str,a.signature_seeds),'--devices',*a.devices], check=True)
    finally:
        reports(cfg)
    print('AUROC_RERUN_COMPLETE: '+str(root/'SUMMARY_AUROC_RERUN.md'), flush=True)


if __name__ == '__main__':
    main()
