"""Full/Basic modality ablation and Mixed-CBM structured-clinical restoration."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from infosign.common import REMOVALS, read_json, write_json, backbone_dir
from infosign.train import load, cached_removals
from infosign.stats import PairedF1, f1
from infosign.cbm.model import restore, predict
from infosign.cbm.model import concept_id_groups, restoration_batch

ANALYSIS_VERSION = 5


def paired_effects(boot, a, b, ca, cb):
    f = boot.effects(a, b, ca, cb)
    ba = boot.effects(a, b, ca, cb, metric='balanced_accuracy')
    auc = boot.effects(a, b, ca, cb, metric='auroc')
    return [{**x, **{'ba_'+k: v for k, v in y.items()}, **{'auroc_'+k: v for k, v in z.items()}}
            for x, y, z in zip(f, ba, auc)]


def performance(boot, p, cut):
    return {**boot.performance(p, cut),
            **{'ba_'+k: v for k, v in boot.performance(p, cut, metric='balanced_accuracy').items()},
            **{'auroc_'+k: v for k, v in boot.performance(p, cut, metric='auroc').items()}}


def restoration_details(y, restored, masked, cut, learnable):
    """Show ties and distinguish equal metrics from identical patient predictions."""
    delta = f1(y, restored, cut)-f1(y, masked, cut)
    rank = pd.Series(delta).where(learnable).rank(method='min', ascending=False)
    counts = pd.Series(delta[np.asarray(learnable, bool)]).value_counts()
    decisions = restored >= cut
    _, probability_group = np.unique(restored, axis=0, return_inverse=True)
    _, decision_group = np.unique(decisions, axis=0, return_inverse=True)
    return [dict(rank=None if pd.isna(rank[i]) else int(rank[i]),
                 tied_concepts=int(counts[delta[i]]) if learnable[i] else 0,
                 mean_abs_probability_change=float(np.abs(p-masked).mean()),
                 max_abs_probability_change=float(np.abs(p-masked).max()),
                 n_prediction_changes=int(np.count_nonzero(decisions[i] != (masked >= cut))),
                 probability_group=int(probability_group[i])+1, decision_group=int(decision_group[i])+1,
                 tp=int(np.sum(decisions[i] & (y==1))), fp=int(np.sum(decisions[i] & (y==0))),
                 fn=int(np.sum(~decisions[i] & (y==1))), tn=int(np.sum(~decisions[i] & (y==0))))
            for i, p in enumerate(restored)]


def run_analysis(task, cfg, device):
    root = Path(cfg['output_root'])/task
    backbone = backbone_dir(task, cfg)
    out = root/'analysis'; out.mkdir(parents=True, exist_ok=True)
    done = read_json(out/'DONE.json') if (out/'DONE.json').exists() else {}
    if done.get('analysis_version') == ANALYSIS_VERSION:
        print(f'[reuse analysis] {task}', flush=True); return
    # Invalidate old completion before writing the replacement tables.
    (out/'DONE.json').unlink(missing_ok=True)
    # Original multimodal checkpoints/features are read-only, including on upgrade.
    removals = {v: cached_removals(task, v, cfg, device, out) for v in ('full','basic')}
    artifact = load(root/'mixed_cbm/model.pt')
    model = restore(artifact, device)
    groups = concept_id_groups(artifact['spec']['items'])
    cuts = {k: read_json(backbone/k/'TRAINING_DONE.json')['threshold'] for k in ('full','basic')}
    cuts['mixed_cbm'] = artifact['threshold']
    torch.set_float32_matmul_precision('highest')
    classification, comparisons, ablations, recoveries, cohorts = [], [], [], [], []
    selection = dict(metric='positive_class_f1', selection_split='val', top_modalities={},
                     restoration_mode='mixed', restoration_input='nonbasic_clinical')
    for split in ('val','test'):
        basic = load(backbone/'basic'/f'features_{split}.pt')
        full = load(backbone/'full'/f'features_{split}.pt')
        if basic['person_ids'] != full['person_ids'] or not torch.equal(basic['y'], full['y']):
            raise ValueError(f'{task}/{split}: Full and Basic predictions must use the same ordered participants')
        y = basic['y'].numpy(); n = len(y)
        boot = PairedF1(y, cfg['bootstrap_samples'], cfg['seed'], device)
        fz, fp = predict(model, full['embedding'])
        mz, mp = predict(model, full['masked_embedding'])
        conditions = {
            'full': (full['original_logits'].sigmoid().numpy(), cuts['full']),
            'full_basic_inference': (full['masked_logits'].sigmoid().numpy(), cuts['full']),
            'basic': (basic['original_logits'].sigmoid().numpy(), cuts['basic']),
            'mixed_full': (fp, cuts['mixed_cbm']),
            'mixed_basic': (mp, cuts['mixed_cbm'])}
        cohorts.append(dict(task=task, split=split, n=n, positive=int(y.sum()), negative=int(n-y.sum()),
                            prevalence=float(y.mean()), all_positive_f1=float(2*y.sum()/(n+y.sum()))))
        predictions = dict(person_id=basic['person_ids'], y=y)
        for name, (p, cut) in conditions.items():
            classification.append(dict(task=task, split=split, model=name, threshold=cut,
                                       n=n, **performance(boot, p, cut)))
            predictions[name] = p; predictions[name+'__decision'] = (p>=cut).astype(int)
        definitions = [
            ('full_input_loss','full','full_basic_inference','downstream'),
            ('basic_vs_full','basic','full','downstream'),
            ('retraining_gain','basic','full_basic_inference','downstream'),
            ('mixed_cbm_gain','mixed_basic','full_basic_inference','cbm'),
            ('mixed_input_loss','mixed_full','mixed_basic','cbm')]
        for label, a, b, family in definitions:
            pa, ca = conditions[a]; pb, cb = conditions[b]
            comparisons.append(dict(task=task, split=split, comparison=label, family=family,
                minuend=a, subtrahend=b, n=n, **paired_effects(boot, pa, pb, ca, cb)[0]))
        for variant in ('full','basic'):
            original, cut = conditions[variant]
            removed = removals[variant][split]
            ps = np.stack([removed[k]['probability'] for k in REMOVALS])
            effects = paired_effects(boot, original, ps, cut, cut)
            if split=='val':
                ranking = sorted(zip(REMOVALS, effects), key=lambda pair: (-pair[1]['value'], pair[0]))
                selection['top_modalities'][variant] = [name for name, _ in ranking[:3]]
            baseline = performance(boot, original, cut)
            for name, effect in zip(REMOVALS, effects):
                p = removed[name]['probability']; per = performance(boot, p, cut)
                top = selection['top_modalities'][variant]
                ablations.append(dict(task=task, split=split, model=variant, removed_input=name,
                    family=variant+'_ablation', n=n, threshold=cut,
                    n_available=int(removed[name]['available'].sum()),
                    val_rank=top.index(name)+1 if name in top else None,
                    full_f1=baseline['value'], removed_f1=per['value'],
                    full_ba=baseline['ba_value'], removed_ba=per['ba_value'],
                    full_auroc=baseline['auroc_value'], removed_auroc=per['auroc_value'], **effect))
                predictions[variant+'__without__'+name] = p
        pd.DataFrame(predictions).to_csv(out/f'predictions_{split}.csv', index=False)
        # All tasks use exactly the same Mixed-CBM missing-input condition.
        cut = cuts['mixed_cbm']; names = list(groups)
        restored = np.concatenate([restoration_batch(model, mz, fz, [groups[c]['columns'] for c in names[start:start+32]])
                                   for start in range(0, len(names), 32)])
        details = restoration_details(y, restored, mp, cut, [groups[c]['learnable'] for c in names])
        names.append('ALL_CONCEPTS')
        all_restored = restoration_batch(model, mz, fz, [list(range(model.spec['width']))])
        restored = np.concatenate((restored, all_restored))
        all_details = restoration_details(y, all_restored, mp, cut, [False])[0]
        # Equivalence-group IDs are local to individual concepts in this condition.
        all_details.update(probability_group=None, decision_group=None)
        details.append(all_details)
        rows = paired_effects(boot, restored, mp, cut, cut)
        perfs = boot.intervals(boot.draws(restored, cut))
        ba_perfs = boot.intervals(boot.draws(restored, cut, metric='balanced_accuracy'))
        auc_perfs = boot.intervals(boot.draws(restored, cut, metric='auroc'))
        masked_stat = performance(boot, mp, cut); full_stat = performance(boot, fp, cut)
        for cid, effect, perf, ba_perf, auc_perf, detail in zip(names, rows, perfs, ba_perfs, auc_perfs, details):
            g = groups.get(cid, dict(concept_name='All CBM concepts restored together',
                source_fields=list(dict.fromkeys(c['source_field'] for c in model.spec['items'])),
                columns=list(range(model.spec['width'])), learnable=False))
            recoveries.append(dict(task=task, split=split, mode='mixed', removed_input='nonbasic_clinical',
                concept_id=cid, concept_name=g['concept_name'], source_fields=' | '.join(g['source_fields']),
                channels=len(g['columns']), ranking_split=split, learnable=g['learnable'], **detail,
                intervention='all_concepts' if cid=='ALL_CONCEPTS' else 'individual',
                family='mixed_all_restoration' if cid=='ALL_CONCEPTS' else 'mixed_restoration', n=n, threshold=cut,
                full_f1=full_stat['value'], masked_f1=masked_stat['value'],
                full_ba=full_stat['ba_value'], masked_ba=masked_stat['ba_value'],
                full_auroc=full_stat['auroc_value'], masked_auroc=masked_stat['auroc_value'],
                restored_f1=perf['value'], restored_ci_lower=perf['ci_lower'], restored_ci_upper=perf['ci_upper'],
                restored_ba=ba_perf['value'], restored_ba_ci_lower=ba_perf['ci_lower'], restored_ba_ci_upper=ba_perf['ci_upper'],
                restored_auroc=auc_perf['value'], restored_auroc_ci_lower=auc_perf['ci_lower'], restored_auroc_ci_upper=auc_perf['ci_upper'],
                **effect))
        table = pd.DataFrame(restored.T, columns=names)
        table.insert(0,'masked_probability',mp); table.insert(0,'y',y); table.insert(0,'person_id',basic['person_ids'])
        table.to_csv(out/f'restoration_mixed__nonbasic_clinical_{split}.csv', index=False)
        print(f'[restore {task}/{split}/mixed] {len(groups)} IDs + all concepts; ranked on {split}', flush=True)
        del boot
    write_json(out/'SELECTION.json', selection)
    for name, rows in [('classification',classification), ('comparisons',comparisons), ('ablation',ablations),
                       ('restoration',recoveries), ('cohorts',cohorts)]:
        pd.DataFrame(rows).to_csv(out/(name+'.csv'), index=False)
    write_json(out/'DONE.json', dict(task=task, seed=cfg['seed'], bootstrap_samples=cfg['bootstrap_samples'],
                                   analysis_version=ANALYSIS_VERSION,
                                   reported_metrics=['f1','balanced_accuracy','auroc']))
