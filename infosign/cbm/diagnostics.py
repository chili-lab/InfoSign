"""Post-hoc probability diagnostics and measured Concept-ID supplementation.

No fitting, raw-data extraction, threshold selection, or encoder passes.
All target encodings come from the existing Mixed-CBM checkpoint.
"""
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from infosign.common import backbone_dir, read_json, write_json
from infosign.train import load
from infosign.stats import PairedF1
from infosign.cbm.model import encode, predict, restore
from infosign.cbm.model import concept_id_groups, restoration_batch

VERSION = 2
METRICS = {'f1': 'f1', 'ba': 'balanced_accuracy', 'brier': 'brier', 'auroc': 'auroc'}
DIAGNOSTIC_FILES = tuple(name+'.csv' for name in ('quality','performance','comparisons','restoration')) + tuple(
    f'measured_{kind}_{split}.csv' for split in ('val','test') for kind in ('predictions','availability'))


def completed_diagnostics(folder, task, cfg):
    marker = folder/'DONE.json'
    if not marker.exists() or not all((folder/name).is_file() for name in DIAGNOSTIC_FILES):
        return None
    done = read_json(marker)
    expected = dict(version=VERSION,task=task,seed=cfg['seed'],
                    bootstrap_samples=cfg['bootstrap_samples'],training=False)
    return done if all(done.get(key)==value for key,value in expected.items()) else None


def aligned_labels(frame, ids):
    return frame.reindex(list(map(str, ids))).fillna('')


def measured_targets(model, frame, masked):
    """Replace only observed, encodable items whose channels were learnable.

    Unknown categories and absent/invalid labels retain the masked prediction.
    Constant/untrained channels are not offered as acquisition candidates.
    """
    targets = masked.clone()
    available = np.zeros((len(frame), len(model.spec['items'])), bool)
    for j, item in enumerate(model.spec['items']):
        y, known = encode(frame[item['name']].to_numpy(str), item)
        if not item['learnable']: continue
        available[:, j] = known
        rows = torch.as_tensor(np.flatnonzero(known), device=masked.device)
        if item['type'] == 'continuous':
            targets[rows, item['start']] = torch.as_tensor(y[known], device=masked.device)
        else:
            targets[rows, item['start']:item['stop']] = 0
            categories = torch.as_tensor(y[known], dtype=torch.long, device=masked.device)
            targets[rows, item['start']+categories] = 1
    return targets, available


def concept_quality(items, frame, training_frame, full, masked):
    """Evaluate each ID/source/unit item; do not average incompatible units."""
    rows = []
    for item in items:
        raw = frame[item['name']].to_numpy(str)
        y, known = encode(raw, item)
        recorded = raw != ''
        ty, tk = encode(training_frame[item['name']].to_numpy(str), item)
        cols = slice(item['start'], item['stop'])
        row = {k: item.get(k, '') for k in ('name', 'concept_id', 'concept_name', 'source_field',
               'domain', 'unit', 'qualifier_concept_id', 'type', 'encoding', 'status', 'learnable', 'n_train_observed')}
        row.update(n_total=len(frame), n_recorded=int(recorded.sum()), n_evaluable=int(known.sum()),
                   n_unseen_category=int((recorded & ~known).sum()) if item['type']=='categorical' else 0,
                   n_invalid_numeric=int((recorded & ~known).sum()) if item['type']=='continuous' else 0,
                   n_channels=item['stop']-item['start'])
        if item['type'] == 'continuous':
            row['mean_abs_full_masked_change_standardized'] = float(np.abs(full[:,cols]-masked[:,cols]).mean())
            if known.any():
                for view, z in (('full',full), ('masked',masked)):
                    error = z[known, item['start']]-y[known]
                    row[view+'_standardized_mse'] = float(np.square(error).mean())
                    row[view+'_rmse'] = float(np.sqrt(np.square(error).mean())*item['std'])
                    row[view+'_mae'] = float(np.abs(error).mean()*item['std'])
                # The checkpoint's mean is fitted on unique training labels.
                row['train_mean_standardized_mse'] = float(np.square(y[known]).mean())
                row['masking_standardized_mse_increase'] = row['masked_standardized_mse']-row['full_standardized_mse']
        else:
            row['mean_full_masked_total_variation'] = float(.5*np.abs(full[:,cols]-masked[:,cols]).sum(1).mean())
            row['categories'] = ' | '.join(item['categories'])
            if known.any():
                target = y[known].astype(int); classes = np.unique(target)
                truth = np.eye(item['stop']-item['start'])[target]
                for view, z in (('full',full), ('masked',masked)):
                    p = z[known, cols]; decision = p.argmax(1)
                    row[view+'_accuracy'] = float((decision==target).mean())
                    row[view+'_macro_recall'] = float(np.mean([(decision[target==k]==k).mean() for k in classes]))
                    row[view+'_multiclass_brier'] = float(np.square(p-truth).sum(1).mean())
                    row[view+'_log_loss'] = float(-np.log(np.clip(p[np.arange(len(p)),target],1e-7,1)).mean())
                counts = np.bincount(ty[tk].astype(int), minlength=truth.shape[1])
                prior = counts/counts.sum()
                row['train_prior_multiclass_brier'] = float(np.square(prior-truth).sum(1).mean())
                row['masking_multiclass_brier_increase'] = row['masked_multiclass_brier']-row['full_multiclass_brier']
        rows.append(row)
    return rows


def metric_draws(boot, probabilities, cut):
    output = {}
    for name, metric in METRICS.items():
        if name in ('ba','auroc') and len(np.unique(boot.y))<2:
            output[name] = np.full((len(np.atleast_2d(probabilities)), boot.samples+1), np.nan)
        else:
            output[name] = boot.draws(probabilities, cut, metric)
    return output


def interval_fields(name, row):
    return {name if k=='value' else name+'_'+k: v for k,v in row.items()}


def supplementation_statistics(boot, baseline, restored, cut):
    """Batched paired effects; positive gains mean improvement for all metrics."""
    restored = np.atleast_2d(restored)
    base_draws = metric_draws(boot, baseline, cut)
    new_draws = metric_draws(boot, restored, cut)
    rows = [{} for _ in restored]
    for name, metric in METRICS.items():
        base = boot.intervals(base_draws[name])[0]
        perfs = boot.intervals(new_draws[name])
        sign = -1 if name=='brier' else 1
        effects = boot.intervals(sign*(new_draws[name]-base_draws[name]))
        ps = (np.full(len(rows),np.nan) if name in ('ba','auroc') and len(np.unique(boot.y))<2
              else boot.pvalues(restored,baseline,cut,cut,metric))
        for row, perf, effect, p in zip(rows,perfs,effects,ps):
            row.update(**interval_fields('baseline_'+name,base), **interval_fields('restored_'+name,perf),
                       **interval_fields(name+'_gain',effect), **{name+'_p_value':float(p)})
    before = baseline>=cut
    for row, p in zip(rows,restored):
        after = p>=cut; correct_before = before==boot.y; correct_after = after==boot.y
        row.update(n=len(boot.y), positive=int(boot.y.sum()), negative=int((boot.y==0).sum()),
                   changed=int((before!=after).sum()), corrected=int((~correct_before & correct_after).sum()),
                   newly_wrong=int((correct_before & ~correct_after).sum()),
                   tp=int((after & (boot.y==1)).sum()), fp=int((after & (boot.y==0)).sum()),
                   fn=int((~after & (boot.y==1)).sum()), tn=int((~after & (boot.y==0)).sum()),
                   mean_abs_probability_change=float(np.abs(p-baseline).mean()))
    return rows


def restore_ids(model, missing, reference, groups):
    columns = [g['columns'] for g in groups.values()]
    return np.concatenate([restoration_batch(model,missing,reference,columns[s:s+32])
                           for s in range(0,len(columns),32)])


def run_diagnostics(task, cfg, device):
    root = Path(cfg['output_root'])/task
    source_root = Path(cfg['output_root'])
    out = root/'concept_diagnostics'; out.mkdir(parents=True,exist_ok=True)
    if completed_diagnostics(out,task,cfg):
        print(f'[reuse diagnostics] {task}',flush=True); return
    (out/'DONE.json').unlink(missing_ok=True)
    backbone = backbone_dir(task,cfg)/'full'
    artifact = load(source_root/task/'mixed_cbm/model.pt'); model = restore(artifact,device)
    torch.set_float32_matmul_precision('highest')
    cut = artifact['threshold']; full_cut = read_json(backbone/'TRAINING_DONE.json')['threshold']
    frame = pd.read_csv(source_root/'labels'/task/'labels.csv',dtype=str,keep_default_na=False).set_index('person_id')
    training = aligned_labels(frame,load(backbone/'features_train.pt')['person_ids'])
    groups = concept_id_groups(model.spec['items']); names = list(groups)
    indices = {cid:[j for j,c in enumerate(model.spec['items']) if str(c['concept_id'])==cid] for cid in names}
    quality, performance, comparisons, restorations = [], [], [], []
    for split in ('val','test'):
        cache = load(backbone/f'features_{split}.pt'); y = cache['y'].numpy()
        labels = aligned_labels(frame,cache['person_ids'])
        fz, fp = predict(model,cache['embedding']); mz, mp = predict(model,cache['masked_embedding'])
        measured, item_available = measured_targets(model,labels,mz)
        available = np.stack([item_available[:,indices[cid]].any(1) for cid in names])
        quality += [dict(task=task,split=split,**r) for r in concept_quality(model.spec['items'],labels,training,
                                                                         fz.cpu().numpy(),mz.cpu().numpy())]
        predicted = restore_ids(model,mz,fz,groups)
        supplemented = restore_ids(model,mz,measured,groups)
        boot = PairedF1(y,cfg['bootstrap_samples'],cfg['seed'],device)
        conditions = {'full_original':(cache['original_logits'].sigmoid().numpy(),full_cut),
                      'full_original_masked':(cache['masked_logits'].sigmoid().numpy(),full_cut),
                      'mixed_full':(fp,cut),'mixed_masked':(mp,cut)}
        for name,(p,threshold) in conditions.items():
            row = dict(task=task,split=split,condition=name,threshold=threshold,n=len(y))
            for metric,draws in metric_draws(boot,p,threshold).items():
                row.update(interval_fields(metric,boot.intervals(draws)[0]))
            performance.append(row)
        for label,full,missing in [('original', 'full_original','full_original_masked'),('mixed','mixed_full','mixed_masked')]:
            p,threshold = conditions[full]; baseline,_ = conditions[missing]
            comparisons.append(dict(task=task,split=split,model=label,threshold=threshold,
                               **supplementation_statistics(boot,baseline,p,threshold)[0]))
        meta = [dict(task=task,split=split,concept_id=cid,concept_name=g['concept_name'],
                     source_fields=' | '.join(g['source_fields']),channels=len(g['columns']),learnable=g['learnable'],
                     threshold=cut,n_total=len(y),n_eligible=int(available[i].sum()),
                     n_positive_eligible=int(y[available[i]].sum()),
                     n_negative_eligible=int((y[available[i]]==0).sum()),
                     coverage=float(available[i].mean()),
                     n_complete=int(item_available[:,[j for j in indices[cid] if model.spec['items'][j]['learnable']]].all(1).sum())
                         if g['learnable'] else 0)
                for i,(cid,g) in enumerate(groups.items())]
        for source,ps in [('full_input_prediction',predicted),('measured',supplemented)]:
            stats = supplementation_statistics(boot,mp,ps,cut)
            restorations += [dict(**m,source=source,population='all_participants',
                                  status=('not_learnable' if not m['learnable'] else
                                          'no_observed_learnable_label' if source=='measured' and m['n_eligible']==0 else 'evaluated'),**r)
                             for m,r in zip(meta,stats)]
        # Reuse whole-cohort results; batch IDs with identical observed cohorts.
        cohorts = defaultdict(list)
        for i,mask in enumerate(available): cohorts[mask.tobytes()].append(i)
        del boot
        for number, js in enumerate(cohorts.values(),1):
            mask = available[js[0]]
            if mask.any() and not mask.all():
                subset = PairedF1(y[mask],cfg['bootstrap_samples'],cfg['seed'],device,allow_single_class=True)
                rows = supplementation_statistics(subset,mp[mask],supplemented[js][:,mask],cut)
                del subset
            elif mask.all(): rows = [stats[j] for j in js]  # Last full-cohort source was measured.
            else: rows = [dict(n=0,positive=0,negative=0) for _ in js]
            restorations += [dict(**meta[j],source='measured',population='observed_participants',
                                 status='evaluated' if mask.any() else 'no_observed_learnable_label',**row)
                             for j,row in zip(js,rows)]
            if number%10==0:
                print(f'[diagnostics {task}/{split}] observed cohorts {number}/{len(cohorts)}',flush=True)
        predictions = pd.DataFrame(dict(person_id=cache['person_ids'],y=y,masked_probability=mp,full_probability=fp))
        predictions = pd.concat([predictions,pd.DataFrame(supplemented.T,columns=names)],axis=1)
        predictions.to_csv(out/f'measured_predictions_{split}.csv',index=False)
        pd.DataFrame(available.T,columns=names,index=cache['person_ids']).rename_axis('person_id').to_csv(out/f'measured_availability_{split}.csv')
        print(f'[diagnostics {task}/{split}] {len(names)} IDs; quality, AUROC, Brier and measured supplementation complete',flush=True)
    for name,rows in [('quality',quality),('performance',performance),('comparisons',comparisons),('restoration',restorations)]:
        pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False)
    write_json(out/'DONE.json',dict(version=VERSION,task=task,bootstrap_samples=cfg['bootstrap_samples'],
                                  threshold=cut,seed=cfg['seed'],training=False,
                                  exclude_observation=bool(cfg.get('exclude_observation')),
                                  cbm_source_root=str(source_root),backbone_root=str(backbone.parent.parent)))
