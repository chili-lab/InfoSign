"""Fresh Full/Basic multimodal fitting and reusable frozen feature exports."""
import math
import random
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from infosign.common import MODALITIES, REMOVALS, read_json, write_json, save_torch
from infosign.data import CachedDataset, clinical_inputs, collate, cache_config
from infosign.stats import BalancedSampler, f1, threshold
from infosign.encoders import Task2V2Model, Task2V2ModelConfig, PresenceAwareFusion, BinaryTaskHead


class CachedModel(Task2V2Model):
    def _retinal(self, batch, group):
        features = batch[group+'_features']; valid = batch[group+'_view_mask']
        encoder = self.retinal_encoder
        tokens = encoder.view_projection(features)*valid.unsqueeze(-1)
        assets = encoder.view_pool(tokens, valid); present = valid.any(-1)
        value = encoder.group_projections[group](encoder.asset_pools[group](assets, present))
        return value, present.any(-1)


def load(path): return torch.load(path, map_location='cpu', weights_only=False)
def move(batch, device): return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items()}
def amp(device): return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type=='cuda')


def loader(dataset, cfg, training=False):
    y = dataset.master.iloc[dataset.indices].y.to_numpy(int)
    sampler = BalancedSampler(y, cfg['seed']) if training else None
    return DataLoader(dataset, batch_size=cfg['batch_size'], sampler=sampler,
        num_workers=cfg['num_workers'], collate_fn=collate, pin_memory=True,
        persistent_workers=bool(cfg['num_workers']),
        **({'multiprocessing_context':'spawn'} if cfg['num_workers'] else {}))


@torch.no_grad()
def predict_loader(model, dl, task, device):
    model.eval(); probabilities = []; truth = []
    for batch in dl:
        b = move(batch, device)
        with amp(device): logits = model(b, task=task)['logits'].reshape(-1)
        probabilities.append(logits.float().sigmoid().cpu().numpy()); truth.append(b['y'].cpu().numpy())
    return np.concatenate(truth), np.concatenate(probabilities)


def fit(model, train_dl, val_dl, task, cfg, root, device):
    if (root/'TRAINING_DONE.json').exists():
        model.load_state_dict(load(root/'best.pt')['model'])
        return read_json(root/'TRAINING_DONE.json')
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                           lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    start = 0; best = (-math.inf, -math.inf); bad = 0; history = []
    if (root/'last.pt').exists():
        last = load(root/'last.pt'); model.load_state_dict(last['model']); opt.load_state_dict(last['optimizer'])
        start, best, bad, history = last['epoch'], last['best'], last['bad'], last['history']
        torch.set_rng_state(last['rng']); np.random.set_state(last['numpy_rng']); random.setstate(last['python_rng'])
        if device.type=='cuda': torch.cuda.set_rng_state(last['cuda_rng'], device)
        print(f'[resume training] {task}/{root.name} epoch {start}', flush=True)
    steps = cfg['max_epochs']*len(train_dl); warmup = max(1, int(.1*steps))
    for epoch in range(start, cfg['max_epochs']):
        if bad >= cfg['patience']: break
        train_dl.sampler.epoch = epoch; model.train(); loss_sum = torch.zeros((), device=device)
        for step, batch in enumerate(train_dl):
            number = epoch*len(train_dl)+step+1
            factor = number/warmup if number<=warmup else .5*(1+math.cos(math.pi*(number-warmup)/max(1,steps-warmup)))
            for group in opt.param_groups: group['lr'] = cfg['learning_rate']*factor
            b = move(batch, device); opt.zero_grad(set_to_none=True)
            with amp(device):
                logits = model(b, task=task)['logits'].reshape(-1)
                # Exact class balancing already occurs in the sampler.
                loss = F.binary_cross_entropy_with_logits(logits, b['y'])
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step(); loss_sum += loss.detach()
        y, p = predict_loader(model, val_dl, task, device); cut = threshold(y, p)
        score = float(f1(y, p, cut)); clipped = np.clip(p, 1e-7, 1-1e-7)
        val_loss = float(-(y*np.log(clipped)+(1-y)*np.log(1-clipped)).mean())
        record = dict(epoch=epoch+1, train_loss=float(loss_sum)/len(train_dl), val_f1=score,
                      val_loss=val_loss, threshold=cut); history.append(record)
        key = (score, -val_loss)
        if key > best:
            best = key; bad = 0
            save_torch(root/'best.pt', dict(model=model.state_dict(), model_config=asdict(model.config), **record))
        else: bad += 1
        save_torch(root/'last.pt', dict(model=model.state_dict(), optimizer=opt.state_dict(), epoch=epoch+1,
            best=best, bad=bad, history=history, rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
            python_rng=random.getstate(), cuda_rng=torch.cuda.get_rng_state(device) if device.type=='cuda' else None))
        pd.DataFrame(history).to_csv(root/'training.csv', index=False)
        print(f'[epoch {task}/{root.name}] {epoch+1}/{cfg["max_epochs"]} loss={record["train_loss"]:.4f} val_F1={score:.4f} best={best[0]:.4f}', flush=True)
    checkpoint = load(root/'best.pt'); model.load_state_dict(checkpoint['model'])
    info = {k:v for k,v in checkpoint.items() if k not in ('model','model_config')}
    y = train_dl.dataset.master.iloc[train_dl.dataset.indices].y.to_numpy(int)
    info.update(task=task, threshold_source='maximum positive-class F1 on natural validation cohort',
                n_train=len(y), n_positive=int(y.sum()), epoch_size=len(train_dl.sampler),
                oversampling='minority class with replacement to equal counts; training only',
                model_config=asdict(model.config))
    write_json(root/'TRAINING_DONE.json', info)
    (root/'last.pt').unlink(missing_ok=True)
    return info


@torch.no_grad()
def export(model, dl, task, split, device, root, basic_indices=None):
    path = root/f'features_{split}.pt'
    if path.exists(): return load(path)
    model.eval(); collected = {k:[] for k in ('embedding','source_tokens','source_present','original_logits','y')}
    ids = []; masked_embeddings = []; masked_logits = []
    for batch in dl:
        b = move(batch, device); ids.extend(batch['person_id'])
        with amp(device):
            sources = model.encode_sources(b); out = model.predict_from_source_tokens(**sources, task=task)
            if basic_indices is not None:
                values = torch.zeros_like(b['clinical_values']); mask = torch.zeros_like(b['clinical_mask'])
                values[:,basic_indices] = b['clinical_values'][:,basic_indices]
                mask[:,basic_indices] = b['clinical_mask'][:,basic_indices]
                token, present = model.clinical_encoder(values, mask)
                tokens = sources['source_tokens'].clone(); availability = sources['source_present'].clone()
                present = present & availability[:,0]; tokens[:,0] = token*present.unsqueeze(-1); availability[:,0] = present
                masked = model.predict_from_source_tokens(tokens, availability, task=task)
                masked_embeddings.append(masked['embedding'].float().cpu())
                masked_logits.append(masked['logits'].reshape(-1).float().cpu())
        for key in ('embedding','source_tokens'): collected[key].append(out[key].float().cpu())
        collected['source_present'].append(out['source_present'].cpu())
        collected['original_logits'].append(out['logits'].reshape(-1).float().cpu()); collected['y'].append(b['y'].long().cpu())
    result = {k:torch.cat(v) for k,v in collected.items()}; result.update(person_ids=ids, split=split, task=task)
    if masked_logits:
        result['masked_embedding'] = torch.cat(masked_embeddings)
        result['masked_logits'] = torch.cat(masked_logits)
    save_torch(path, result)
    return result


@torch.no_grad()
def source_outputs(model, cache, task, device, batch_size):
    tokens = cache['source_tokens'].to(device); presence = cache['source_present'].to(device); results = {}
    for name, members in REMOVALS.items():
        indices = [MODALITIES.index(g) for g in members]
        z = tokens.clone(); m = presence.clone(); z[:,indices] = 0; m[:,indices] = False
        embeddings, logits = [], []
        # Match export batch shapes so a naturally absent source stays an exact
        # no-op, rather than changing a threshold decision through BF16 rounding.
        for start in range(0, len(z), batch_size):
            with amp(device): out = model.predict_from_source_tokens(z[start:start+batch_size], m[start:start+batch_size], task=task)
            embeddings.append(out['embedding'].float().cpu()); logits.append(out['logits'].reshape(-1).float().cpu())
        results[name] = dict(embedding=torch.cat(embeddings),
                            probability=torch.cat(logits).sigmoid().numpy(),
                            available=presence[:,indices].any(1).cpu().numpy())
    return results


class FrozenFusionReadout(torch.nn.Module):
    """Load only the trained fusion/head for ablation of cached source tokens."""
    def __init__(self, checkpoint, task):
        super().__init__()
        config = Task2V2ModelConfig(**checkpoint['model_config'])
        self.fusion = PresenceAwareFusion(config)
        self.head = BinaryTaskHead(config.d_model, config.dropout)
        state = checkpoint['model']
        for prefix, module in (('fusion.', self.fusion), ('heads.'+task+'.', self.head)):
            module.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)})

    def predict_from_source_tokens(self, tokens, present, task):
        embedding, _ = self.fusion(tokens, present)
        return dict(embedding=embedding, logits=self.head(embedding))


def cached_removals(task, variant, cfg, device, output, source_analysis=None):
    """Reuse existing ablations or compute missing ones without touching the source run."""
    from infosign.common import backbone_dir
    root = backbone_dir(task, cfg)/variant
    model = None
    results = {}
    for split in ('val', 'test'):
        original = root/f'removals_{split}.pt'
        path = output/f'removals_{variant}_{split}.pt'
        previous = source_analysis/f'removals_{variant}_{split}.pt' if source_analysis else None
        if original.exists():
            results[split] = load(original)
        elif path.exists():
            results[split] = load(path)
        elif previous is not None and previous.exists():
            results[split] = load(previous)
        else:
            if model is None:
                model = FrozenFusionReadout(load(root/'best.pt'), task).to(device).eval()
            cache = load(root/f'features_{split}.pt')
            # Preserve the original export batch shape for BF16 inference.
            import yaml
            runtime = root.parent.parent/'runtime.yaml'
            batch_size = yaml.safe_load(runtime.read_text())['batch_size'] if runtime.exists() else cfg['batch_size']
            results[split] = source_outputs(model, cache, task, device, batch_size)
            save_torch(path, results[split])
            print(f'[ablation {task}/{variant}/{split}] {len(REMOVALS)} removals from cached source tokens', flush=True)
    return results


def run_model(task, variant, cfg, device):
    root = Path(cfg['output_root'])/task/variant; root.mkdir(parents=True, exist_ok=True)
    if (root/'DONE.json').exists(): print(f'[reuse] {task}/{variant}', flush=True); return
    source = Path(cfg.get('prepared_inputs_root') or cfg['source_root']); entry = read_json(source/'inputs.json')[task]
    master, values, mask, basic, names, excluded = clinical_inputs(task, cfg, entry)
    if variant=='basic': values=values[:,basic]; mask=mask[:,basic]; names=[names[i] for i in basic]
    write_json(root/'input_columns.json', dict(clinical_features=names, target_columns_removed=excluded,
        clinical_source=entry['clinical_csv'], full_clinical_source_task=entry['full_clinical_source_task']))
    datasets = {s:CachedDataset(source, master, values, mask, task, s) for s in ('train','val','test')}
    dims = read_json(source/'cache/dimensions.json'); dims['clinical_dim'] = len(names)
    # CNN outputs are already cached from the original frozen ImageNet backbone.
    # The inherited unused CNN needs no weights download for fitting on features.
    mc = replace(cache_config(cfg, dims), retinal_weights=None, retinal_weights_path=None, task_names=(task,))
    torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed']); random.seed(cfg['seed'])
    model = CachedModel(mc).to(device)
    # Do not store an unused random CNN in new checkpoints.
    model.retinal_encoder.backbone = torch.nn.Identity()
    train_dl, val_dl = loader(datasets['train'], cfg, training=True), loader(datasets['val'], cfg)
    fit(model, train_dl, val_dl, task, cfg, root, device)
    del train_dl, val_dl
    for split in ('train','val','test'):
        cache = export(model, loader(datasets[split], cfg), task, split, device, root,
                       basic_indices=basic if variant=='full' else None)
        if split!='train':
            path=root/f'removals_{split}.pt'
            if not path.exists(): save_torch(path, source_outputs(model, cache, task, device, cfg['batch_size']))
    write_json(root/'DONE.json', dict(task=task, variant=variant))
