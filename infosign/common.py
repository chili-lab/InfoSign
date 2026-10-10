"""Study definitions, paths, and small output helpers."""
from pathlib import Path
import json
import torch

from infosign.encoders import PRIMARY_MODALITY_GROUPS as MODALITIES

TASK_LABELS={'moca_binary':'MoCA < 26','cvd_broad_binary':'CVD broad','cvd_strict_binary':'CVD strict',
 'cardiovascular_conditions_binary':'Broad cardiovascular conditions',
 'major_cardiovascular_events_binary':'Major cardiovascular events',
 'osteoporosis_history_binary':'Osteoporosis history','kidney_history_binary':'Kidney-problem history',
 'lung_history_binary':'Lung-problem history'}
FULL_CONFIGS={'moca_binary':'moca','cvd_broad_binary':'cvd_broad','cvd_strict_binary':'cvd_strict'}
BASIC_KEYS={'age','cage','brthyy','years_of_education','pxed1','cl_maristat','pxep1','pxep2','pxep3'}
HISTORY_TARGETS={'osteoporosis_history_binary':('mhoccur_oa',80502),
 'kidney_history_binary':('mhoccur_rnl',2005200017),'lung_history_binary':('mhoccur_plm',4186898)}
REMOVALS={**{x:(x,) for x in MODALITIES},'retinal_all':MODALITIES[3:7],
 'wearable_all':MODALITIES[7:]}
INPUT_LABELS=dict(zip(MODALITIES,('Basic clinical information','Electrocardiography (ECG)',
 'Environmental measurements','Retinal FLIO','Retinal OCT','Retinal OCTA','Retinal photography',
 'Wearable activity monitoring','Wearable blood glucose monitoring')))
INPUT_LABELS.update(retinal_all='All retinal modalities',wearable_all='All wearable modalities')
def read_json(path):return json.loads(Path(path).read_text())
def write_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2,default=str)+'\n');temp.replace(path)
def save_torch(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');torch.save(data,temp);temp.replace(path)
def save_csv(path,frame):
    path=Path(path);temp=path.with_suffix('.csv.tmp');frame.to_csv(temp,index=False);temp.replace(path)
def original_config(task,cfg):
    root=Path(cfg['artifact_root'])
    return {'dataset': {
        'task': task, 'artifact_root': str(root),
        'master_csv': str(root/f'model_full_assets_v1/inputs/master_{task}.csv'),
        'master_metadata_json': str(root/f'model_full_assets_v1/inputs/master_{task}.metadata.json'),
        'scaler_json': str(root/f'scalers/train_scalers_{task}.json'),
        'clinical_schema_json': str(root/f'clinical/{task}/clinical_schema_{task}.json'),
        'environment_schema_json': str(root/'environment_v4/environment_5min_schema.json'),
        'wearable_schema_json': str(root/'wearable_v5/wearable_5min_schema.json'),
        'ecg_schema_json': str(root/'ecg/ecg_schema.json'),
        'strict_lineage': True, 'allow_missing_modalities': True,
        'retinal': {'deterministic_selection': True,
                    'max_assets': {'photography':6, 'oct':2, 'octa':16, 'flio':4}}
    }}

def backbone_dir(task,cfg):
    """Read frozen multimodal outputs from this run or an explicitly reused run."""
    return Path(cfg.get("backbone_root") or cfg["output_root"])/task
