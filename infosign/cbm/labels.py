"""Extract the approved task subsets from the supplied clinical CSVs only.

An item is an ID/source pair, further separated when qualifier or unit differs.
No form pooling, disease inference from missing records, or clinical cutoffs.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

BANK = Path(__file__).with_name('concepts.json')
FOUR_OBSERVATION_FORMS = (
    'px_food_insecurity_survey',
    'px_neighborhood_environment_survey',
    'px_racial_ethnic_discrimination_survey',
    'px_sdoh_combined_survey',
)
SURVEY_ADMIN_IDS = ('4160030', '2005200102', '2005200558', '2005200559')
TABLES = {'observation.csv': 'observation', 'measurement.csv': 'measurement',
          'condition_occurrence.csv': 'condition', 'procedure_occurrence.csv': 'procedure'}


def norm(value):
    value = str(value or '').strip()
    if value.lower() == 'nan': return ''
    try:
        n = float(value)
        return format(n, '.15g') if n == n and abs(n) != float('inf') else value
    except ValueError:
        return value


def blocked(field, codes, policy):
    return (field in policy['excluded_source_fields']
            or any(p in field for p in policy['excluded_source_patterns'])
            or bool(codes.intersection(policy['excluded_concept_ids'])))


def recorded_value(row):
    for column in ('value_as_number', 'value_as_string', 'value_source_value'):
        value = str(row.get(column, '') or '').strip()
        if value and value.lower() != 'nan':
            return norm(value), column
    cid = norm(row.get('value_as_concept_id'))
    if cid and cid != '0':
        return 'concept_id:' + cid, 'value_as_concept_id'
    return None, ''


def load_bank(bank_path=BANK, observation_forms=None, excluded_ids=()):
    """Restrict concept targets while retaining each task's outcome exclusions."""
    bank = json.loads(Path(bank_path).read_text())
    if observation_forms is not None:
        bank['observation_forms'] = list(observation_forms)
    if excluded_ids:
        bank['excluded_inventory_ids'] = list(excluded_ids)
    if observation_forms is not None or excluded_ids:
        bank['concepts'] = {k: c for k, c in bank['concepts'].items()
                            if c['concept_id'] not in excluded_ids
                            and (observation_forms is None or c['domain'] != 'Observation'
                                 or c['form_name'] in observation_forms)}
        for policy in bank['tasks'].values():
            policy['concept_keys'] = [k for k in policy['concept_keys'] if k in bank['concepts']]
            policy['inventory_ids'] = len({bank['concepts'][k]['concept_id'] for k in policy['concept_keys']})
    return bank


def extract(clinical_dir, output, bank_path=BANK, observation_forms=None, excluded_ids=(), tasks=None):
    """Read each local table once; retain source values and task-specific masks."""
    clinical_dir, output = Path(clinical_dir), Path(output)
    bank = load_bank(bank_path, observation_forms, excluded_ids)
    if tasks is not None:
        bank['tasks'] = {task: bank['tasks'][task] for task in tasks}
    with (clinical_dir / 'person.csv').open(newline='', encoding='utf-8-sig') as f:
        people = sorted({norm(r['person_id']) for r in csv.DictReader(f)})
    cohort = set(people)
    definitions = bank['concepts']
    by_pair = {(c['concept_id'], c['source_field']): key for key, c in definitions.items()}
    ids = {cid for cid, _ in by_pair}
    allowed = {t: set(p['concept_keys']) for t, p in bank['tasks'].items()}
    records = {t: defaultdict(dict) for t in allowed}
    details = {}
    for filename, prefix in TABLES.items():
        with (clinical_dir / filename).open(newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            code_cols = [c for c in reader.fieldnames if c.endswith('_concept_id')]
            for row in reader:
                pid = norm(row['person_id'])
                if pid not in cohort:
                    continue
                field = row[prefix + '_source_value'].split(',', 1)[0].strip().lower()
                codes = {norm(row[c]) for c in code_cols}
                matches = [(cid, by_pair[(cid, field)]) for cid in codes & ids if (cid, field) in by_pair]
                if not matches:
                    continue
                value, value_column = recorded_value(row)
                # Occurrence files do not contain the questionnaire response.
                # A positive record is not an observed "No" for everyone else.
                if value is None and prefix in ('condition', 'procedure'):
                    value, value_column = 'recorded_occurrence', 'occurrence_presence'
                priority = 0 if value_column == 'occurrence_presence' else 1
                tasks = [t for t, p in bank['tasks'].items() if not blocked(field, codes, p)]
                for cid, base in matches:
                    qualifier = norm(row.get('qualifier_concept_id'))
                    if qualifier == '0': qualifier = ''
                    if prefix == 'condition': qualifier = cid
                    unit = norm(row.get('unit_source_value', ''))
                    if not unit:
                        unit_id = norm(row.get('unit_concept_id'))
                        unit = 'unit_id:' + unit_id if unit_id not in ('', '0') else ''
                    identity = (base, qualifier, unit)
                    if identity not in details:
                        details[identity] = {**definitions[base], 'qualifier_concept_id': qualifier,
                                             'unit': unit, 'value_columns': set(), 'tables': set()}
                    details[identity]['value_columns'].add(value_column or 'no_recorded_value')
                    details[identity]['tables'].add(filename)
                    for task in tasks:
                        if base not in allowed[task]: continue
                        previous = records[task][identity].get(pid)
                        if previous is None or priority > previous[0]:
                            records[task][identity][pid] = (priority, {value} if value is not None else set())
                        elif priority == previous[0] and value is not None:
                            previous[1].add(value)
        print('[concept labels] read ' + filename, flush=True)

    output.mkdir(parents=True, exist_ok=True)
    for task, policy in bank['tasks'].items():
        folder = output / task
        folder.mkdir(exist_ok=True)
        items, columns, counts = [], {}, []
        for base in policy['concept_keys']:
            contexts = sorted(k for k in details if k[0] == base and k in records[task])
            if not contexts: contexts = [(base, '', '')]
            for context in contexts:
                d = details.get(context, {**definitions[base], 'qualifier_concept_id': '', 'unit': '',
                                          'tables': set(), 'value_columns': set()})
                name = base + (f'__q{context[1]}__u{context[2]}' if len(contexts) > 1 else '')
                entries = records[task].get(context, {})
                # A participant with conflicting repeated answers has no single
                # measured target; retain the item and report the conflict count.
                values = {p: next(iter(v)) for p, (_, v) in entries.items() if len(v) == 1}
                item = {**d, 'name': name, 'base_key': base,
                        'tables': sorted(d['tables']), 'value_columns': sorted(d['value_columns'])}
                items.append(item); columns[name] = values
                counts.append({'concept': name, 'concept_id': item['concept_id'],
                    'source_field': item['source_field'], 'n_recorded': len(entries),
                    'n_label': len(values), 'n_conflicting': sum(len(v) > 1 for _, v in entries.values()),
                    'n_empty': sum(not v for _, v in entries.values())})
        with (folder / 'labels.csv').open('w', newline='') as f:
            writer = csv.writer(f); writer.writerow(['person_id', *columns])
            for p in people: writer.writerow([p, *[v.get(p, '') for v in columns.values()]])
        with (folder / 'coverage.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(counts[0])); writer.writeheader(); writer.writerows(counts)
        schema = {'source_workbook': bank['source_workbook'], 'task': task, 'cohort_size': len(people),
                  'inventory_ids': policy['inventory_ids'], 'source_pairs': len(policy['concept_keys']),
                  'observation_forms': bank.get('observation_forms'),
                  'excluded_inventory_ids': bank.get('excluded_inventory_ids', []),
                  'concepts': items, 'exclusions': {k: v for k, v in policy.items() if k.startswith('excluded_')}}
        (folder / 'schema.json').write_text(json.dumps(schema, indent=2) + '\n')
        print(f'[concept labels] {task}: {policy["inventory_ids"]} IDs, {len(items)} individual items', flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clinical-dir', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--bank', type=Path, default=BANK)
    parser.add_argument('--four-surveys', action='store_true', help='Restrict Observation to the four approved surveys')
    parser.add_argument('--drop-survey-admin', action='store_true', help='Exclude the four specified survey date, timestamp and metadata IDs')
    args = parser.parse_args()
    extract(args.clinical_dir, args.out_dir, args.bank,
            FOUR_OBSERVATION_FORMS if args.four_surveys else None,
            SURVEY_ADMIN_IDS if args.drop_survey_admin else ())
