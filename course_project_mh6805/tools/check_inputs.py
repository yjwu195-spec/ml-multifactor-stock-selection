"""Check local cache presence and reference hashes without accessing the network."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    raw = ROOT / 'data/raw'
    stems = ['trade_cal', 'index_weight_000905_SH', 'index_daily_000905_SH', 'stock_basic']
    stems += [f'{kind}_{year}' for year in range(2010, 2026) for kind in ['daily', 'daily_basic', 'adj_factor']]
    missing = [s for s in stems if not any((raw / (s + ext)).is_file() for ext in ['.parquet', '.csv.gz'])]
    for prefix in ['fina_indicator_part_', 'balancesheet_part_']:
        if not any(p.is_file() and p.name.endswith(('.parquet', '.csv.gz')) for p in raw.glob(prefix + '*')):
            missing.append(prefix + '*')
    if missing:
        print('Missing authorized raw inputs / 缺少授权原始输入:')
        for name in missing:
            print('  ' + name)
        print('See docs/DATA_ACCESS.md. No download or synthetic fallback is attempted.')
        return 1
    audit = json.loads((ROOT / 'results/data_audit.json').read_text(encoding='utf-8'))
    changed = []
    matched = 0
    for item in audit['raw_files']:
        path = raw / item['file']
        if not path.is_file():
            continue
        with path.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        if digest != item['sha256']:
            changed.append(path.name)
        else:
            matched += 1
    print(f'Required filename families present. Reference hash matches: {matched}.')
    if changed:
        print('Different provider snapshot; record provenance before comparing results:')
        for name in changed:
            print('  ' + name)
        return 2
    print('Schema, time alignment and research execution still require the full pipeline.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
