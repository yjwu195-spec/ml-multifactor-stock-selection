"""Offline checks; does not access Tushare or claim that a backtest ran."""
from pathlib import Path
import ast
import json
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    failures = []
    for path in ROOT.rglob('*'):
        if not path.is_file() or any(p in {'.git', '.venv', '__pycache__', 'data', 'outputs'} for p in path.relative_to(ROOT).parts):
            continue
        if path.name == '.env':
            continue
        if path.suffix in {'.docx', '.pptx'}:
            with zipfile.ZipFile(path) as archive:
                text = '\n'.join(archive.read(n).decode('utf-8', errors='replace') for n in archive.namelist() if n.endswith('.xml'))
        elif path.suffix in {'.py', '.md', '.yaml', '.yml', '.json', '.txt', '.html', '.csv'} or path.name == '.env.example':
            text = path.read_text(encoding='utf-8-sig')
        else:
            continue
        if path.suffix == '.py':
            ast.parse(text, filename=str(path))
        if re.search(r'(?i)(?<![0-9a-f])[0-9a-f]{56}(?![0-9a-f])', text):
            failures.append(f'Possible provider credential: {path.relative_to(ROOT)}')
        if path.suffix in {'.py', '.yaml'} and re.search(r'[A-Za-z]:[\\/]', text):
            failures.append(f'Absolute workstation path: {path.relative_to(ROOT)}')
    for relative in ['README.md', 'README.zh-CN.md', 'requirements.txt', '.env.example', '.gitignore', 'config/config.yaml', 'config/config_improved.yaml', 'run_research.py']:
        if not (ROOT / relative).is_file():
            failures.append(f'Missing {relative}')
    if failures:
        print('\n'.join(failures))
        sys.exit(1)
    print(json.dumps({'status': 'PASS', 'checks': ['python_syntax', 'credential_patterns_in_text_and_office_xml', 'portable_source_paths', 'required_files'], 'full_backtest_executed': False}, indent=2))


if __name__ == '__main__':
    main()
