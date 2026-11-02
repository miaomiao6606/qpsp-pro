"""Check the explicit public-file allowlist before committing or publishing."""

import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {
    '.gitignore', 'README.md', 'pyproject.toml',
    'data/README.md', 'data/reactome_fi_score075.tsv.gz',
    'examples/simulate_expression.py',
    'src/qpsp_pro/__init__.py', 'src/qpsp_pro/core.py',
    'src/qpsp_pro/qpsp_pro.py', 'src/qpsp_pro/scd.py',
    'tests/test_smoke.py', 'tools/check_public_files.py',
    '.github/workflows/tests.yml',
}


def main():
    candidates = subprocess.check_output(
        ['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'],
        cwd=ROOT,
    ).decode().split('\0')
    files = {name for name in candidates if name and (ROOT / name).exists()}
    unexpected = sorted(files - ALLOWED)
    if unexpected:
        raise SystemExit('Files need explicit publication review: ' + ', '.join(unexpected))
    for name in files:
        path = ROOT / name
        if path.is_symlink():
            raise SystemExit('Symlinks are not allowed: ' + name)
    network = ROOT / 'data/reactome_fi_score075.tsv.gz'
    digest = hashlib.sha256(network.read_bytes()).hexdigest()
    if digest != NETWORK_SHA256:
        raise SystemExit('Public interaction network changed; review its provenance first')
    print(f'Public-file check passed ({len(files)} files).')
    print('This checks candidate paths and the reference network, not Git history or all file contents.')


NETWORK_SHA256 = '0e3fc8b435920702fb870115e8cb2b2208afa89765f6d4b59ac7f285c76ebef8'

if __name__ == '__main__':
    main()
