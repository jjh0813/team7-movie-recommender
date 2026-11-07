"""Download official archives without extracting untrusted archive paths."""
from pathlib import Path
import hashlib
import json
import urllib.request
import zipfile
import sys
import subprocess
import urllib.error

ROOT = Path(__file__).resolve().parent
SOURCES = {
    'ml-32m.zip': 'https://files.grouplens.org/datasets/movielens/ml-32m.zip',
    'ml_belief_2024_data_release_2.zip': 'https://files.grouplens.org/datasets/movielens/ml_belief_2024_data_release_2.zip',
}

def download():
    raw = ROOT / 'data' / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, url in SOURCES.items():
        dest = raw / name
        if not dest.exists():
            print('Downloading', name, flush=True)
            temp = dest.with_suffix('.partial')
            try:
                with urllib.request.urlopen(url, timeout=120) as source, temp.open('wb') as out:
                    while block := source.read(1024 * 1024):out.write(block)
            except urllib.error.URLError:
                if sys.platform!='win32':raise
                # Windows certificate-store fallback; TLS verification remains enabled.
                subprocess.run(['powershell','-NoProfile','-File',str(ROOT/'download_windows.ps1'),
                                '-SourceUrl',url,'-Destination',str(temp)],check=True)
            if not zipfile.is_zipfile(temp):
                raise ValueError(f'Invalid ZIP: {temp}')
            temp.replace(dest)
        digest = hashlib.sha256()
        with dest.open('rb') as source:
            while block := source.read(1024 * 1024): digest.update(block)
        with zipfile.ZipFile(dest) as archive:
            manifest[name] = {'url': url, 'bytes': dest.stat().st_size,
                              'sha256': digest.hexdigest(), 'members': archive.namelist()}
        print('Ready', name, dest.stat().st_size, flush=True)
    (raw / 'download_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

if __name__ == '__main__': download()
