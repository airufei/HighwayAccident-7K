"""Verify released file hashes, Python syntax, and absence of credential literals."""
from pathlib import Path
import argparse,ast,hashlib,re

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--skip-models',action='store_true',help='Permit a checkout with model LFS pointers instead of downloaded tensors');args=p.parse_args()
    checked=skipped=0;failures=[]
    for line in (ROOT/'SHA256SUMS.txt').read_text(encoding='utf8').splitlines():
        if not line.strip():continue
        expected,name=line.split(None,1);path=ROOT/name.strip().removeprefix('./')
        assert path.resolve().is_relative_to(ROOT)
        if args.skip_models and path.suffix=='.pt':skipped+=1;continue
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:failures.append(name)
        else:checked+=1
        if path.suffix=='.py':ast.parse(path.read_text(encoding='utf-8-sig'),filename=name)
        if path.suffix.lower() in ['.py','.md','.json','.yaml','.yml','.txt','.cff']:
            text=path.read_text(encoding='utf-8-sig')
            for pattern in [r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',r'\bgh[pousr]_[A-Za-z0-9]{25,}\b',r'\bgithub_pat_[A-Za-z0-9_]{30,}\b',r'\bsk-[A-Za-z0-9_-]{30,}\b']:
                if re.search(pattern,text):failures.append(name+' [credential-pattern]')
    if failures:raise SystemExit('Release verification failed: '+', '.join(failures)+'; use git lfs pull if model files are pointers.')
    print(f'PASS: {checked} file hashes; {skipped} model files skipped; Python syntax and credential-literal checks passed.')

if __name__=='__main__':main()
