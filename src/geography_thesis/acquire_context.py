"""Freeze official V-Dem release; never modify inherited data."""
from pathlib import Path
import requests,hashlib,json,base64
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/raw/geography'; OUT.mkdir(parents=True,exist_ok=True)
s=requests.Session();s.mount('https://',requests.adapters.HTTPAdapter(max_retries=3))
records=[]
for file in ['vdem.RData','codebook.RData']:
    url=f'https://api.github.com/repos/vdeminstitute/vdemdata/contents/data/{file}?ref=V16'
    r=s.get(url,timeout=120);r.raise_for_status();meta=r.json()
    p=OUT/file
    if not p.exists():
        try:
            b=s.get(meta['download_url'],timeout=180);b.raise_for_status();data=b.content
        except requests.RequestException:
            b=s.get(meta['git_url'],timeout=180);b.raise_for_status();data=base64.b64decode(b.json()['content'])
        assert len(data)==meta['size'];p.write_bytes(data)
    data=p.read_bytes();assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==meta['sha']
    records.append(dict(path=p.relative_to(ROOT).as_posix(),url=url,git_blob_sha=meta['sha'],sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)))
    print('VERIFIED',file,len(data),flush=True)
release=s.get('https://api.github.com/repos/vdeminstitute/vdemdata/releases/tags/V16',timeout=60);release.raise_for_status();(OUT/'release.json').write_text(json.dumps(release.json(),indent=2))
(OUT/'provenance.json').write_text(json.dumps({'retrieved':'2026-11-22','tag':'V16','files':records},indent=2))
