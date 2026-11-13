"""Restore per-frame labels from released tables into a new, empty directory."""
from pathlib import Path
import argparse,csv,gzip,itertools,json,re

ROOT=Path(__file__).resolve().parents[1]
CAUSES=['tailgating','speeding','dangerous_lane_change','standing_shoulder','pedestrian_highway','heavy_rain','heavy_snow','distracted_driving','low_visibility']

def safe_stem(value):
    if not value or value in ['.','..'] or '/' in value or '\\' in value or ':' in value:
        raise ValueError('Invalid label stem')
    return value

def rows(path):
    with gzip.open(path,'rt',encoding='utf8',newline='') as handle:yield from csv.DictReader(handle)

def restore(split,output,limit=None,selected=None):
    source=ROOT/'01_data/annotations';scene=rows(source/f'scene_cause_{split}.csv.gz');chosen={};counts={}
    if selected is not None:scene=(r for r in scene if r['stem'] in selected)
    for row in itertools.islice(scene,limit):
        stem=safe_stem(row['stem']);chosen[stem]=row
        text=str(int(row['scene']))+'\n'
        # Preserve annotation presence even for ignored scene frames.
        if int(row['cause_valid']):text+=','.join(str(int(row['cause_'+name])) for name in CAUSES)+'\n'
        (output/'scene_labels'/split/(stem+'.txt')).write_text(text,encoding='utf8')
        (output/'labels'/split/(stem+'.txt')).touch()
    counts['scene']=len(chosen);types=0
    for row in rows(source/f'type_{split}.csv.gz'):
        stem=safe_stem(row['stem'])
        if stem not in chosen:continue
        (output/'type_labels'/split/(stem+'.txt')).write_text(str(int(row['type']))+'\n',encoding='utf8');types+=1
    counts['type']=types
    written=set();boxes=0
    for stem,group in itertools.groupby(rows(source/f'detection_{split}.csv.gz'),lambda r:r['stem']):
        if stem not in chosen:continue
        safe_stem(stem)
        if stem in written:raise ValueError('Detection rows must be grouped by stem: '+stem)
        written.add(stem);lines=[]
        for row in group:
            cls=int(row['cls']);assert cls in [0,1,2]
            coords=[row[k] for k in ['cx','cy','w','h']];assert all(0<=float(v)<=1 for v in coords)
            lines.append(str(cls)+' '+' '.join(coords));boxes+=1
        (output/'labels'/split/(stem+'.txt')).write_text('\n'.join(lines)+'\n',encoding='utf8')
    counts['boxes']=boxes;lanes=0
    with gzip.open(source/f'lane_{split}.jsonl.gz','rt',encoding='utf8') as h:
        for line in h:
            if not line.strip():continue
            obj=json.loads(line);stem=safe_stem(obj['stem'])
            if stem not in chosen:continue
            (output/'lane_labels'/split/(stem+'.json')).write_text(json.dumps(obj,ensure_ascii=False)+'\n',encoding='utf8');lanes+=1
    counts['lane']=lanes
    assert types==len(chosen),'Missing type rows'
    return counts

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--split',choices=['train','val','both'],default='both');parser.add_argument('--limit',type=int);parser.add_argument('--stems',type=Path,help='Optional newline-separated subset of image stems');args=parser.parse_args()
    output=args.output.resolve()
    if args.limit is not None and args.limit<1:parser.error('--limit must be positive')
    if output.exists() and any(output.iterdir()):parser.error('Output directory must be empty; existing labels are never overwritten.')
    output.mkdir(parents=True,exist_ok=True);splits=['train','val'] if args.split=='both' else [args.split];result={}
    for split in splits:
        for folder in ['scene_labels','type_labels','labels','lane_labels']:(output/folder/split).mkdir(parents=True,exist_ok=True)
        selected={safe_stem(x.strip()) for x in args.stems.read_text(encoding='utf8').splitlines() if x.strip()} if args.stems else None
        result[split]=restore(split,output,args.limit,selected)
    (output/'restoration_manifest.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
