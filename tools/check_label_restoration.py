"""Exercise scene/type/detection/lane restoration against the released source tables."""
from pathlib import Path
import csv,gzip,json,subprocess,sys,tempfile

ROOT=Path(__file__).resolve().parents[1]

def main():
    source=ROOT/'01_data/annotations';selected=[]
    with gzip.open(source/'scene_cause_val.csv.gz','rt',encoding='utf8',newline='') as h:
        reader=csv.DictReader(h)
        for row in reader:
            selected.append(row['stem'])
            if len(selected)==16:break
    lane_source={}
    with gzip.open(source/'lane_val.jsonl.gz','rt',encoding='utf8') as h:
        for line in h:
            obj=json.loads(line);lane_source[obj['stem']]=obj
            if len(lane_source)==2:break
    selected+=list(lane_source);selected=set(selected)
    with tempfile.TemporaryDirectory(prefix='highwayaccident-label-check-') as temp:
        base=Path(temp);ids=base/'stems.txt';ids.write_text('\n'.join(sorted(selected))+'\n',encoding='utf8');output=base/'labels'
        subprocess.run([sys.executable,str(ROOT/'tools/restore_labels.py'),'--split','val','--stems',str(ids),'--output',str(output)],check=True)
        scene_seen=type_seen=boxes=0
        with gzip.open(source/'scene_cause_val.csv.gz','rt',encoding='utf8',newline='') as h:
            for row in csv.DictReader(h):
                if row['stem'] not in selected:continue
                lines=(output/'scene_labels/val'/(row['stem']+'.txt')).read_text(encoding='utf8').splitlines();assert int(lines[0])==int(row['scene'])
                assert (len(lines)==2)==bool(int(row['cause_valid']))
                if len(lines)==2:assert list(map(int,lines[1].split(',')))==[int(v) for k,v in row.items() if k.startswith('cause_') and k!='cause_valid']
                scene_seen+=1
        with gzip.open(source/'type_val.csv.gz','rt',encoding='utf8',newline='') as h:
            for row in csv.DictReader(h):
                if row['stem'] in selected:
                    assert int((output/'type_labels/val'/(row['stem']+'.txt')).read_text())==int(row['type']);type_seen+=1
        expected={s:[] for s in selected}
        with gzip.open(source/'detection_val.csv.gz','rt',encoding='utf8',newline='') as h:
            for row in csv.DictReader(h):
                if row['stem'] in selected:expected[row['stem']].append([float(row[k]) for k in ['cls','cx','cy','w','h']]);boxes+=1
        for stem,rows in expected.items():
            restored=[[float(x) for x in line.split()] for line in (output/'labels/val'/(stem+'.txt')).read_text().splitlines() if line.strip()]
            assert restored==rows
        for stem,obj in lane_source.items():assert json.loads((output/'lane_labels/val'/(stem+'.json')).read_text(encoding='utf8'))==obj
        assert scene_seen==type_seen==len(selected)
        print(f'PASS: {len(selected)} scene/type labels, {boxes} detection boxes and {len(lane_source)} lane JSON objects restored without value changes.')

if __name__=='__main__':main()
