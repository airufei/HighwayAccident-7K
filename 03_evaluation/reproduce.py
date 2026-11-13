"""Recompute the released score-based results without raw images or a GPU."""
from pathlib import Path
import argparse,collections,csv,gzip,hashlib,json
import numpy as np
from metrics import aggregate,cluster_ci,family,scene_metrics,summarize,type_metrics

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'03_evaluation/data'

def load_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def load_scores(name):
    with gzip.open(DATA/f'scores/{name}.jsonl.gz','rt',encoding='utf8') as h:
        return [json.loads(line) for line in h if line.strip()]

def final_metrics(rows,draws):
    event_map={}
    with gzip.open(ROOT/'01_data/manifests/video_split_map.csv.gz','rt',encoding='utf8',newline='') as h:
        for row in csv.DictReader(h):
            if row['splits']=='val':
                assert row['video_id'] not in event_map or event_map[row['video_id']]==row['event_id']
                event_map[row['video_id']]=row['event_id']
    for row in rows:row['parsed_event_id']=event_map[row['video_id']]
    exclusions=json.loads((ROOT/'01_data/manifests/evaluation_exclusions.json').read_text(encoding='utf8'))
    kept=[r for r in rows if r['parsed_event_id'] not in exclusions['event_ids']]
    result={'original':summarize(rows),'after_two_exclusions':summarize(kept),'cluster_statistics':{}}
    for name,subset in [('original',rows),('after_two_exclusions',kept),('non_carla_after_two_exclusions',[r for r in kept if family(r['video_id'])!='synthetic'])]:
        result['cluster_statistics'][name]={'point':summarize(subset),'video_id':cluster_ci(subset,lambda r:r['video_id'],draws),'parsed_event_id':cluster_ci(subset,lambda r:r['parsed_event_id'],draws)}
    result['source_after_exclusions']={f:summarize([r for r in kept if family(r['video_id'])==f]) for f in sorted({family(r['video_id']) for r in kept})}
    return result

def control_metrics():
    va=load_npz(DATA/'labels/val.npz');seq=va['scene_seq'];last=np.max(np.where(seq!=0,np.arange(16),-1),1)
    gt=np.where(last>=0,seq[np.arange(len(last)),last.clip(0)]-1,-1)
    exclude=['real_kaggle-fahaddalwai-cctvfootagevideo_5','real_ccd_001397']
    keep=np.array([not any(e in v for e in exclude) for v in va['video_id']]);keep3=np.array([not any(e in v for e in exclude+['real_gZdaWZGFUs4']) for v in va['video_id']])
    supplied=json.loads((ROOT/'03_evaluation/reference/fair_baselines_curriculum.json').read_text(encoding='utf8'))
    out={};bits=[0,1,2,3,5,6,8]
    for arm in ['last','mean','single_mid','gru']:
        d=load_npz(DATA/f'outputs/{arm}_curriculum.npz');fp=d['frame_scene'].argmax(-1)[np.arange(len(last)),last.clip(0)];cp=d['clip_scene'].argmax(-1)
        row={'reported_frame_head_endpoint':scene_metrics(gt,fp),'actual_clip_head_endpoint':scene_metrics(gt,cp),
             'clip_head_after_two_exclusions':scene_metrics(gt[keep],cp[keep]),'clip_head_after_three_exclusions':scene_metrics(gt[keep3],cp[keep3]),
             'frame_head_after_two_exclusions':scene_metrics(gt[keep],fp[keep]),'n_clip_head_predictions_differ_from_reported_endpoint':int(((fp!=cp)&(last>=0)).sum()),
             'type_raw':type_metrics(va['accident_type'],d['clip_type'],va['scene_cls'],False),'type_masked_as_supplied_script':type_metrics(va['accident_type'],d['clip_type'],va['scene_cls'],True)}
        row['scene_summary_matches']={k:round(row['reported_frame_head_endpoint'][k],2)==supplied[arm]['scene'][v] for k,v in [('accuracy','scene_acc'),('balanced_accuracy','scene_bal_acc'),('macro_f1','scene_macro_f1'),('crash_recall','crash_recall'),('pre_crash_recall','pre_crash_recall'),('far','FAR')]}
        valid=(va['cause_valid']>0)&(va['scene_cls']>=2);y=va['cause'][valid][:,bits];p=d['clip_cause'][valid][:,bits]>=0
        tp=int((p&(y==1)).sum());fpn=int((p&(y==0)).sum());fn=int(((~p)&(y==1)).sum())
        row['cause']={'n':int(valid.sum()),'tp':tp,'fp':fpn,'fn':fn,'precision':100*tp/(tp+fpn),'recall':100*tp/(tp+fn),'f1':100*2*tp/(2*tp+fpn+fn)}
        out[arm]=row
    d=load_npz(DATA/'outputs/gru_all.npz')
    seven={'type_raw':type_metrics(va['accident_type'],d['clip_type'],va['scene_cls']),'clip_head_endpoint':scene_metrics(gt,d['clip_scene'].argmax(1)),
           'reported_frame_head_endpoint':scene_metrics(gt,d['frame_scene'].argmax(-1)[np.arange(len(last)),last.clip(0)])}
    return out,seven

def split_counts():
    fam=collections.defaultdict(lambda:{'frames':collections.Counter(),'videos':collections.defaultdict(set),'events':collections.defaultdict(set)})
    videos={};events={}
    with gzip.open(ROOT/'01_data/manifests/frame_split_map.csv.gz','rt',encoding='utf8',newline='') as h:
        for row in csv.DictReader(h):
            f,s,v,e=[row[k] for k in ['family','split','video_id','event_id']]
            fam[f]['frames'][s]+=1;fam[f]['videos'][s].add(v);fam[f]['events'][s].add(e)
            videos.setdefault((f,v),{'splits':set(),'n':0,'events':set()});videos[(f,v)]['splits'].add(s);videos[(f,v)]['n']+=1;videos[(f,v)]['events'].add(e)
            events.setdefault((f,e),{'splits':set(),'n':0,'videos':set()});events[(f,e)]['splits'].add(s);events[(f,e)]['n']+=1;events[(f,e)]['videos'].add(v)
    counts={f:{'frames_train':d['frames']['train'],'frames_val':d['frames']['val'],'videos_train':len(d['videos']['train']),'videos_val':len(d['videos']['val']),
               'events_train':len(d['events']['train']),'events_val':len(d['events']['val']),'videos_cross':len(d['videos']['train']&d['videos']['val']),'events_cross':len(d['events']['train']&d['events']['val'])} for f,d in fam.items()}
    mismatches=[]
    for kind,groups in [('video',videos),('event',events)]:
        with gzip.open(ROOT/f'01_data/manifests/{kind}_split_map.csv.gz','rt',encoding='utf8',newline='') as h:
            for row in csv.DictReader(h):
                key=(row['family'],row[kind+'_id']);d=groups[key]
                mismatch=set(row['splits'].split('|'))!=d['splits'] or int(row['n_frames'])!=d['n']
                mismatch|=(row['event_id'] not in d['events']) if kind=='video' else int(row['n_videos'])!=len(d['videos'])
                if mismatch:mismatches.append(row[kind+'_id'])
    strict={'real_kaggle-fahaddalwai-cctvfootagevideo_5','real_ccd_001397'}
    flagged={e:{'family':f,'splits':sorted(d['splits']),'frames':d['n'],'n_videos':len(d['videos'])} for (f,e),d in events.items() if e in strict}
    return {'families':counts,'n_video_identifiers':len(videos),'n_event_identifiers':len(events),'map_mismatches':mismatches,'flagged_events':flagged}

def assert_close(actual,expected,path='results'):
    if isinstance(expected,dict):
        assert isinstance(actual,dict) and actual.keys()==expected.keys(),path+' keys'
        for key in expected:assert_close(actual[key],expected[key],path+'.'+str(key))
    elif isinstance(expected,list):
        assert len(actual)==len(expected),path+' length'
        for i,(a,b) in enumerate(zip(actual,expected)):assert_close(a,b,path+f'[{i}]')
    elif isinstance(expected,(int,float)) and not isinstance(expected,bool):
        assert np.isclose(actual,expected,atol=1e-9,rtol=1e-11),f'{path}: {actual} != {expected}'
    else:assert actual==expected,f'{path}: {actual} != {expected}'

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true');parser.add_argument('--output',type=Path,default=ROOT/'04_results/recomputed');args=parser.parse_args()
    identity=json.loads((DATA/'score_identity.json').read_text(encoding='utf8'))
    for name,meta in identity.items():
        with gzip.open(DATA/f'scores/{name}.jsonl.gz','rb') as h:assert hashlib.sha256(h.read()).hexdigest()==meta['sha256_uncompressed']
    print('Recomputing final checkpoint and clustered intervals...',flush=True)
    rows=load_scores('final_epoch10');va=load_npz(DATA/'labels/val.npz')
    assert len(rows)==5556==len(va['video_id'])
    for i,r in enumerate(rows):assert r['video_id']==va['video_id'][i] and r['scene_gt_seq']==va['scene_seq'][i].tolist() and r['type_gt']==int(va['accident_type'][i])
    result={'final':final_metrics(rows,2000)}
    print('Recomputing frozen-feature controls and split maps...',flush=True)
    result['controls'],result['seven_class_control']=control_metrics();result['split_map_recount']=split_counts();result['external']={}
    for name in ['DADA2000','CCD']:
        print('Recomputing '+name+' video aggregation...',flush=True);rows=load_scores(name)
        original=json.loads((ROOT/f'03_evaluation/reference/{name}_aggregation.json').read_text(encoding='utf8'));item={'n_rows':len(rows),'channels':{}}
        for channel in ['clip_head','last_valid_frame_head']:
            item['channels'][channel]={}
            for threshold in ['0.5','0.8','0.9','argmax']:
                r=aggregate(rows,channel,threshold,float(original['fps_assumed']))
                if channel=='clip_head':
                    s=original['variants'][threshold]
                    r['supplied_count_match']={k:r[a]==s[b] for k,a,b in [('positive_videos','n_positive_videos','n_positive_events'),('detected_videos','n_positive_videos_hit_anywhere','n_detected_events'),('false_windows','n_false_positive_windows','false_alerts'),('normal_windows','n_normal_windows','normal_clips')]}
                    r['supplied_ci_match']={k:([round(v,2) for v in r['ci95'][a]]==s['ci'][b] if r['ci95'][a] is not None else s['ci'][b] is None) for k,a,b in [('hit','anywhere_video_hit','event_recall_ci95'),('windows','false_windows_per_nominal_hour','fa_per_hour_ci95')]}
                item['channels'][channel][threshold]=r
        result['external'][name]=item
    if args.check:
        expected=json.loads((ROOT/'03_evaluation/reference/verified_metrics.json').read_text(encoding='utf8'));assert_close(result,expected)
        print('PASS: all released point estimates, counts, AP definitions, split counts and cluster intervals match the verified reference.',flush=True)
    args.output.mkdir(parents=True,exist_ok=True);(args.output/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')

if __name__=='__main__':main()
