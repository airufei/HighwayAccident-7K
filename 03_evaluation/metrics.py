"""Metrics for the released score streams. All percentages use a 0–100 scale."""

import collections

import numpy as np

from sklearn.metrics import average_precision_score, roc_auc_score

def softmax(a):
    a=a.astype(np.float64);e=np.exp(a-a.max(-1,keepdims=True));return e/e.sum(-1,keepdims=True)

def scene_metrics(gt,pred):
    valid=(gt>=0)&(gt<3);cm=np.bincount(3*gt[valid]+pred[valid],minlength=9).reshape(3,3)
    tp=cm.diagonal();n=cm.sum(1);p=cm.sum(0)
    recall=tp/np.maximum(n,1);f1=2*tp/np.maximum(n+p,1)
    return {'n':int(cm.sum()),'confusion':cm.tolist(),'accuracy':100*float(tp.sum()/max(cm.sum(),1)),
        'balanced_accuracy':100*float(recall.mean()),'macro_f1':100*float(f1.mean()),
        'normal_recall':100*float(recall[0]),'pre_crash_recall':100*float(recall[1]),'crash_recall':100*float(recall[2]),
        'far':100*float((cm[0,1]+cm[0,2])/max(n[0],1))}

def type_metrics(gt,logits,scene_cls,mask=False):
    z=logits.copy();cur=np.isin(gt,[0,1,4,5,6]);lab=(gt>=0)&(gt<7)
    if mask:z[:,[2,3]]=-1e9
    pred=z.argmax(1);prob=softmax(z);correct=int((pred[lab]==gt[lab]).sum());alln=int(lab.sum()+((gt<0)&(scene_cls==3)).sum())
    result={'active_n':int(cur.sum()),'active_correct':int((pred[cur]==gt[cur]).sum()),'active_acc':100*float((pred[cur]==gt[cur]).mean()),
        'labeled_n':int(lab.sum()),'labeled_correct':correct,'labeled_acc':100*correct/int(lab.sum()),'candidate_n':alln,'correct_type_yield':100*correct/alln,'per_class':{}}
    for k in range(7):
        positive=(gt[lab]==k);n=int(positive.sum());tp=int(((gt==k)&(pred==k)&lab).sum());pp=int(((pred==k)&lab).sum())
        result['per_class'][str(k)]={'n':n,'correct':tp,'precision':100*tp/max(pp,1),'recall':100*tp/max(n,1),
            'ap_from_logits':100*float(average_precision_score(positive,z[lab,k])) if n else None,
            'ap_from_probability':100*float(average_precision_score(positive,prob[lab,k])) if n else None}
    return result

def family(vid):
    for prefix,name in [('synthetic_','synthetic'),('tum_','tumtraf'),('realv','realv'),('real_ccd_','ccd'),('real_','real'),('tv','video_sampling')]:
        if vid.startswith(prefix):return name
    return 'other'

def archived_rank_ap(y,p):
    if not np.any(y):return None
    order=np.argsort(-np.asarray(p),kind='stable');positive=np.asarray(y)[order]
    return float((np.cumsum(positive)/np.arange(1,len(y)+1))[positive.astype(bool)].sum()/positive.sum())

def summarize(rows):
    gt=[];pred=[];ft=fc=0
    for r in rows:
        s=np.asarray(r['scene_gt_seq']);p=np.asarray(r['frame_scene_prob']).argmax(-1);valid=np.flatnonzero(s)
        ft+=len(valid);fc+=int((p[valid]==s[valid]-1).sum())
        if len(valid):gt.append(int(s[valid[-1]])-1);pred.append(int(p[valid[-1]]))
    out=scene_metrics(np.asarray(gt),np.asarray(pred));out.update(frame_n=ft,frame_correct=fc,frame_accuracy=100*fc/ft,invalid_rows=len(rows)-len(gt))
    ty=np.array([r['type_gt'] for r in rows]);tp=np.array([r['type_prob'] for r in rows]);pr=tp.argmax(-1);lab=(ty>=0)&(ty<7);active=np.isin(ty,[0,1,4,5,6])
    correct=int((pr[lab]==ty[lab]).sum());candidate=int(lab.sum()+sum(r['type_gt']<0 and r['clip_scene_gt']==3 for r in rows))
    out['type']={'labeled_n':int(lab.sum()),'correct':correct,'labeled_accuracy':100*correct/int(lab.sum()) if lab.any() else None,'active_n':int(active.sum()),'active_accuracy':100*int((pr[active]==ty[active]).sum())/int(active.sum()) if active.any() else None,'candidate_n':candidate,'yield':100*correct/candidate if candidate else None,'per_class':{}}
    for k in range(7):
        target=(ty[lab]==k);hits=int(((pr==k)&(ty==k)&lab).sum());pp=int(((pr==k)&lab).sum())
        candidates=np.array([r['clip_scene_gt']==3 for r in rows]);archive_y=(ty[candidates]==k);archive_p=tp[candidates,k]
        out['type']['per_class'][str(k)]={'n':int(target.sum()),'correct':hits,'recall':hits/max(int(target.sum()),1),'precision':hits/max(pp,1),'ap':float(average_precision_score(target,tp[lab,k])) if target.any() else None,'archived_candidate_precision':hits/max(int(((pr==k)&candidates).sum()),1),'archived_candidate_rank_ap':archived_rank_ap(archive_y,archive_p)}
    cr=[r for r in rows if r['cause_valid']>0 and r['clip_scene_gt']>=2]
    if not cr:
        out['cause']={'n':0};return out
    cg=np.asarray([r['cause_gt'] for r in cr]);cp=np.asarray([r['cause_prob'] for r in cr]);bits=[0,1,2,3,5,6,8];target=np.maximum(cg[:,bits],0);pred=cp[:,bits]>=.5
    t=int((pred&(target==1)).sum());f=int((pred&(target==0)).sum());n=int(((~pred)&(target==1)).sum())
    out['cause']={'n':len(cr),'tp':t,'fp':f,'fn':n,'precision':100*t/(t+f),'recall':100*t/(t+n),'f1':100*2*t/(2*t+f+n),'per_bit':{}}
    for k in bits:
        eligible=cg[:,k]>=0;y=cg[eligible,k];p=cp[eligible,k]
        out['cause']['per_bit'][str(k)]={'n':len(y),'positive':int(y.sum()),'ap':float(average_precision_score(y,p)) if y.any() else None,'archived_rank_ap':archived_rank_ap(y,p),'auc':float(roc_auc_score(y,p)) if 0<y.sum()<len(y) else None}
    return out

def cluster_ci(rows,key,draws=2000):
    groups=collections.defaultdict(lambda:np.zeros(16,dtype=int))
    for r in rows:
        s=np.asarray(r['scene_gt_seq']);valid=np.flatnonzero(s)
        ty=int(r['type_gt']);has_type=0<=ty<7;has_cause=r['cause_valid']>0 and r['clip_scene_gt']>=2
        if not len(valid) and not has_type and not has_cause:continue
        a=groups[key(r)]
        if len(valid):
            g=int(s[valid[-1]])-1;p=int(np.argmax(r['frame_scene_prob'][int(valid[-1])]))
            a[3*g+p]+=1
        if has_type:
            correct=int(np.argmax(r['type_prob']))==ty;a[9]+=int(correct);a[10]+=1
            if ty in [0,1,4,5,6]:a[11]+=int(correct);a[12]+=1
        if has_cause:
            bits=[0,1,2,3,5,6,8];cg=np.maximum(np.asarray(r['cause_gt'])[bits],0);cp=np.asarray(r['cause_prob'])[bits]>=.5
            a[13]+=int((cp&(cg==1)).sum());a[14]+=int((cp&(cg==0)).sum());a[15]+=int(((~cp)&(cg==1)).sum())
    arr=np.stack(list(groups.values()));rng=np.random.default_rng(0);stats=collections.defaultdict(list)
    for _ in range(draws):
        total=arr[rng.integers(0,len(arr),len(arr))].sum(0);cm=total[:9].reshape(3,3);tp=cm.diagonal();support=cm.sum(1);pp=cm.sum(0)
        stats['accuracy'].append(100*float(tp.sum()/cm.sum()))
        if (support>0).all():
            stats['balanced_accuracy'].append(100*float((tp/support).mean()))
            stats['macro_f1'].append(100*float((2*tp/(support+pp)).mean()))
        if support[2]>0:stats['crash_recall'].append(100*float(tp[2]/support[2]))
        if support[1]>0:stats['pre_crash_recall'].append(100*float(tp[1]/support[1]))
        if support[0]>0:stats['far'].append(100*float((cm[0,1]+cm[0,2])/support[0]))
        if total[10]:stats['labeled_type_accuracy'].append(100*float(total[9]/total[10]))
        if total[12]:stats['active_type_accuracy'].append(100*float(total[11]/total[12]))
        denom=2*total[13]+total[14]+total[15]
        if denom:stats['cause_f1'].append(100*float(2*total[13]/denom))
    return {'n_clusters':len(groups),'draws':draws,'seed':0,'ci95':{k:np.percentile(v,[2.5,97.5]).tolist() for k,v in stats.items()},'valid_draws':{k:len(v) for k,v in stats.items()}}

def aggregate(rows,channel,decision,fps):
    groups={};cmgt=[];cmpred=[];invalid=0
    for row in rows:
        vid=str(row['video_id']);g=groups.setdefault(vid,{'positive':False,'max_any':-1.0,'max_crash_phase':-1.0,'false_windows':0,'normal_windows':0,'valid_windows':0})
        seq=row['scene_gt_seq'];valid=np.flatnonzero(np.asarray(seq)!=0)
        if not len(valid):invalid+=1;continue
        gt=int(seq[int(valid[-1])]);p=np.asarray(row['clip_scene_prob'] if channel=='clip_head' else row['frame_scene_prob'][int(valid[-1])],dtype=float)
        score=float(np.argmax(p)==2) if decision=='argmax' else float(p[2]);threshold=0.5 if decision=='argmax' else float(decision)
        g['valid_windows']+=1;g['max_any']=max(g['max_any'],score)
        if gt==3:g['positive']=True;g['max_crash_phase']=max(g['max_crash_phase'],score)
        if gt==1:g['normal_windows']+=1;g['false_windows']+=int(score>=threshold)
        cmgt.append(gt-1);cmpred.append(int(np.argmax(p)))
    pos=np.array([g['positive'] for g in groups.values()]);anyscore=np.array([g['max_any'] for g in groups.values()]);crashscore=np.array([g['max_crash_phase'] for g in groups.values()]);false=np.array([g['false_windows'] for g in groups.values()]);normal=np.array([g['normal_windows'] for g in groups.values()]);threshold=0.5 if decision=='argmax' else float(decision)
    hit_any=(anyscore>=threshold)&pos;hit_crash=(crashscore>=threshold)&pos
    rate=float(false.sum()/(normal.sum()*8/fps/3600)) if normal.sum() else None
    result={'n_videos':len(groups),'n_positive_videos':int(pos.sum()),'n_valid_clips':len(cmgt),'n_invalid_clips':invalid,
        'n_positive_videos_hit_anywhere':int(hit_any.sum()),'positive_video_hit_rate_anywhere':100*float(hit_any.sum()/pos.sum()) if pos.sum() else None,
        'n_positive_videos_hit_during_crash':int(hit_crash.sum()),'positive_video_hit_rate_during_crash':100*float(hit_crash.sum()/pos.sum()) if pos.sum() else None,
        'hits_only_outside_crash_phase':int((hit_any&~hit_crash).sum()),'n_normal_windows':int(normal.sum()),'n_false_positive_windows':int(false.sum()),
        'normal_false_crash_window_percent':100*float(false.sum()/normal.sum()) if normal.sum() else None,
        'fps_assumed':fps,'nominal_normal_hours':float(normal.sum()*8/fps/3600),'false_positive_windows_per_nominal_hour':rate,
        'argmax_scene_metrics':scene_metrics(np.array(cmgt),np.array(cmpred))}
    rng=np.random.default_rng(0);ci_any=[];ci_crash=[];ci_window=[]
    for _ in range(1000):
        idx=rng.integers(0,len(groups),len(groups));positive=pos[idx].sum()
        if positive:ci_any.append(100*float(hit_any[idx].sum()/positive));ci_crash.append(100*float(hit_crash[idx].sum()/positive))
        ns=normal[idx].sum()
        if ns:ci_window.append(float(false[idx].sum()/(ns*8/fps/3600)))
    result['ci95']={k:np.percentile(v,[2.5,97.5]).tolist() if v else None for k,v in [('anywhere_video_hit',ci_any),('crash_phase_video_hit',ci_crash),('false_windows_per_nominal_hour',ci_window)]}
    return result
