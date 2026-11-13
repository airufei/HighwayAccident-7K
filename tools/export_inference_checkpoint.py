"""Export the unchanged model tensors, omitting optimizer and private run metadata."""
from pathlib import Path
import argparse,hashlib,json
import torch

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():p.error('Output already exists')
    ckpt=torch.load(args.source,map_location='cpu',weights_only=True);state=ckpt['model_state_dict']
    assert state and all(isinstance(v,torch.Tensor) for v in state.values())
    epoch=int(ckpt['epoch']);export={'epoch':epoch,'phase':str(ckpt.get('phase','3c')),'model_state_dict':state}
    args.output.parent.mkdir(parents=True,exist_ok=True);torch.save(export,args.output)
    restored=torch.load(args.output,map_location='cpu',weights_only=True)['model_state_dict']
    assert state.keys()==restored.keys() and all(state[k].dtype==restored[k].dtype and torch.equal(state[k],restored[k]) for k in state)
    metadata={'epoch':epoch,'tensor_keys':len(state),'all_model_tensors_unchanged':True,'original_sha256':hashlib.sha256(args.source.read_bytes()).hexdigest(),'export_sha256':hashlib.sha256(args.output.read_bytes()).hexdigest(),'export_bytes':args.output.stat().st_size,'omitted':['optimizer_state_dict','scheduler_state_dict','scaler_state_dict','private_run_metadata'],'license':'AGPL-3.0-only'}
    args.output.with_suffix('.identity.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf8');print(json.dumps(metadata,indent=2))

if __name__=='__main__':main()
