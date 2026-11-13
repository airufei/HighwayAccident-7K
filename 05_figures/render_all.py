from pathlib import Path
import os,subprocess,sys
HERE=Path(__file__).resolve().parent
for lang in ['en','zh']:
    env=dict(os.environ,FIG_LANG=lang,FIG_OUT=str(HERE/'output'/lang),PYTHONIOENCODING='utf-8')
    for module in ['diagrams_focus','fig06_type_ceiling','fig07_lead_time','fig10_external','fig13_dada_repair']:
        subprocess.run([sys.executable,str(HERE/(module+'.py'))],env=env,check=True)
subprocess.run([sys.executable,str(HERE/'frozen_readouts.py')],check=True)
