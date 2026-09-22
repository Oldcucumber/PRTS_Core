"""Reproduce the bounded public replay subset; no private media is downloaded."""
import argparse,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DEMO=[
 ('0xCqEk5hjEvrygxu26MZkieSv45D_gaJ',45,4),
 ('0zEhKDk1j7KSuUQYR_rmCLmlbb5FCYG6',45,4),
 ('1sftEbnzzIfYBdODrjDO9TbhcnsyKjAv',45,4),
 ('28tdwxgz-zPU06lpeDY3OPWTFxZyBv2c',20,8),
 ('-5OCPnbrwJdu3jH70ieU7pUiFsOJQoeG',45,4),
 ('30S_d-kuvDkznn3Rhea0G3FoBQc5rXoA',30,5)]
EXPERIMENT=[
 '3fNdc-1R7QcLCPHG7Zn9oQXrPqKQbuys','5z7n1PIsHGyQmO64hAlMynF3Cdj0rBeu',
 '8GbY4HVFE794yBRg8-wMqeDi_uLHjQPX','ALYuoq-pWdqtJ-Jm9R8ceVzhsXYawzfi',
 'AMWsWL1lxlW0NelUU8w8yL4psb1R1vCd','AdtGuq73TlhK5miUkWcUO9BoKdnmV-8j']

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--include-training',action='store_true');args=ap.parse_args()
    cases=DEMO+([(sid,24,10) for sid in EXPERIMENT] if args.include_training else [])
    for sid,count,stride in cases:
        subprocess.run([sys.executable,str(ROOT/'scripts/sanpo_samples.py'),'--session='+sid,
            '--count',str(count),'--stride',str(stride)],cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/prepare_demo_cases.py'),'--skip-synthetic'],cwd=ROOT,check=True)

if __name__=='__main__':main()
