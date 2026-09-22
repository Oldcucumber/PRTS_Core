"""Tree-sitter grammar check only. This does not compile, type-check or run Swift."""
import argparse
import hashlib
import json
from pathlib import Path
from tree_sitter import Language,Parser
import tree_sitter_swift
ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    parser=Parser(Language(tree_sitter_swift.language()));rows=[]
    for file in sorted((ROOT/'apple').rglob('*.swift')):
        data=file.read_bytes();tree=parser.parse(data)
        rows.append(dict(file=file.relative_to(ROOT).as_posix(),syntax_ok=not tree.root_node.has_error,
                         sha256=hashlib.sha256(data).hexdigest()))
    result=dict(scope=__doc__,compiled=False,executed=False,files=rows,passed=all(r['syntax_ok'] for r in rows))
    (a.output/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=result['passed'],files=len(rows),compiled=False,executed=False)))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
