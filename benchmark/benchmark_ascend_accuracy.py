"""Ascend accuracy probe using SciPy/NumPy references."""
from __future__ import annotations
import argparse, json
from benchmark_ascend import Case, run


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--op', required=True)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--output', required=True)
    # f16 is a delivery dtype for gather/scatter; without it those variants read NF.
    p.add_argument('--dtypes', default='float16,float32,float64')
    args = p.parse_args()
    payload = {}
    for dtype in (x.strip() for x in args.dtypes.split(',') if x.strip()):
        items = run(Case(64, 96, 256, 16, dtype), 0, 1, device_id=args.device)
        item = next((x for x in items if x.get('op') == args.op and x.get('dtype') == dtype), None)
        key = f'{args.op}[{dtype}]'
        if item is None:
            payload[key] = {'result': 'failed', 'reason': 'operator record was not produced'}
            continue
        err = (item.get('scipy_max_abs_error') or {}).get('flagsparse')
        status = str(item.get('status') or '')
        passed = err is not None and status.startswith('') and not status.startswith('FlagSparse:')
        if passed:
            tol = 2e-2 if dtype in ('float16', 'bfloat16', 'float32') else 1e-8
            passed = float(err) <= tol
        payload[key] = {
            'params': {'dtype': dtype, 'scipy_max_abs_err': err},
            'result': 'passed' if passed else 'failed',
            'reason': None if passed else (status or f'SciPy max abs error {err} exceeds tolerance'),
            'opname': [args.op],
        }
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == '__main__':
    main()
