"""Run one predeclared slate on 2016-2021; no automatic selection or validation."""
import argparse
import json
from pathlib import Path
import sys
import math
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from tfcta.research import study
from tfcta.research import context
from tfcta import config as C


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--specs',default=str(C.CONFIG_DIR/'research_candidates.json'))
    parser.add_argument('--slippage-ticks',type=float,default=1.)
    parser.add_argument('--skip-ic',action='store_true')
    args = parser.parse_args()
    if not math.isfinite(args.slippage_ticks) or args.slippage_ticks<0:
        parser.error('slippage must be non-negative')
    specs = json.loads(Path(args.specs).read_text())['candidates']
    study.validate_specs(specs)
    run = context.run_dir('research_v3')
    context.dump_json(run/'definition.json',study.provenance(specs,args.slippage_ticks))
    print(f'Loading research data. Output: {run}',flush=True)
    data = study.load_research(args.slippage_ticks)
    rows,daily,phases = [],{},[]
    for spec in specs:
        result = study.run_spec(data,spec)
        rows.extend(study.performance_rows(result,spec))
        daily[spec['id']] = result.net
        result.to_csv(run/f"{spec['id']}_daily.csv")
        if spec.get('mode','single') == 'single' and spec.get('days',1)>1:
            for phase in range(spec['days']):
                shifted = {**spec,'phase':phase}
                alternate = result if phase==spec.get('phase',0) else study.run_spec(data,shifted)
                row = study.performance_rows(alternate,shifted)[0]
                row['phase'] = phase
                phases.append(row)
        print(f"{spec['id']}: complete",flush=True)
    pd.DataFrame(rows).to_csv(run/'performance.csv',index=False)
    pd.DataFrame(phases).to_csv(run/'phase_sensitivity.csv',index=False)
    pd.DataFrame(daily).corr().to_csv(run/'correlations.csv')
    if not args.skip_ic:
        print('Raw-factor diagnostics across 1/3/5/10-day horizons',flush=True)
        study.factor_diagnostics(data).to_csv(run/'factor_diagnostics.csv',index=False)
    print(f'Research complete: {run}',flush=True)


if __name__=='__main__':
    main()
