"""研究评估层。日收益、时序 IC、绩效、等权回测、walk-forward。不读 holdout。"""
from . import book, costs, jobs, panel, protocol, stats

returns = panel
ic = metrics = stats
backtest = combo = book
folds = protocol
paths = runtime = jobs
