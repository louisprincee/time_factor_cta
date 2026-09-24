"""研究评估层（设计文档第 5-11 步）。

收益、标准化、信号、时序 IC、绩效、回测与分族合成、walk-forward、
中心点选参、冻结配置。这一层不读 holdout_locked/。

模块
----
panel     日收益、滚动 MAD、滚动分位信号
stats     时序 IC 与六项绩效
costs     tick 相对滑点（手续费之外的成本）
book      等权回测与分族合成
protocol  walk-forward 折、中心点选参、冻结配置
jobs      落盘路径与各步共用的读取

下面的名字是合并前的模块。scripts 仍按这些名字导入。
"""
from . import book, costs, jobs, panel, protocol, stats

returns = standardize = signal = panel
ic = metrics = stats
backtest = combo = book
folds = center = freeze = protocol
paths = runtime = jobs
