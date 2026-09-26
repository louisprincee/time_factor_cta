import rqdatac as rq
import pandas as pd
from tqdm import tqdm
import pickle
import os
import re
import sys
import argparse
from datetime import date, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from tfcta.rq_auth import rqdata_credentials  # noqa: E402


# 通用函数：查找并加载同一前缀的所有数据文件
def find_and_load_all(prefix, directory):
    """查找并加载目录中所有匹配文件名的数据"""
    pattern = re.compile(rf"{prefix}(\d{{8}})-(\d{{8}})\.txt")
    all_data = pd.DataFrame()
    file_paths = []

    for file in os.listdir(directory):
        if pattern.match(file):
            file_path = os.path.join(directory, file)
            file_paths.append(file_path)

    for path in file_paths:
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                if all_data.empty:
                    all_data = data
                else:
                    all_data = pd.concat([all_data, data], axis=0)
            print(f"已加载: {os.path.basename(path)}")
        except Exception as e:
            print(f"加载文件失败: {path}: {e}")

    if not all_data.empty:
        # 去重排序
        all_data = all_data[~all_data.index.duplicated(keep='last')]
        all_data = all_data.sort_index()
        print(f"已合并 {len(file_paths)} 个文件")

    return all_data, file_paths


# 通用函数：保存数据并清理旧文件
def save_and_clean(data, file_path):
    """保存数据并清理相同类型但日期范围不同的旧文件"""
    if data.empty:
        print("无数据可保存")
        return

    # 提取文件名关键部分
    file_name = os.path.basename(file_path)
    prefix_parts = file_name.split('_')[:-1]  # 取出日期范围前面的所有部分
    prefix = '_'.join(prefix_parts)  # 重新组合关键前缀

    # 保存新数据
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"已保存: {file_name}")

    # 查找并删除同前缀但不同日期范围的文件
    to_delete = []
    pattern = re.compile(rf"^{re.escape(prefix)}_\d{{8}}-\d{{8}}\.txt$")

    output_dir = os.path.dirname(os.path.abspath(file_path))
    for file in os.listdir(output_dir):
        full_path = os.path.join(output_dir, file)
        # 只处理匹配前缀的文件，且不是当前正在保存的文件
        if pattern.match(file) and file != file_name:
            to_delete.append(full_path)

    for path in to_delete:
        try:
            # 删除主文件
            os.remove(path)

            # 删除对应的CSV文件
            csv_file = path.replace('.txt', '.csv')
            if os.path.exists(csv_file):
                os.remove(csv_file)

            print(f"已清理旧文件: {os.path.basename(path)}")
        except Exception as e:
            print(f"清理文件失败: {path}: {e}")


def load_existing_date(file_path):
    """从文件名中提取已有数据的起止日期"""
    if not file_path:
        return None, None

    # 使用正则从文件名中提取日期
    match = re.search(r'(\d{8})-(\d{8})', file_path)
    if match:
        existing_start = match.group(1)
        existing_end = match.group(2)
        print(f"已有数据范围: {existing_start}-{existing_end}")
        return existing_start, existing_end
    else:
        print("无法从已有文件名解析日期范围")
        return None, None


parser = argparse.ArgumentParser(description="从米筐下载期货分钟行情")

# 设置--time_period参数
parser.add_argument('-t', '--time-period', '--time_period', dest='time_period',
                    type=str,
                    default=f'20240101-{date.today() - timedelta(days=1):%Y%m%d}',
                    help="日期范围，格式 YYYYMMDD-YYYYMMDD")
# 添加数据路径参数
parser.add_argument('--datapath', type=str, default='./data/data_min',
                    help='数据保存目录（默认: ./data/data_min）')

args = parser.parse_args()
if not re.fullmatch(r'\d{8}-\d{8}', args.time_period):
    parser.error("--time-period 格式必须为 YYYYMMDD-YYYYMMDD")
target_start, target_end = args.time_period.split('-')
if target_start > target_end:
    parser.error("--time-period 的开始日期不能晚于结束日期")

credentials = rqdata_credentials()
if credentials is None:
    parser.error("请在 config/rqdata.env 填写 RQDATAC_LICENSE，或填写用户名和密码")
rq.init(*credentials)

names = ['AG', 'AL', 'AU', 'BC', 'BU', 'CU', 'FU', 'HC', 'LU', 'NI', 'NR', 'PB', 'RB',
            'RU', 'SC', 'SN', 'SP', 'SS', 'WR', 'ZN', 'A', 'B', 'BB', 'C', 'CS', 'EB',
            'EG', 'FB', 'I', 'J', 'JD', 'JM', 'L', 'LH', 'M', 'P', 'PG', 'PP', 'RR', 'V',
            'Y', 'AP', 'CF', 'CJ', 'CY', 'FG', 'JR', 'LR', 'MA', 'OI', 'PF', 'PK', 'PM',
            'RI', 'RM', 'RS', 'SA', 'SF', 'SM', 'SR', 'TA', 'UR', 'WH', 'ZC', 'LC',
            'IC', 'IF', 'IH', 'IM', 'T', 'TF', 'TS', 'TL','AO','EC','SH','AD','BR','LG','PR','PX','SI','PS']
typelst = ['88', '889', '99']
time_type = '1m'
index_type = 'datetime'
time_period = args.time_period
data_dir = os.path.abspath(args.datapath)
original_time_period = args.time_period

os.makedirs(data_dir, exist_ok=True)


file_prefix = 'future_all1mdata'  # 文件前缀（用于查找已有数据）
file_extension = '.txt'  # 选择要读取的文件类型（.txt 或 .csv）
existing_file = None
for file in os.listdir(data_dir):
    if file.startswith(file_prefix) and file.endswith(file_extension):
        existing_file = os.path.join(data_dir, file)
        break  # 只读取一个文件

if existing_file:
    print(f"发现已有总数据文件: {os.path.basename(existing_file)}")

# 读取已有数据，确定时间范围
existing_start, existing_end = load_existing_date(existing_file) if existing_file else (None, None)
print(f"目标日期范围: {time_period}")

if existing_file is not None and target_end > existing_end:
    next_start_date = (pd.to_datetime(existing_end) + pd.Timedelta(days=1)).strftime("%Y%m%d")
    time_period = f'{next_start_date}-{target_end}'
    print(f"增量下载范围: {time_period}")
elif existing_file is not None and target_end <= existing_end:
    print(f"已有数据已覆盖 {time_period}，无需更新")
    sys.exit(0)



# %%
# TODO 纯数据
for _type in typelst:
    data = pd.DataFrame()
    request_start, request_end = time_period.split('-')
    for name in tqdm(names, desc=f"下载 {_type}"):
        _data = rq.get_price(name + _type, start_date=request_start,
                             end_date=request_end, frequency=time_type)
        _data.index = _data.index.get_level_values(index_type).tolist()
        _data.columns = [[name for i in range(len(_data.columns))], _data.columns.tolist()]
        if data.empty:
            data = _data.copy()
        else:
            data = pd.concat([data, _data], axis=1)

    print(f"{_type} 分钟行情下载完成")

    if _type != '888' and data_dir:
        # 文件前缀格式：future{time_type}{_type}_
        file_prefix = f'future{time_type}{_type}_'

        # 查找并加载所有匹配文件
        all_existing_data, existing_paths = find_and_load_all(file_prefix, data_dir)

        if not all_existing_data.empty:
            # 合并新数据到已有数据
            print("合并已有行情数据")
            # 只保留新数据中不在已有数据的时间点
            new_data_only = data[~data.index.isin(all_existing_data.index)]

            # 如果新数据部分不为空
            if not new_data_only.empty:
                # 纵向拼接
                all_existing_data = pd.concat([all_existing_data, new_data_only], axis=0)
                # 去重并排序
                all_existing_data = all_existing_data[~all_existing_data.index.duplicated(keep='last')]
                all_existing_data = all_existing_data.sort_index()
                data = all_existing_data

    # 输出文件名（使用完整时间范围）
    file_name = f'future{time_type}{_type}_{original_time_period}.txt'
    output_path = os.path.join(data_dir, file_name)

    # 保存数据并清理旧文件
    save_and_clean(data, output_path)

# %%
#################################################################
# %%
# TODO 合并含贴水价格
with open(os.path.join(data_dir, f'future{time_type}88_{original_time_period}.txt'), 'rb') as f:
    data_88 = pickle.load(f)

with open(os.path.join(data_dir, f'future{time_type}889_{original_time_period}.txt'), 'rb') as f:
    data_889 = pickle.load(f)

lst = ['open','close','high','low']
lstw = ['openw','closew','highw','loww']

data = pd.DataFrame()
for name in tqdm(names):
    _dataw = data_889[name][lst].copy()
    _dataw.columns = lstw
    _data = pd.concat([data_88[name], _dataw], axis=1)
    _data.columns = [[name for i in range(len(_data.columns))], _data.columns.tolist()]
    if data.empty:
        data = _data.copy()
    else:
        data = pd.concat([data, _data], axis=1)

# 文件路径
file_name = f'future{time_type}_agio_{original_time_period}.txt'
output_path = os.path.join(data_dir, file_name)
# 保存数据并清理旧文件（使用通用函数）
save_and_clean(data, output_path)

# %%
####################################################################
# %%
# TODO 从99获取成交量成交额数据
with open(os.path.join(data_dir, f'future{time_type}_agio_{original_time_period}.txt'), 'rb') as f:
    data_all = pickle.load(f)

with open(os.path.join(data_dir, f'future{time_type}99_{original_time_period}.txt'), 'rb') as f:
    data_v = pickle.load(f)

lst = ['open_interest', 'volume']
lstw = ['open_interest99', 'volume99']
data = pd.DataFrame()
for name in tqdm(names, desc="合并品种数据"):
    _dataw = data_v[name][lst].copy()
    _dataw.columns = lstw
    _data = pd.concat([data_all[name], _dataw], axis=1)
    _data.columns = [[name for i in range(len(_data.columns))], _data.columns.tolist()]
    if data.empty:
        data = _data.copy()
    else:
        data = pd.concat([data, _data], axis=1)

def convert_to_float32(df):
    float64_cols = df.select_dtypes(include=['float64']).columns
    df[float64_cols] = df[float64_cols].astype('float32')
    return df


# 加载并合并历史数据（如果存在增量更新）
if existing_file:
    print("合并已有总数据")
    try:
        file_prefix = f'future_all{time_type}data_'
        # 查找并加载所有历史完整数据文件
        all_history, history_paths = find_and_load_all(file_prefix, data_dir)

        if not all_history.empty:
            # 合并新数据
            data = pd.concat([all_history, data], axis=0)
            # 去重并排序
            data = data[~data.index.duplicated(keep='last')]
            data = data.sort_index()
            print("总数据合并完成")

    except Exception as e:
        print(f"合并历史总数据失败: {e}")

data = convert_to_float32(data)
# 最终文件路径
file_name = f'future_all{time_type}data_{original_time_period}.txt'
output_path = os.path.join(data_dir, file_name)

# 保存最终数据并清理旧文件
save_and_clean(data, output_path)

print(f"下载完成: {output_path}")