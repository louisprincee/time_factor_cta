"""商品期货五大板块。每个商品品种只属于一个板块，金融期货不在这张表里。

归类里有判断成分的几处：
    玻璃 FG、纯碱 SA、尿素 UR 归能源化工（纯碱、玻璃同属化工-建材链，与煤化工共振）；
    动力煤 ZC、不锈钢 SS、线材 WR 归黑色金属；
    碳酸锂 LC、工业硅 SI、多晶硅 PS、铸造铝 AD、氧化铝 AO 归有色金属；
    纸浆 SP、原木 LG、胶合板 BB、纤维板 FB 作为林产品归农产品；
    集运指数 EC 不是实物商品，暂归能源化工（运价与油价联动最强）。
"""
from __future__ import annotations

ALL_POOL = "全部"

SECTOR_BY_SYMBOL = {
    # 有色金属
    "AD": "有色金属", "AL": "有色金属", "AO": "有色金属", "BC": "有色金属",
    "CU": "有色金属", "NI": "有色金属", "PB": "有色金属", "SN": "有色金属",
    "ZN": "有色金属", "LC": "有色金属", "SI": "有色金属", "PS": "有色金属",
    # 黑色金属
    "HC": "黑色金属", "I": "黑色金属", "J": "黑色金属", "JM": "黑色金属",
    "RB": "黑色金属", "SF": "黑色金属", "SM": "黑色金属", "SS": "黑色金属",
    "WR": "黑色金属", "ZC": "黑色金属",
    # 贵金属
    "AG": "贵金属", "AU": "贵金属",
    # 能源化工
    "BU": "能源化工", "FU": "能源化工", "LU": "能源化工", "PG": "能源化工",
    "SC": "能源化工", "BR": "能源化工", "EB": "能源化工", "EG": "能源化工",
    "L": "能源化工", "MA": "能源化工", "NR": "能源化工", "PF": "能源化工",
    "PP": "能源化工", "PR": "能源化工", "PX": "能源化工", "RU": "能源化工",
    "SH": "能源化工", "TA": "能源化工", "UR": "能源化工", "V": "能源化工",
    "FG": "能源化工", "SA": "能源化工", "EC": "能源化工",
    # 农产品（含林产品）
    "A": "农产品", "AP": "农产品", "B": "农产品", "C": "农产品",
    "CF": "农产品", "CJ": "农产品", "CS": "农产品", "CY": "农产品",
    "JD": "农产品", "JR": "农产品", "LH": "农产品", "LR": "农产品",
    "M": "农产品", "OI": "农产品", "P": "农产品", "PK": "农产品",
    "PM": "农产品", "RI": "农产品", "RM": "农产品", "RR": "农产品",
    "RS": "农产品", "SR": "农产品", "WH": "农产品", "Y": "农产品",
    "SP": "农产品", "LG": "农产品", "BB": "农产品", "FB": "农产品",
}

SECTORS = ("有色金属", "黑色金属", "贵金属", "能源化工", "农产品")

_ALIASES = {
    "有色": "有色金属",
    "黑色": "黑色金属",
    "黑色系": "黑色金属",
    "能化": "能源化工",
    "能源": "能源化工",
    "化工": "能源化工",
    "农产": "农产品",
    "农业": "农产品",
    "all": ALL_POOL,
    "ALL": ALL_POOL,
}


def sector_names() -> list[str]:
    return list(SECTORS)


def sector_of(symbol: str) -> str:
    try:
        return SECTOR_BY_SYMBOL[symbol]
    except KeyError:
        raise KeyError(f"品种 {symbol} 没有板块归属，请在 data/sectors.py 登记") from None


def symbols_in(sector: str) -> list[str]:
    name = canonicalize(sector)
    if name == ALL_POOL:
        return sorted(SECTOR_BY_SYMBOL)
    return sorted(symbol for symbol, label in SECTOR_BY_SYMBOL.items() if label == name)


def canonicalize(name: str) -> str:
    text = _ALIASES.get(str(name).strip(), str(name).strip())
    if text != ALL_POOL and text not in SECTORS:
        raise ValueError(f"未知板块 {name}。可选: {ALL_POOL}, {', '.join(SECTORS)}")
    return text


def parse_pools(names: list[str] | None) -> list[str]:
    if not names:
        return []
    seen = []
    for name in names:
        label = canonicalize(name)
        if label not in seen:
            seen.append(label)
    return seen


def catalog() -> str:
    lines = ["商品期货板块（每个品种只出现一次；另有特殊池 全部）"]
    for sector in SECTORS:
        members = symbols_in(sector)
        lines.append(f"{sector} ({len(members)}): {', '.join(members)}")
    return "\n".join(lines)
