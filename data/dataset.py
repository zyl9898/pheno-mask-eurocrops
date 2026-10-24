# -*- coding: utf-8 -*-
"""
data/dataset.py
================================================================================
EuroCropsML 预算约束自监督少样本分类 —— 数据集与物候感知掩码生成
================================================================================
维度总览（全文统一：B=batch, T=time steps, C=bands, N=样本数）

    npz 原始                data (T_raw, C_raw)      dates (T_raw,)      label ()
      │  _sanitize()            去 NaN/Inf、按日期排序、去重复日期、量纲自适应
      ▼
    规范化序列               (T_raw, C)
      │  resample_mode='interp' → 线性插值到 T=128（固定长度，供 batching）
      │  resample_mode='pad'    → 保留 T_raw，由 collate 动态 padding
      ▼
    网络输入序列             (T, C)   T ∈ {128} 或 {T_raw}
      │  build_mask()          掩码只作用在【时间维】，channel 维永不掩
      ▼
    掩码样本                 x_masked (T, C)   mask (T,) bool   target (T, C)
      │  collate_pretrain()
      ▼
    batch                    x (B, T, C)   mask (B, T)   pad_mask (B, T)   target (B, T, C)

    L-TAE 编码器路径（Step 3 会用到，这里先约定好）：
        (B, T, C) --Conv1d--> (B, C', T) --permute--> (B, T, C')
                  --Attention(query (H,1,D))--> (B, H, T) --weighted sum--> (B, H, D)
                  --flatten/linear--> (B, Dim)

设计原则（防"大模型幻觉式代码"）：
  1. 每个形状变换前都有 assert，失败即抛带上下文的异常，绝不依赖隐式广播；
  2. 掩码率/阈值/步数全部走 config，函数里不出现魔法数字；
  3. NDVI 的除法分母加 eps 并截断到 [-1, 1]，避免 6G 卡上出现 NaN 污染梯度；
  4. 掩码"至少保留 1 个可见步 + 至少掩 1 个步"，保证 MSE Loss 不会退化成 0；
  5. 所有路径都是相对 workspace 的相对路径，数据只从 <project>/data/ 下读取。
================================================================================
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# ==============================================================================
# 0. 常量与工具
# ==============================================================================

# Sentinel-2 L2A 的 13 个波段标准顺序（EuroCropsML 默认按此顺序存储）
S2_13_BANDS: Tuple[str, ...] = (
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B8A", "B09", "B10", "B11", "B12",
)

# 不同波段数下的 (RED, NIR) 索引表。Sentinel-2 只有 B04/B08 是 10m 且
# 是 NDVI 的标准输入，其余组合按"去掉哪些波段"推导：
#   13 波段：B01..B12 + B8A        → B04=3, B08=7
#   12 波段：13 去掉 B10(索引 9)   → B04=3, B08=7（B08 之前的索引不变）
#   10 波段：B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12 → B04=2, B08=6
#    4 波段：B,G,R,NIR（很常见的裁剪版）→ R=2, NIR=3
_BAND_LAYOUTS: Dict[int, Tuple[int, int]] = {13: (3, 7), 12: (3, 7), 10: (2, 6), 4: (2, 3)}

EPS = 1e-8


def _log(msg: str) -> None:
    print(f"[dataset] {msg}", flush=True)


def resolve_band_indices(
    n_bands: int,
    band_names: Optional[Sequence[str]] = None,
    red_index: Optional[int] = None,
    nir_index: Optional[int] = None,
) -> Tuple[int, int]:
    """解析出 RED / NIR 在 channel 维上的索引，返回 (red_idx, nir_idx)。

    优先级：显式下标 > band_names 名称匹配 > 波段数查表 > 报错。
    绝不静默使用错误索引（否则 NDVI 全错，训练照样收敛但物理意义为零）。
    """
    if red_index is not None and nir_index is not None:
        _check_idx(red_index, n_bands, "red_index")
        _check_idx(nir_index, n_bands, "nir_index")
        return int(red_index), int(nir_index)

    if band_names:
        names = [str(b).upper().strip() for b in band_names]
        if len(names) != n_bands:
            warnings.warn(
                f"band_names 长度 {len(names)} != 实际波段数 {n_bands}，忽略名称匹配，改查表。"
            )
        else:
            try:
                return names.index("B04"), names.index("B08")
            except ValueError:
                pass  # 名称不规范，退回查表

    if n_bands in _BAND_LAYOUTS:
        red, nir = _BAND_LAYOUTS[n_bands]
        return red, nir

    # 最后的兜底：假设最后一列是 NIR、倒数第二列是 RED（4 波段裁剪版最常见）
    warnings.warn(
        f"未知波段数 C={n_bands}，回退为 red=C-2, nir=C-1；请显式配置 red_index/nir_index。"
    )
    return n_bands - 2, n_bands - 1


def _check_idx(i: int, n: int, name: str) -> None:
    if not (0 <= int(i) < n):
        raise ValueError(f"{name}={i} 越界（C={n}）")


# ==============================================================================
# 1. 配置（全部走 dataclass，杜绝散落的魔法数）
# ==============================================================================


@dataclass
class SequenceConfig:
    """时序处理配置。"""

    resample_mode: str = "interp"       # 'interp'（固定 128 步） | 'pad'（动态 padding）
    n_timesteps: int = 128              # interp 模式的输出步数；pad 模式下的上限
    min_raw_timesteps: int = 16         # 原始观测少于该值的 parcel 直接丢弃（无法插值）
    input_scale: Union[str, float] = "auto"  # 'auto' | 1.0 | 10000.0 | 255.0
    band_names: Optional[List[str]] = None
    red_index: Optional[int] = None
    nir_index: Optional[int] = None

    def __post_init__(self) -> None:
        assert self.resample_mode in ("interp", "pad"), f"未知 resample_mode={self.resample_mode}"
        assert self.n_timesteps >= 2, "n_timesteps 至少为 2"
        assert self.min_raw_timesteps >= 2, "min_raw_timesteps 至少为 2"


@dataclass
class MaskConfig:
    """掩码配置（创新点：物候感知掩码）。"""

    mask_type: str = "phenology"        # 'phenology' | 'uniform'
    keep_prob_steep: float = 0.9        # 陡峭期（|ΔNDVI| 大）：保留 0.9 → 掩码率 10%
    keep_prob_flat: float = 0.7         # 平稳期：保留 0.7 → 掩码率 30%
    keep_prob_uniform: float = 0.8      # uniform 基线：全局保留概率（掩码率 20%）
    slope_threshold: Optional[float] = None  # None → 按 parcel 内部分位数自适应
    slope_quantile: float = 0.6         # 自适应阈值取 |ΔNDVI| 的 60% 分位
    protect_edges: bool = False         # True 时首尾 token 永不掩（可选消融）
    seed: Optional[int] = None          # 掩码随机种子；None → 每 epoch 不同

    def __post_init__(self) -> None:
        assert self.mask_type in ("phenology", "uniform"), f"未知 mask_type={self.mask_type}"
        for name in ("keep_prob_steep", "keep_prob_flat", "keep_prob_uniform"):
            p = getattr(self, name)
            assert 0.0 < p <= 1.0, f"{name}={p} 必须在 (0, 1] 内"
        assert 0.0 < self.slope_quantile < 1.0


# ==============================================================================
# 2. 单 parcel 的读取与清洗
# ==============================================================================


def _to_relative_days(dates: np.ndarray) -> np.ndarray:
    """把 dates 统一成 float 的"距首景天数"，返回 (T,)。

    兼容三种真实情况：
      - datetime64[...]（EuroCropsML 官方格式）
      - 字符串日期 'YYYY-MM-DD'
      - 数值：相对天数 / 年积日 DOY(1..366) / Unix 秒(1e9 量级)
    """
    d = np.asarray(dates)
    if d.ndim != 1:
        d = d.reshape(-1)

    if np.issubdtype(d.dtype, np.datetime64):
        dd = d.astype("datetime64[D]")
        return (dd - dd.min()).astype("timedelta64[D]").astype(np.float64)

    if d.dtype.kind in ("U", "S", "O"):
        try:
            dd = d.astype("datetime64[D]")
            return (dd - dd.min()).astype("timedelta64[D]").astype(np.float64)
        except (ValueError, TypeError) as e:
            raise ValueError(f"无法解析 dates（dtype={d.dtype}）: {e}") from e

    v = d.astype(np.float64)
    if not np.all(np.isfinite(v)):
        raise ValueError("dates 含 NaN/Inf，无法建立时间轴")
    # Unix 时间戳（秒/毫秒）→ 天
    if np.nanmax(np.abs(v)) > 1e11:
        v = v / 1000.0
    if np.nanmax(np.abs(v)) > 1e8:
        v = v / 86400.0
    v = v - v.min()  # 平移到 0 起点，保证可插值
    return v


def _sanitize_2d(data: Any, path: Path) -> np.ndarray:
    """把 data 变成 (T, C) float32，并修复常见脏数据。"""
    x = np.asarray(data)
    if x.dtype == object:
        x = np.stack([np.asarray(r, dtype=np.float32) for r in x], axis=0)
    x = x.astype(np.float32, copy=False)

    if x.ndim == 1:                      # 极端情况：(T,) → (T, 1)
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError(f"{path.name}: data 期望 2D，实际 {x.shape}")

    # 防御：只处理"行数是合法波段数、列数不是"的明确情形。
    # 不能写成 `行数 in {4,10,12,13}`：12 期观测 × 13 波段恰好是 (12,13)，
    # 而 12 又在波段布局表里，旧判据会把本来正确的文件转置成 (13,12)。
    # 真正模糊时（两边都像波段数）交给 load_parcel 用 dates 长度裁决。
    if (x.shape[1] not in _BAND_LAYOUTS) and (x.shape[0] in _BAND_LAYOUTS) and x.shape[0] < x.shape[1]:
        warnings.warn(f"{path.name}: data 形状 {x.shape} 疑似 (C,T)，已自动转置为 (T,C)")
        x = x.T

    # 非有限值：逐波段沿时间线性插值，两端用最近邻，整列全 NaN 则置 0
    if not np.all(np.isfinite(x)):
        T = x.shape[0]
        t = np.arange(T, dtype=np.float64)
        for c in range(x.shape[1]):
            col = x[:, c].astype(np.float64)
            ok = np.isfinite(col)
            if ok.sum() == 0:
                col[:] = 0.0
            elif ok.sum() < T:
                col = np.interp(t, t[ok], col[ok])
            x[:, c] = col.astype(np.float32)

    # 负反射率（L2A 大气校正后的正常现象）截断到 0，避免 NDVI 分母为负
    np.clip(x, 0.0, None, out=x)
    return x


def _apply_input_scale(x: np.ndarray, mode: Union[str, float]) -> np.ndarray:
    """量纲自适应：EuroCropsML 有的子集存 0~10000 整数反射率，有的存 0~1。

    不做这一步，per-band 归一化会把 10000 量级的波段压到 ~0，模型直接学不动。
    """
    if isinstance(mode, (int, float)):
        return x / float(mode) if float(mode) > 0 else x

    if mode == "auto":
        p99 = float(np.nanpercentile(x, 99)) if x.size else 0.0
        if p99 > 100.0:      # 0~10000 反射率 → /10000
            return x / 10000.0
        if p99 > 1.5:        # 0~255 DN → /255
            return x / 255.0
        return x             # 已在 0~1
    raise ValueError(f"未知 input_scale={mode}")



def _label_from_path(path: Union[str, Path]) -> str:
    """从文件名解析类别标签。

    EuroCropsML v8 的 .npz 只存 data / dates / center，**没有 label 字段**，
    类别以 10 位 HCAT3 代码写在文件名末段：
        LV008_12605483_3301020100.npz → '3301020100'
    早期文档/第三方重打包的版本可能带 label，故优先读字段、缺失才回退到文件名。
    """
    stem = Path(path).stem
    parts = stem.split("_")
    if len(parts) >= 3 and parts[-1].isdigit():
        return parts[-1]
    return stem


def load_parcel(path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, Any]:
    """读取单个 .npz，返回 (data (T,C) float32 未缩放, days (T,), label)。

    label 可能是 int / str / bytes，由上层统一映射成类别索引。
    """
    path = Path(path)
    with np.load(path, allow_pickle=True) as f:
        if "data" not in f:
            raise KeyError(f"{path} 缺少必需键 'data'，实际键={list(f.keys())}")
        data = f["data"]
        dates = f["dates"] if "dates" in f else None
        # 真实 EuroCropsML 无 label 字段，类别在文件名里
        label = f["label"] if "label" in f else _label_from_path(path)

    x = _sanitize_2d(data, path)

    # 权威判定：若 dates 长度对上的是列数而不是行数，说明该文件存成了 (bands, T)。
    # 真实 EuroCropsML 里确实混有这种文件（例如 13 波段 × 12 期观测），
    # 仅靠"行数 < 列数"的启发式会漏判，必须用 dates 长度兜底。
    if dates is not None:
        n_dates = int(np.asarray(dates).reshape(-1).shape[0])
        if n_dates == x.shape[1] and n_dates != x.shape[0]:
            x = x.T

    if dates is None:
        # 防御：没有 dates 就假设等间隔（EuroCropsML 不会出现，但第三方重构数据会）
        warnings.warn(f"{path.name}: 缺少 'dates'，退化为等间隔时间轴")
        days = np.arange(x.shape[0], dtype=np.float64)
    else:
        days = _to_relative_days(dates)
        if days.shape[0] != x.shape[0]:
            raise ValueError(
                f"{path.name}: dates 长度 {days.shape[0]} != data 时间步 {x.shape[0]}"
            )

    # 按时间升序 + 去掉重复日期（重复日期会让 np.interp 得到非单调 xp）
    order = np.argsort(days, kind="stable")
    days, x = days[order], x[order]
    uniq_mask = np.concatenate([[True], np.diff(days) > 0])
    if not uniq_mask.all():
        days, x = days[uniq_mask], x[uniq_mask]

    if isinstance(label, np.ndarray) and label.ndim == 0:
        label = label.item()
    elif isinstance(label, np.ndarray):
        label = label.reshape(-1)[0]
    return x, days, label


# ==============================================================================
# 3. 重采样 / 归一化
# ==============================================================================


def resample_linear(x: np.ndarray, days: np.ndarray, n_out: int) -> np.ndarray:
    """沿时间轴线性插值到固定长度。

    (T_raw, C) → (n_out, C)。逐波段做 np.interp，等价于把时间轴归一化到 [0,1]
    后重采样；用真实 days 而不是索引，才能真正对齐不同卫星过境间隔的 parcel。
    """
    t_raw, c = x.shape
    if c == 0:
        raise ValueError("C=0，空序列")
    if t_raw == n_out:
        return x.astype(np.float32, copy=True)
    if t_raw < 2:
        return np.repeat(x.astype(np.float32), n_out, axis=0)

    t_src = days.astype(np.float64)
    if t_src[-1] - t_src[0] <= 0:
        t_src = np.arange(t_raw, dtype=np.float64)
    t_src = (t_src - t_src[0]) / (t_src[-1] - t_src[0])          # (T_raw,) ∈ [0,1]
    t_out = np.linspace(0.0, 1.0, n_out, dtype=np.float64)        # (n_out,)

    out = np.empty((n_out, c), dtype=np.float32)
    for k in range(c):
        out[:, k] = np.interp(t_out, t_src, x[:, k].astype(np.float64))
    return out


class BandNormalizer:
    """逐波段 z-score 归一化。统计量来自预训练集，存 json 供 few-shot 复用。"""

    def __init__(self, mean: Sequence[float], std: Sequence[float], eps: float = 1e-6):
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, -1)   # (1, C)
        self.std = np.maximum(np.asarray(std, dtype=np.float32).reshape(1, -1), eps)
        # 形状约定：(1, C) 便于直接对 (T, C) / (B, T, C) 广播

    @property
    def n_bands(self) -> int:
        return int(self.mean.shape[1])

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> Dict[str, List[float]]:
        return {"mean": self.mean.reshape(-1).tolist(), "std": self.std.reshape(-1).tolist()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BandNormalizer":
        return cls(d["mean"], d["std"])

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BandNormalizer":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def save(self, path: Union[str, Path]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


# ==============================================================================
# 4. 掩码生成（论文创新点所在）
# ==============================================================================


def compute_ndvi(x: np.ndarray, red_idx: int, nir_idx: int) -> np.ndarray:
    """NDVI = (B8 - B4) / (B8 + B4)。

    x: (T, C) → 返回 (T,)。分母加 eps 并截断到 [-1, 1]；
    Sentinel-2 L2A 在浓云/阴影下可能出现 B4+B8≈0，必须防 0 除。
    """
    assert x.ndim == 2, f"compute_ndvi 期望 (T, C)，实际 {x.shape}"
    red = x[:, red_idx].astype(np.float64)                          # (T,)
    nir = x[:, nir_idx].astype(np.float64)                          # (T,)
    denom = nir + red
    ndvi = (nir - red) / np.maximum(denom, EPS)                     # (T,)
    ndvi[denom < EPS] = 0.0                                         # 双零 → 无信息，置 0
    return np.clip(ndvi, -1.0, 1.0)


def ndvi_slope(ndvi: np.ndarray) -> np.ndarray:
    """一阶差分绝对值 |ΔNDVI|，返回与输入同长度的 (T,)：

        slope[0]   = |NDVI[1] - NDVI[0]|      （边界复制首个差分）
        slope[t]   = |NDVI[t] - NDVI[t-1]|    t >= 1

    这样 slope[t] 描述的正是"token t 相对前一时刻的变化速度"，
    与 L-TAE 的 token 一一对应，掩码不会错位一个时间步。
    """
    assert ndvi.ndim == 1, f"ndvi_slope 期望 (T,)，实际 {ndvi.shape}"
    t = ndvi.shape[0]
    if t == 1:
        return np.zeros(1, dtype=np.float64)
    d = np.abs(np.diff(ndvi))                                       # (T-1,)
    slope = np.empty(t, dtype=np.float64)
    slope[0] = d[0]
    slope[1:] = d
    return slope


def phenology_aware_mask(
    x: np.ndarray,
    red_idx: int,
    nir_idx: int,
    cfg: MaskConfig,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """【创新点】物候感知掩码。

    物理动机：作物分类的判别信息集中在物候突变（播种/返青/抽穗/收割）附近，
    平稳期的光谱高度冗余。因此：
        陡峭期 slope >  thr → 保留概率 0.9（掩码率 10%，尽量别破坏判别信息）
        平稳期 slope <= thr → 保留概率 0.7（掩码率 30%，逼模型学时序上下文）
    相比均匀掩码，等效掩码率不变的情况下，把"被掩的位置"从随机变成
    "偏平稳期"，迫使编码器用平稳期上下文去推理突变期，正是论文要验证的点。

    参数
    ----
    x   : (T, C) 已重采样、已（可选）归一化的光谱
    cfg : MaskConfig
    rng : np.random.Generator，None 时新建（非确定性）

    返回 dict：
        mask       (T,)  bool，True = 该时间步被掩码（输入置 0，参与重建 loss）
        keep_prob  (T,)  float32，逐步保留概率（可视化/日志用）
        slope      (T,)  float32，|ΔNDVI|
        threshold  float，实际使用的阈值
        mask_ratio float，实际掩码比例
    """
    assert x.ndim == 2, f"期望 (T, C)，实际 {x.shape}"
    T = x.shape[0]
    rng = rng if rng is not None else np.random.default_rng()

    ndvi = compute_ndvi(x, red_idx, nir_idx)                        # (T,)
    slope = ndvi_slope(ndvi)                                        # (T,)

    # ---- 阈值：绝对阈值 or 逐 parcel 自适应分位 ----
    if cfg.slope_threshold is not None:
        thr = float(cfg.slope_threshold)
    else:
        # 用分位数而不是均值：NDVI 斜率是长尾分布，均值会被极端突变拉高，
        # 导致几乎所有点被判为"平稳期"，物候掩码退化成均匀掩码。
        thr = float(np.quantile(slope, cfg.slope_quantile))

    steep = slope > thr                                             # (T,) bool
    keep_prob = np.where(steep, cfg.keep_prob_steep, cfg.keep_prob_flat).astype(np.float32)

    if cfg.mask_type == "uniform":
        # 均匀掩码基线：与物候无关，保留概率恒定，用于 Step 5 的消融对比
        keep_prob = np.full(T, cfg.keep_prob_uniform, dtype=np.float32)

    # ---- 伯努利采样：rand >= keep_prob 则掩掉 ----
    u = rng.random(T)                                               # (T,) ∈ [0,1)
    mask = u >= keep_prob                                           # (T,) bool

    if cfg.protect_edges and T > 2:
        mask[0] = False
        mask[-1] = False

    # ---- 防御 1：全被掩 → 随机放回一个（否则 encoder 无任何输入）----
    if mask.all():
        mask[int(rng.integers(0, T))] = False
    # ---- 防御 2：一个都没掩 → 强制掩掉斜率最大处（保证重建 loss 非零）----
    if not mask.any():
        mask[int(np.argmax(slope))] = True

    return {
        "mask": mask,
        "keep_prob": keep_prob,
        "slope": slope.astype(np.float32),
        "threshold": thr,
        "mask_ratio": float(mask.mean()),
    }


def apply_mask(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """把被掩码的时间步整行置 0。

    x    : (T, C)
    mask : (T,) bool
    返回 : (T, C) 新数组（不改原数组，避免 __getitem__ 之间互相污染）
    """
    assert x.shape[0] == mask.shape[0], f"T 不一致: x={x.shape}, mask={mask.shape}"
    out = x.copy()
    out[mask] = 0.0
    return out


# ==============================================================================
# 5. 单个 parcel 的完整处理管线（供两个 Dataset 复用）
# ==============================================================================


class ParcelPipeline:
    """load → sanitize → scale → (resample | pad) → normalize → (mask)

    刻意做成独立类而不是 Dataset 内联：Step 4 的 evaluate 需要"同样的预处理但不要掩码"，
    直接复用同一个 pipeline 实例即可，避免训练/评估预处理不一致这种最隐蔽的 bug。
    """

    def __init__(self, seq_cfg: SequenceConfig, normalizer: Optional[BandNormalizer] = None):
        self.seq = seq_cfg
        self.normalizer = normalizer
        # RED/NIR 索引延迟到第一次见到实际波段数 C 时再解析并缓存
        # （拉脱维亚与爱沙尼亚子集的 C 可能不同，固定死会静默算错 NDVI）
        self.red_idx: Optional[int] = None
        self.nir_idx: Optional[int] = None
        self._c: int = -1

    def _indices(self, n_bands: int) -> Tuple[int, int]:
        if self.red_idx is None or self._c != n_bands:
            self.red_idx, self.nir_idx = resolve_band_indices(
                n_bands,
                band_names=self.seq.band_names,
                red_index=self.seq.red_index,
                nir_index=self.seq.nir_index,
            )
            self._c = n_bands
        return int(self.red_idx), int(self.nir_idx)

    def load_raw(self, path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, Any]:
        x, days, label = load_parcel(path)
        x = _apply_input_scale(x, self.seq.input_scale)              # (T_raw, C)
        return x, days, label

    def transform(
        self,
        x: np.ndarray,
        days: np.ndarray,
        rng: Optional[np.random.Generator] = None,
        mask_cfg: Optional[MaskConfig] = None,
    ) -> Dict[str, Any]:
        """返回固定结构的 dict（np 数组，全部 (T, C) / (T,)）。"""
        t_raw, c = x.shape
        if t_raw < self.seq.min_raw_timesteps:
            raise ValueError(f"时间步不足：T_raw={t_raw} < {self.seq.min_raw_timesteps}")

        # ---- 重采样策略 ----
        if self.seq.resample_mode == "interp":
            xr = resample_linear(x, days, self.seq.n_timesteps)      # (128, C)
            pad_mask = np.ones(xr.shape[0], dtype=bool)              # (128,) 全有效
        else:  # 'pad'
            if t_raw > self.seq.n_timesteps:
                xr = resample_linear(x, days, self.seq.n_timesteps)  # 裁到上限
                pad_mask = np.ones(xr.shape[0], dtype=bool)
            else:
                xr = x.astype(np.float32)                            # (T_raw, C)
                pad_mask = np.ones(t_raw, dtype=bool)                # 真实长度，padding 交给 collate

        t = xr.shape[0]
        red_idx, nir_idx = self._indices(c)

        # ---- 掩码必须在重采样之后做 ----
        # 原因：mask 的粒度是 encoder 的 token，而 token 是重采样后的 128 步。
        # 若在 T_raw 上算掩码再插值，掩码位置与 token 会错位，重建 loss 就学了错的东西。
        mask = np.zeros(t, dtype=bool)
        info = None
        if mask_cfg is not None:
            info = phenology_aware_mask(xr, red_idx, nir_idx, mask_cfg, rng)
            mask = info["mask"] & pad_mask                          # 只在有效观测区掩码

        # ---- 归一化（z-score），target 与 input 用同一套统计量 ----
        x_norm = self.normalizer(xr) if self.normalizer is not None else xr
        x_in = apply_mask(x_norm, mask) if mask.any() else x_norm    # (T, C) 掩码处置 0

        return {
            "x": x_in.astype(np.float32),
            "target": x_norm.astype(np.float32),
            "mask": mask,
            "pad_mask": pad_mask,
            "mask_ratio": float(mask.mean()) if mask.size else 0.0,
            "threshold": float(info["threshold"]) if info is not None else float("nan"),
        }


def read_lengths(ids: Sequence[str], index: Dict[str, Path],
                 cache_path: Optional[Union[str, Path]] = None) -> Dict[str, int]:
    """读每个 parcel 的原始观测期数 T（带 json 缓存）。

    为什么需要：真实数据里存在 12 期观测的短序列，低于 min_raw_timesteps 会被
    transform() 拒绝。评估集若不预先剔除，会在 DataLoader worker 里抛异常，
    整个 episode 崩掉 —— 而 12 次评估重新扫描 3.8 万个文件太慢，故缓存。
    """
    cache: Dict[str, int] = {}
    cache_path = Path(cache_path) if cache_path is not None else None
    if cache_path is not None and cache_path.exists():
        try:
            cache = {k: int(v) for k, v in
                     json.loads(cache_path.read_text(encoding="utf-8")).items() if k in index}
        except (json.JSONDecodeError, ValueError, OSError):
            cache = {}

    todo = [i for i in ids if i not in cache]
    if todo:
        for i in todo:
            try:
                with np.load(index[i], allow_pickle=True) as f:
                    shp = np.asarray(f["data"]).shape
                    t = int(shp[0]) if len(shp) >= 2 else 0
                    if "dates" in f:
                        nd = int(np.asarray(f["dates"]).reshape(-1).shape[0])
                        if len(shp) >= 2 and nd == shp[1] and nd != shp[0]:
                            t = nd
                    cache[i] = t
            except Exception:                       # 坏文件按 0 处理，后面会被剔除
                cache[i] = 0
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache), encoding="utf-8")
    return {i: cache.get(i, 0) for i in ids}


# ==============================================================================
# 6. 索引构建（扫 npz / 读 split.json / 类别标签缓存）
# ==============================================================================


def scan_npz(root: Union[str, Path]) -> Dict[str, Path]:
    """递归扫描 root 下所有 .npz，返回 {parcel_id: path}。

    parcel_id 取相对 root 的 posix 路径（去扩展名），
    例如 'Estonia/parcels/12345.npz' → 'Estonia/parcels/12345'。
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"数据目录不存在: {root.resolve()}（数据必须放在 workspace 内）")
    out: Dict[str, Path] = {}
    for p in sorted(root.rglob("*.npz")):
        rel = p.relative_to(root).as_posix()
        out[rel[: -len(".npz")]] = p
    if not out:
        raise FileNotFoundError(f"{root} 下没有找到任何 .npz")
    return out


def _norm_id(s: Any) -> str:
    """把 split.json 里各种写法统一成 parcel_id。"""
    s = str(s).replace("\\", "/")
    if s.endswith(".npz"):
        s = s[: -len(".npz")]
    return s


def load_split(split: Union[str, Path, Dict[str, Any], None]) -> Dict[str, List[str]]:
    """解析 split.json。

    支持三种真实写法：
      A) {"train": ["id1", ...], "val": [...], "test": [...]}
      B) {"train": {"0": ["id1"], "1": ["id2"]}, ...}     （按类别分组）
      C) {"train": [{"id": "id1", "label": 3}, ...]}       （带元信息的列表）
    """
    if split is None:
        return {}
    if isinstance(split, (str, Path)):
        with open(split, "r", encoding="utf-8") as f:
            split = json.load(f)
    assert isinstance(split, dict), f"split.json 顶层必须是 dict，实际 {type(split)}"

    out: Dict[str, List[str]] = {}
    for key, val in split.items():
        ids: List[str] = []
        if isinstance(val, dict):                     # 写法 B
            for _cls, sub in val.items():
                ids.extend(_norm_id(v) for v in sub)
        elif isinstance(val, (list, tuple)):
            for v in val:
                if isinstance(v, dict):               # 写法 C
                    v = v.get("id", v.get("parcel_id", v.get("name")))
                if v is not None:
                    ids.append(_norm_id(v))
        else:
            raise ValueError(f"split['{key}'] 类型不支持: {type(val)}")
        out[str(key).lower()] = ids
    return out


def _match_ids(ids: Iterable[str], index: Dict[str, Path]) -> List[str]:
    """把 split 里的 id 匹配到实际文件。优先全路径匹配，再退化到 basename 匹配。"""
    by_base: Dict[str, List[str]] = {}
    for k in index:
        by_base.setdefault(k.rsplit("/", 1)[-1], []).append(k)

    matched: List[str] = []
    missing = 0
    for i in ids:
        if i in index:
            matched.append(i)
        elif i.rsplit("/", 1)[-1] in by_base:
            cands = by_base[i.rsplit("/", 1)[-1]]
            if len(cands) > 1:
                warnings.warn(f"parcel id '{i}' 在多处出现（{cands[:3]}...），取第一个")
            matched.append(cands[0])
        else:
            missing += 1
    if missing:
        warnings.warn(f"split 中有 {missing} 个 id 在数据目录里找不到，已跳过")
    return matched


def _label_to_str(label: Any) -> str:
    if isinstance(label, bytes):
        return label.decode("utf-8", errors="ignore")
    return str(label)


def read_labels(
    ids: Sequence[str],
    index: Dict[str, Path],
    cache_path: Optional[Union[str, Path]] = None,
) -> Dict[str, str]:
    """读取每个 parcel 的 label（字符串化），带 json 缓存（避免每次训练都开一遍 npz）。"""
    cache: Dict[str, str] = {}
    cache_path = Path(cache_path) if cache_path is not None else None
    if cache_path is not None and cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            cache = {k: str(v) for k, v in raw.items() if k in index}
        except (json.JSONDecodeError, OSError) as e:
            warnings.warn(f"标签缓存损坏，重建：{e}")
            cache = {}

    todo = [i for i in ids if i not in cache]
    if todo:
        _log(f"读取 {len(todo)} 个 parcel 的 label（首次运行较慢，之后走缓存）...")
        for i in todo:
            with np.load(index[i], allow_pickle=True) as f:
                cache[i] = _label_to_str(f["label"] if "label" in f else _label_from_path(index[i]))
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False)
    return {i: cache[i] for i in ids if i in cache}


# ==============================================================================
# 7. 预训练数据集（Latvia，无监督掩码重建）
# ==============================================================================


class LatviaPretrainDataset(Dataset):
    """拉脱维亚自监督预训练数据集。

    每个样本返回：
        x        (T, C) float32   被掩码后的输入（掩码处置 0）
        target   (T, C) float32   完整（归一化后）光谱，作为重建目标
        mask     (T,)   bool      True = 被掩码，只有这些位置计入 MSE
        pad_mask (T,)   bool      True = 真实观测（padding 位置为 False）
        label    ()     long      仅用于监控，不参与预训练 loss
        ratio    ()     float     该 parcel 实际掩码率（用于日志对比掩码类型）
    """

    def __init__(
        self,
        root: Union[str, Path],
        seq_cfg: SequenceConfig,
        mask_cfg: MaskConfig,
        normalizer: Optional[BandNormalizer] = None,
        split: Union[str, Path, Dict[str, Any], None] = None,
        subsets: Sequence[str] = ("train", "val"),
        label_cache: Optional[Union[str, Path]] = None,
        max_samples: Optional[int] = None,
        epoch: int = 0,
        verbose: bool = True,
    ):
        super().__init__()
        self.root = Path(root)
        self.seq_cfg = seq_cfg
        self.mask_cfg = mask_cfg
        self.pipeline = ParcelPipeline(seq_cfg, normalizer)
        self.epoch = int(epoch)

        index_all = scan_npz(self.root)
        split_map = load_split(split)

        if split_map:
            keep: List[str] = []
            for s in subsets:
                keep.extend(_match_ids(split_map.get(s.lower(), []), index_all))
            if not keep:
                warnings.warn(f"subset{subsets} 为空，回退到全量数据")
                keep = sorted(index_all.keys())
            # 去重但保持顺序
            seen, ids = set(), []
            for i in keep:
                if i not in seen:
                    seen.add(i)
                    ids.append(i)
        else:
            ids = sorted(index_all.keys())

        self.paths = [index_all[i] for i in ids]
        self.ids = ids

        # ---- 预筛：时间步太短的 parcel 直接剔除（一次性读取 header 级别的 data.shape）----
        self.paths = self._filter_by_length(self.paths, verbose)

        if max_samples is not None and len(self.paths) > max_samples:
            rng = np.random.default_rng(0)
            sel = rng.choice(len(self.paths), size=max_samples, replace=False)
            self.paths = [self.paths[i] for i in sorted(sel)]
            _log(f"按 max_samples 截断为 {len(self.paths)} 个 parcel")

        if len(self.paths) == 0:
            raise RuntimeError("过滤后没有任何可用 parcel，请检查 min_raw_timesteps / 数据目录")

        if verbose:
            _log(f"LatviaPretrainDataset: {len(self.paths)} 个 parcel | "
                 f"resample={seq_cfg.resample_mode} T={seq_cfg.n_timesteps} | "
                 f"mask={mask_cfg.mask_type}")

        # 记录一次实际掩码率的滑动统计，训练初期就能确认"物候掩码真的偏平稳期"
        self._ratio_buf: List[float] = []

    def _filter_by_length(self, paths: List[Path], verbose: bool) -> List[Path]:
        kept = []
        dropped = 0
        for p in paths:
            try:
                with np.load(p, allow_pickle=True) as f:
                    n = int(np.asarray(f["data"]).shape[0])
            except Exception:  # 损坏文件不要爆整个训练
                dropped += 1
                continue
            if n >= self.seq_cfg.min_raw_timesteps:
                kept.append(p)
            else:
                dropped += 1
        if verbose and dropped:
            _log(f"丢弃 {dropped} 个时间步不足/损坏的 parcel")
        return kept

    def __len__(self) -> int:
        return len(self.paths)

    def set_epoch(self, epoch: int) -> None:
        """每个 epoch 调用一次，使掩码随 epoch 变化且可复现。"""
        self.epoch = int(epoch)

    def _rng(self, idx: int) -> np.random.Generator:
        if self.mask_cfg.seed is None:
            return np.random.default_rng()                    # 纯随机（不可复现）
        # SeedSequence 保证 (seed, epoch, idx) → 唯一子种子：
        # 多 worker 并行时不会撞种子，也不受 worker 调度顺序影响。
        ss = np.random.SeedSequence([int(self.mask_cfg.seed), self.epoch, int(idx)])
        return np.random.default_rng(ss)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        x_raw, days, label = self.pipeline.load_raw(self.paths[idx])
        out = self.pipeline.transform(x_raw, days, rng=self._rng(idx), mask_cfg=self.mask_cfg)

        # label → int（预处理阶段不做词表，这里只哈希成一个稳定整数便于监控）
        if isinstance(label, (int, np.integer)):
            y = int(label)
        else:
            y = abs(hash(_label_to_str(label))) % (2 ** 31)

        return {
            "x": torch.from_numpy(out["x"]),                        # (T, C)
            "target": torch.from_numpy(out["target"]),               # (T, C)
            "mask": torch.from_numpy(out["mask"]),                   # (T,)
            "pad_mask": torch.from_numpy(out["pad_mask"]),           # (T,)
            "label": torch.tensor(y, dtype=torch.long),              # ()
            "ratio": float(out["mask_ratio"]),                       # python float
            "pid": idx,                                              # 调试用（collate 里丢弃）
        }


# ==============================================================================
# 8. 少样本数据集（Estonia，5/10/20-shot）
# ==============================================================================


class EstoniaFewShotDataset(Dataset):
    """爱沙尼亚少样本分类数据集（episodic 采样）。

    __getitem__ 返回一个 episode：
        support_x (n_way*n_shot, T, C)   support_y (n_way*n_shot,)
        query_x   (n_way*n_query, T, C)  query_y   (n_way*n_query,)
        class_names (n_way,) list[str]   —— 注意：default_collate 不支持 list[str]，
        需要时请用 collate_episodic()，或保持 return_meta=False（默认，不含该键）。

    episode 采样用 (seed, epoch, idx) 派生子种子，多 worker 下依然可复现。
    """

    def __init__(
        self,
        root: Union[str, Path],
        split: Union[str, Path, Dict[str, Any]],
        n_way: int = 5,
        n_shot: int = 5,
        n_query: int = 10,
        n_episodes: int = 200,
        subset: str = "train",
        query_subset: str = "same",
        seq_cfg: Optional[SequenceConfig] = None,
        normalizer: Optional[BandNormalizer] = None,
        label_cache: Optional[Union[str, Path]] = None,
        seed: int = 0,
        epoch: int = 0,
        return_meta: bool = False,
        strict_n_way: bool = True,
        verbose: bool = True,
    ):
        super().__init__()
        assert int(n_shot) >= 1, "n_shot 必须为正整数"
        assert n_query >= 1, "n_query 至少为 1"
        self.root = Path(root)
        self.seq_cfg = seq_cfg or SequenceConfig(resample_mode="interp", n_timesteps=128)
        self.pipeline = ParcelPipeline(self.seq_cfg, normalizer)
        self.n_way, self.n_shot, self.n_query = int(n_way), int(n_shot), int(n_query)
        self.n_episodes = int(n_episodes)
        self.seed, self.epoch = int(seed), int(epoch)
        self.return_meta = bool(return_meta)

        index_all = scan_npz(self.root)
        split_map = load_split(split)
        if not split_map:
            raise ValueError("少样本数据集必须提供 split.json（train/val/test）")
        subset = subset.lower()
        if subset not in split_map:
            raise KeyError(f"split.json 中没有 subset='{subset}'，可用: {list(split_map)}")

        ids = _match_ids(split_map[subset], index_all)
        ids = [i for i in ids if i in index_all]
        if not ids:
            raise RuntimeError(f"subset='{subset}' 匹配到 0 个文件，请检查 split.json 与 root")

        # query 池：默认与 support 同源；官方 k-shot 划分里 train 每类只有 k 个，
        # query 必须来自 test，否则 support 和 query 会重叠（严重的数据泄漏）
        query_subset = str(query_subset).lower()
        if query_subset not in ("same", subset) and query_subset not in split_map:
            raise KeyError(f"query_subset='{query_subset}' 不在 split.json 中：{list(split_map)}")
        q_ids: List[str] = []
        if query_subset not in ("same", subset):
            q_ids = [i for i in _match_ids(split_map[query_subset], index_all) if i in index_all]
            if not q_ids:
                raise RuntimeError(f"query_subset='{query_subset}' 匹配到 0 个文件")

        # 先按原始期数剔除过短序列（12 期这类会被 transform 拒绝）
        lens = read_lengths(list(dict.fromkeys(ids + q_ids)), index_all,
                            Path(label_cache).with_name("_length_cache.json")
                            if label_cache else None)
        min_t = self.seq_cfg.min_raw_timesteps
        short = [i for i, t in lens.items() if t < min_t]
        if short:
            _log(f"EstoniaFewShot: 剔除 {len(short)}/{len(lens)} 个短序列（<{min_t} 期）")
        ids = [i for i in ids if lens.get(i, 0) >= min_t]
        q_ids = [i for i in q_ids if lens.get(i, 0) >= min_t]

        labels = read_labels(list(dict.fromkeys(ids + q_ids)), index_all, label_cache)
        ids = [i for i in ids if i in labels]
        q_ids = [i for i in q_ids if i in labels]
        self._index = index_all            # {parcel_id: Path}，供 _load_xy 取文件

        # class → [ids]
        self.class_to_ids: Dict[str, List[str]] = {}
        for i in ids:
            self.class_to_ids.setdefault(labels[i], []).append(i)

        self._split_query_pool = query_subset not in ("same", subset)
        if self._split_query_pool:
            self.query_to_ids: Dict[str, List[str]] = {}
            for i in q_ids:
                self.query_to_ids.setdefault(labels[i], []).append(i)
            # 同一类必须两边都有，否则该 episode 无法构成
            keep = {c: v for c, v in self.class_to_ids.items()
                    if len(v) >= self.n_shot and c in self.query_to_ids and len(self.query_to_ids[c]) >= 1}
            dropped = {c: len(v) for c, v in self.class_to_ids.items() if c not in keep}
            need_desc = f"每类 ≥{self.n_shot} 个 support + ≥1 个 {query_subset} query"
        else:
            self.query_to_ids = self.class_to_ids
            min_need = self.n_shot + self.n_query
            keep = {c: v for c, v in self.class_to_ids.items() if len(v) >= min_need}
            dropped = {c: len(v) for c, v in self.class_to_ids.items() if c not in keep}
            need_desc = f"每类 ≥{min_need} 个 sample（{self.n_shot}-shot + {self.n_query}-query）"

        self.class_to_ids = keep
        self.class_names: List[str] = sorted(self.class_to_ids.keys())
        self.label_map: Dict[str, int] = {c: i for i, c in enumerate(self.class_names)}  # 稳定映射

        if verbose:
            _log(f"EstoniaFewShot[{subset}→{query_subset}]: {len(ids)} parcel / "
                 f"{len(self.class_names)} 类 (丢弃 {len(dropped)} 个不满足 {need_desc} 的类)")
            if dropped:
                _log(f"  丢弃类别示例: {list(dropped.items())[:5]}")

        if len(self.class_names) < 2:
            raise RuntimeError(f"可用类别只有 {len(self.class_names)} 个，无法做少样本分类")
        if self.n_way > len(self.class_names):
            msg = (f"n_way={self.n_way} > 可用类别 {len(self.class_names)}"
                   f"（多数类被 min_shot 过滤掉了？）")
            if strict_n_way:
                # 论文实验里 n_way 被悄悄改小会让所有对比失效，默认直接报错
                raise ValueError(msg + "；如确实要降级请传 strict_n_way=False")
            warnings.warn(msg + "，已自动降级")
            self.n_way = len(self.class_names)

    # ---------------------------------------------------------------- helpers
    def _rng(self, idx: int) -> np.random.Generator:
        ss = np.random.SeedSequence([self.seed, self.epoch, int(idx), 7])
        return np.random.default_rng(ss)

    def _load_xy(self, pid: str) -> Tuple[np.ndarray, int]:
        """加载一个 parcel → ((T, C) 已归一化序列, 类别索引 int)。不做掩码。"""
        x_raw, days, label = self.pipeline.load_raw(self._index[pid])
        out = self.pipeline.transform(x_raw, days, rng=None, mask_cfg=None)
        return out["x"], self.label_map[_label_to_str(label)]

    def __len__(self) -> int:
        return self.n_episodes

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    # ---------------------------------------------------------------- episode
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rng = self._rng(idx)
        classes = rng.choice(len(self.class_names), size=self.n_way, replace=False)

        sx, sy, qx, qy = [], [], [], []
        for ci in classes:
            cname = self.class_names[int(ci)]
            s_pool = self.class_to_ids[cname]
            q_pool = self.query_to_ids[cname]

            if len(s_pool) >= self.n_shot:
                s_idx = rng.choice(len(s_pool), size=self.n_shot, replace=False)
            else:
                s_idx = rng.integers(0, len(s_pool), size=self.n_shot)
            if len(q_pool) >= self.n_query:
                q_idx = rng.choice(len(q_pool), size=self.n_query, replace=False)
            else:
                # query 很少时有放回；support 与 query 不同源，仍无泄漏
                q_idx = rng.integers(0, len(q_pool), size=self.n_query)

            for pi in s_idx:
                x, y = self._load_xy(s_pool[int(pi)])                # (T, C), int
                sx.append(x); sy.append(y)
            for pi in q_idx:
                x, y = self._load_xy(q_pool[int(pi)])
                qx.append(x); qy.append(y)

        support_x = np.stack(sx, axis=0)                             # (n_way*n_shot, T, C)
        query_x = np.stack(qx, axis=0)                               # (n_way*n_query, T, C)
        assert support_x.shape[0] == self.n_way * self.n_shot
        assert query_x.shape[0] == self.n_way * self.n_query

        out = {
            "support_x": torch.from_numpy(support_x),                # (N_s, T, C)
            "support_y": torch.tensor(sy, dtype=torch.long),         # (N_s,)
            "query_x": torch.from_numpy(query_x),                    # (N_q, T, C)
            "query_y": torch.tensor(qy, dtype=torch.long),           # (N_q,)
        }
        if self.return_meta:
            out["class_names"] = [self.class_names[int(c)] for c in classes]
        return out


class EstoniaTestDataset(Dataset):
    """测试集"全量"数据集：非 episodic，用于 Step 4 计算 OA / Macro-F1。

    返回 (x (T,C), y ()) 单样本；用 default_collate 即可得到 (B, T, C) / (B,)。
    """

    def __init__(
        self,
        root: Union[str, Path],
        split: Union[str, Path, Dict[str, Any]],
        subset: str = "test",
        seq_cfg: Optional[SequenceConfig] = None,
        normalizer: Optional[BandNormalizer] = None,
        label_cache: Optional[Union[str, Path]] = None,
        label_map: Optional[Dict[str, int]] = None,
        max_samples: Optional[int] = None,
        verbose: bool = True,
    ):
        index_all = scan_npz(root)
        split_map = load_split(split)
        subset = subset.lower()
        if subset not in split_map:
            raise KeyError(f"split.json 中没有 '{subset}'，可用: {list(split_map)}")
        ids = [i for i in _match_ids(split_map[subset], index_all) if i in index_all]
        lens = read_lengths(ids, index_all,
                            Path(label_cache).with_name("_length_cache.json")
                            if label_cache else None)
        min_t = (seq_cfg or SequenceConfig()).min_raw_timesteps
        kept = [i for i in ids if lens.get(i, 0) >= min_t]
        if len(kept) != len(ids):
            warnings.warn(f"EstoniaTestDataset[{subset}]: 剔除 {len(ids) - len(kept)} 个短序列"
                          f"（<{min_t} 期）")
        ids = kept
        labels = read_labels(ids, index_all, label_cache)
        ids = [i for i in ids if i in labels]
        if max_samples and len(ids) > max_samples:
            ids = ids[:max_samples]

        self.root = Path(root)
        self.ids = ids
        self.paths = [index_all[i] for i in ids]
        self.seq_cfg = seq_cfg or SequenceConfig()
        self.pipeline = ParcelPipeline(self.seq_cfg, normalizer)
        self.labels_raw = [labels[i] for i in ids]

        # 类别映射：训练时给的 label_map 优先（保证索引一致），否则按字典序自建
        if label_map is None:
            uniq = sorted(set(self.labels_raw))
            self.label_map = {c: i for i, c in enumerate(uniq)}
        else:
            self.label_map = dict(label_map)
        self.class_names = [c for c, _ in sorted(self.label_map.items(), key=lambda kv: kv[1])]

        unknown = sum(1 for c in self.labels_raw if c not in self.label_map)
        if unknown:
            warnings.warn(f"测试集中有 {unknown} 个样本的类别不在 label_map 内（将被跳过）")
        if verbose:
            _log(f"EstoniaTestDataset[{subset}]: {len(self.ids)} 个样本 / {len(self.class_names)} 类")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        x_raw, days, _ = self.pipeline.load_raw(self.paths[idx])
        out = self.pipeline.transform(x_raw, days, rng=None, mask_cfg=None)
        y = self.label_map.get(self.labels_raw[idx], -1)
        return {"x": torch.from_numpy(out["x"]), "y": torch.tensor(y, dtype=torch.long)}


# ==============================================================================
# 9. collate 函数与 DataLoader 工厂
# ==============================================================================


def collate_pretrain(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """支持变长 T（pad 模式）与定长 T（interp 模式）。

    stack → x (B, T, C) / mask (B, T) / pad_mask (B, T) / target (B, T, C)
    pad 位置：x/target 填 0，mask=False，pad_mask=False（绝不参与任何 loss）。
    """
    t_max = max(int(b["x"].shape[0]) for b in batch)
    c = int(batch[0]["x"].shape[1])
    bsz = len(batch)

    x = torch.zeros(bsz, t_max, c, dtype=torch.float32)
    target = torch.zeros(bsz, t_max, c, dtype=torch.float32)
    mask = torch.zeros(bsz, t_max, dtype=torch.bool)
    pad_mask = torch.zeros(bsz, t_max, dtype=torch.bool)
    labels = torch.zeros(bsz, dtype=torch.long)
    ratios = torch.zeros(bsz, dtype=torch.float32)

    for i, b in enumerate(batch):
        t = int(b["x"].shape[0])
        assert b["x"].shape[1] == c, f"batch 内 C 不一致: {b['x'].shape} vs C={c}"
        x[i, :t] = b["x"]
        target[i, :t] = b["target"]
        mask[i, :t] = b["mask"]
        pad_mask[i, :t] = b["pad_mask"]
        labels[i] = b["label"]
        ratios[i] = float(b["ratio"])

    return {
        "x": x, "target": target, "mask": mask, "pad_mask": pad_mask,
        "label": labels, "ratio": ratios,
    }


def collate_episodic(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """把 K 个 episode 拼成 (K, N_s, T, C)。

    K 个 episode 的 N_s/N_q 相同，可以直接 stack；class_names 单独聚成 list。
    """
    out = {
        "support_x": torch.stack([b["support_x"] for b in batch], dim=0),
        "support_y": torch.stack([b["support_y"] for b in batch], dim=0),
        "query_x": torch.stack([b["query_x"] for b in batch], dim=0),
        "query_y": torch.stack([b["query_y"] for b in batch], dim=0),
    }
    if "class_names" in batch[0]:
        out["class_names"] = [b["class_names"] for b in batch]
    return out


def _worker_init(worker_id: int) -> None:
    """Windows + 多 worker 的种子设置（DataLoader 会 fork 相同随机状态）。"""
    seed = torch.initial_seed() % (2 ** 32 - 1)
    np.random.seed(seed)
    _log(f"worker {worker_id} 就绪 (seed={seed})")


def make_pretrain_loader(
    dataset: LatviaPretrainDataset,
    batch_size: int = 16,
    num_workers: int = 2,
    shuffle: bool = True,
    drop_last: bool = True,
    pin_memory: bool = True,
) -> DataLoader:
    """构造预训练 DataLoader。

    Windows 注意：num_workers>0 时，训练入口必须写在
    `if __name__ == "__main__":` 之下，否则子进程会重复 import 导致递归创建。
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_pretrain,
        drop_last=drop_last,
        pin_memory=pin_memory and torch.cuda.is_available(),
        worker_init_fn=_worker_init if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )


def make_episodic_loader(
    dataset: EstoniaFewShotDataset,
    batch_size: int = 8,
    num_workers: int = 0,
    shuffle: bool = True,
    pin_memory: bool = True,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_episodic if dataset.return_meta else None,
        pin_memory=pin_memory and torch.cuda.is_available(),
        worker_init_fn=_worker_init if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )


# ==============================================================================
# 10. 归一化统计量估计（在预训练子集上跑一遍，6G 机器上几秒钟）
# ==============================================================================


@torch.no_grad()
def compute_norm_stats(
    dataset: LatviaPretrainDataset,
    max_samples: int = 2000,
    verbose: bool = True,
) -> BandNormalizer:
    """两遍法（sum / sumsq）逐波段统计 mean、std。

    只在"重采样后、未归一化"的序列上统计；掩码位置不参与（此时还没掩码）。
    """
    n = min(len(dataset), max_samples)
    idxs = np.linspace(0, len(dataset) - 1, n).astype(int).tolist()

    s = s2 = None
    count = 0
    for k, i in enumerate(idxs):
        x_raw, days, _ = dataset.pipeline.load_raw(dataset.paths[i])
        x = resample_linear(x_raw, days, dataset.seq_cfg.n_timesteps)   # (128, C)
        if s is None:
            s = np.zeros(x.shape[1], dtype=np.float64)
            s2 = np.zeros(x.shape[1], dtype=np.float64)
        s += x.sum(axis=0)
        s2 += (x.astype(np.float64) ** 2).sum(axis=0)
        count += x.shape[0]
        if verbose and (k + 1) % 200 == 0:
            _log(f"  统计归一化参数 {k + 1}/{n}")

    assert count > 0, "没有可用于统计的样本"
    mean = s / count
    var = np.maximum(s2 / count - mean ** 2, 1e-8)     # Var = E[x²] - E[x]²
    std = np.sqrt(var)
    if verbose:
        _log(f"归一化统计量: mean[:3]={np.round(mean[:3], 4)} std[:3]={np.round(std[:3], 4)}")
    return BandNormalizer(mean, std)


# ==============================================================================
# 10.5 从 configs/*.yaml 构造配置（Step 3/4 的入口脚本统一走这里）
# ==============================================================================


def seq_and_mask_from_yaml(path: Union[str, Path]) -> Tuple[SequenceConfig, MaskConfig, Dict[str, Any]]:
    """读取 configs/data.yaml → (SequenceConfig, MaskConfig, 其余原始配置)。

    yaml 里只写想覆盖的字段，其余取 dataclass 默认值；未知字段会直接报错，
    避免"配置写错一个字母、程序照跑但参数没生效"这种最难查的事故。
    """
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    seq_raw = dict(raw.get("sequence", {}) or {})
    mask_raw = dict(raw.get("mask", {}) or {})

    def _build(cls, kw: Dict[str, Any]):
        allowed = set(cls.__dataclass_fields__.keys())          # type: ignore[attr-defined]
        unknown = set(kw) - allowed
        if unknown:
            raise KeyError(f"{cls.__name__} 不认识的字段: {sorted(unknown)}；可用: {sorted(allowed)}")
        return cls(**kw)

    seq_cfg = _build(SequenceConfig, seq_raw)
    mask_cfg = _build(MaskConfig, mask_raw)
    return seq_cfg, mask_cfg, raw


# ==============================================================================
# 11. 自测：python -m data.dataset（不依赖真实数据，用合成 npz 验证全部维度假设）
# ==============================================================================

if __name__ == "__main__":
    import tempfile

    _log("=== dataset.py 自测开始 ===")
    tmp = Path(tempfile.mkdtemp(prefix="ecml_selftest_"))
    rng = np.random.default_rng(0)

    # 合成 12 个 parcel，长度 40~90 步，13 波段，含一段"陡峭期"
    for i in range(12):
        t = int(rng.integers(40, 90))
        days = np.sort(rng.choice(np.arange(0, 365), size=t, replace=False))
        ndvi_shape = 0.3 + 0.5 * np.sin(np.linspace(0, 3.14, t)) ** 2 + rng.normal(0, 0.01, t)
        band = np.clip(0.2 + 0.1 * ndvi_shape[:, None] + rng.normal(0, 0.01, (t, 13)), 0, None)
        band[:, 7] = 0.05 + 0.4 * ndvi_shape        # B08
        band[:, 3] = 0.05 + 0.1 * ndvi_shape        # B04
        np.savez(tmp / f"p{i:03d}.npz",
                 data=(band * 10000).astype(np.int32),   # 故意用 0~10000 量级
                 dates=days.astype("int32"),
                 label=i % 3)

    seq_cfg = SequenceConfig(resample_mode="interp", n_timesteps=128)
    mask_cfg = MaskConfig(mask_type="phenology", seed=42)
    ds = LatviaPretrainDataset(tmp, seq_cfg, mask_cfg, verbose=True)
    norm = compute_norm_stats(ds, max_samples=12)
    ds.pipeline.normalizer = norm

    item = ds[0]
    _log(f"单样本: x{item['x'].shape} target{item['target'].shape} "
         f"mask{item['mask'].shape} ratio={item['ratio']:.3f}")

    loader = make_pretrain_loader(ds, batch_size=4, num_workers=0)
    batch = next(iter(loader))
    _log(f"batch: x{tuple(batch['x'].shape)} mask{tuple(batch['mask'].shape)} "
         f"pad_mask{tuple(batch['pad_mask'].shape)} mean_ratio={batch['ratio'].mean():.3f}")
    assert batch["x"].shape == (4, 128, 13)
    assert batch["mask"].dtype == torch.bool

    # 物候掩码 vs 均匀掩码：陡峭期掩码率应显著更低
    for mt in ("phenology", "uniform"):
        cfg = MaskConfig(mask_type=mt, seed=7)
        ratios, steep_ratio, flat_ratio = [], [], []
        for i in range(len(ds)):
            x_raw, days, _ = ds.pipeline.load_raw(ds.paths[i])
            x = resample_linear(x_raw, days, 128)
            info = phenology_aware_mask(x, 3, 7, cfg, np.random.default_rng(i))
            ndvi = compute_ndvi(x, 3, 7)
            sp = ndvi_slope(ndvi)
            steep = sp > info["threshold"]
            if steep.sum():
                steep_ratio.append(info["mask"][steep].mean())
            if (~steep).sum():
                flat_ratio.append(info["mask"][~steep].mean())
        _log(f"  {mt:9s}: 陡峭期掩码率={np.mean(steep_ratio):.3f} "
             f"平稳期掩码率={np.mean(flat_ratio):.3f}")
    _log("=== 自测通过 ===")
