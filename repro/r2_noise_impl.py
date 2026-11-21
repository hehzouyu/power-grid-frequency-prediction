"""R2 多噪声实现的枚举与调用层（供 T3 直接调用）。

设计约束（T2 验收 + R2 协议 §3）：
1. 唯一的噪声生成实现仍是 ``analyze_quality_warning.observe()``。本模块只做
   「实现编号 ↔ 种子」映射与逐个实现的循环调用，**不复制、不重写任何随机数逻辑**，
   否则会出现两份实现并随时间漂移。
2. 枚举语义来自 R2 协议：``seed_k = NOISE_IMPLEMENTATION_BASE_SEED + k``，
   ``k=0`` 对应 ``N01`` 且必须逐位复现 R1。因此 ``N0j`` 的种子是基值 + (j-1)。
3. 只有观测噪声随实现变化；场景种子（39192512/39192513）、模型拟合（42）、
   bootstrap（39192515/39192517）均冻结，本模块不暴露它们的参数。
4. 不新增第三方依赖（只用标准库 + numpy）。

用法::

    from r2_noise_impl import enumerate_implementations, observe_implementation

    f, reference = ...                        # 144x1001 轨迹与噪声参考
    for impl in enumerate_implementations(5): # N01..N05，k=0..4
        signals, estimates, quality = observe_implementation(f, reference, impl)
"""
from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

REPRO_DIR = Path(__file__).resolve().parent
if str(REPRO_DIR) not in sys.path:  # 允许从任意工作目录 import 本模块
    sys.path.insert(0, str(REPRO_DIR))

from analyze_quality_warning import (  # noqa: E402
    CONDITIONS,
    NOISE_IMPLEMENTATION_BASE_SEED,
    STRESS,
    observe,
)

CONDITIONS_ALL = CONDITIONS + STRESS
VARIED_BY = "observationNoise"
MAX_IMPLEMENTATIONS = 100


@dataclass(frozen=True)
class NoiseImplementation:
    """一个噪声实现的完整标识。

    ``k`` 是协议里的实现序号（0 起），``seed = 基值 + k``；
    ``noise_impl_id`` 是记录规范要求的 N01…N05 形式（k=0 → N01）。
    """

    noise_impl_id: str
    k: int
    seed: int
    varied_by: str = VARIED_BY

    def as_dict(self) -> dict:
        return asdict(self)


def implementation_id(k: int) -> str:
    """协议序号 k（0 起）→ 记录用实现编号 N01…；k 越界直接报错而不是默默截断。"""
    if not isinstance(k, int) or isinstance(k, bool):
        raise TypeError(f"k 必须是 int，收到 {type(k).__name__}")
    if k < 0:
        raise ValueError(f"k 必须 >= 0（k=0 是 R1 位级锚点），收到 {k}")
    return f"N{k + 1:02d}"


def implementation_seed(k: int, base: int = NOISE_IMPLEMENTATION_BASE_SEED) -> int:
    """协议序号 k → 观测噪声种子；k=0 必须等于 R1 的 39192514。"""
    if not isinstance(base, int) or isinstance(base, bool):
        raise TypeError(f"base 必须是 int，收到 {type(base).__name__}")
    return base + k


def implementation(k: int, base: int = NOISE_IMPLEMENTATION_BASE_SEED) -> NoiseImplementation:
    return NoiseImplementation(
        noise_impl_id=implementation_id(k), k=k, seed=implementation_seed(k, base)
    )


def implementation_from_id(noise_impl_id: str, base: int = NOISE_IMPLEMENTATION_BASE_SEED) -> NoiseImplementation:
    """``N01`` → k=0；编号格式非法时报错，避免 T3 静默跑错实现。"""
    text = str(noise_impl_id).strip().upper()
    if len(text) == 3 and text.startswith("N") and text[1:].isdigit():
        k = int(text[1:]) - 1
        if k >= 0:
            return implementation(k, base)
    raise ValueError(f"noiseImplId 必须形如 N01…N{MAX_IMPLEMENTATIONS:02d}，收到 {noise_impl_id!r}")


def enumerate_implementations(
    count: int = 1,
    start_k: int = 0,
    base: int = NOISE_IMPLEMENTATION_BASE_SEED,
) -> tuple[NoiseImplementation, ...]:
    """按协议枚举 ``count`` 个实现：``k = start_k … start_k+count-1``。

    支持「实现数 + 起始序号」两种参数，因此既可以跑 N01..N20，也可以只跑
    N21..N40 做精度升级，而不必改代码。
    """
    if not isinstance(count, int) or isinstance(count, bool):
        raise TypeError(f"count 必须是 int，收到 {type(count).__name__}")
    if not 1 <= count <= MAX_IMPLEMENTATIONS:
        raise ValueError(f"count 必须在 1..{MAX_IMPLEMENTATIONS}，收到 {count}")
    if not isinstance(start_k, int) or isinstance(start_k, bool):
        raise TypeError(f"start_k 必须是 int，收到 {type(start_k).__name__}")
    if start_k < 0:
        raise ValueError(f"start_k 必须 >= 0，收到 {start_k}")
    return tuple(implementation(k, base) for k in range(start_k, start_k + count))


def implementations_from_seeds(
    seeds, base: int = NOISE_IMPLEMENTATION_BASE_SEED
) -> tuple[NoiseImplementation, ...]:
    """显式种子集合 → 实现列表；种子必须落在 ``base + k`` 的协议格点上。"""
    resolved = []
    for seed in seeds:
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError(f"种子必须是 int，收到 {seed!r}")
        k = seed - base
        if k < 0:
            raise ValueError(f"种子 {seed} 小于基值 {base}，不是协议内的观测噪声种子")
        resolved.append(implementation(k, base))
    if len({i.k for i in resolved}) != len(resolved):
        raise ValueError("种子集合中出现重复实现（同一 k 只能出现一次）")
    return tuple(resolved)


def observe_implementation(f, reference, impl):
    """对单个实现调用唯一噪声入口，返回 (signals, estimates, quality)。

    显式传 ``seed=impl.seed``，因此结果与该实现编号一一对应；``impl`` 也接受
    整数种子，便于手工排查。
    """
    seed = impl.seed if isinstance(impl, NoiseImplementation) else int(impl)
    return observe(f, reference, seed=seed)


def quality_tier_counts(quality) -> dict[str, int]:
    """每条件的 low/high 档计数（low = 未达 boundary，与 observe 的 ``quality`` 反义）。"""
    low = np.asarray(quality) == False  # noqa: E712 —— 保持与协议「low=quality False」同口径
    counts = {}
    for index, condition in enumerate(CONDITIONS_ALL):
        low_count = int(low[:, index].sum())
        counts[f"{condition}_low"] = low_count
        counts[f"{condition}_high"] = int(low.shape[0] - low_count)
    return counts


def describe(count: int = 1, start_k: int = 0, base: int = NOISE_IMPLEMENTATION_BASE_SEED) -> dict:
    """给 run-record 用的自描述块：枚举规则、RNG 抽取顺序、冻结种子。"""
    plan = enumerate_implementations(count, start_k, base)
    return {
        "baseSeed": base,
        "seedRule": "seed_k = baseSeed + k",
        "variedByImplementation": VARIED_BY,
        "rngDrawOrder": [
            "noise = rng.normal(size=(n, m+50))",
            "noise_ar = rng.normal(size=(n, m+50))",
            "zi = rng.normal(size=(n, 1)) * 0.8",
            "correlated = lfilter([0.6], [1, -0.8], noise_ar, axis=1, zi=zi)",
        ],
        "frozenSeeds": {
            "scenarioPopulation": [39192512, 39192513],
            "modelFit": 42,
            "bootstrapSummary": 39192515,
            "bootstrapPaired": 39192517,
        },
        "sharedAcrossConditions": True,
        "conditions": list(CONDITIONS_ALL),
        "implementations": [i.as_dict() for i in plan],
    }
