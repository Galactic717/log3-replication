"""Живі міні-демонстрації навчальних алгоритмів для захисту (нічого не записують на диск).

    python -m log3.defense.demo huffman     канонічний Гаффман: довжини, коди, стиснення, decode == вхід
    python -m log3.defense.demo planner     ДП-рюкзак проти жадібного і повного перебору на малому прикладі
"""
from __future__ import annotations

import itertools
import json
import random
import sys

from .. import huffman
from ..planner.knapsack import solve_dp, solve_greedy, total


def demo_huffman() -> dict:
    text = ("2026-11-18T10:00:01Z sev=3 src=auth msg=login failed user=42\n" * 30
            + "2026-11-18T10:00:02Z sev=1 src=disk msg=ok\n" * 70).encode()
    lengths = huffman.code_lengths(text)
    codes = huffman.canonical_codes(lengths)
    packed = huffman.encode(text, lengths)
    back = huffman.decode(packed, lengths, len(text))
    assert back == text
    kraft = sum(2.0 ** -n for n in lengths.values())
    top = sorted(lengths.items(), key=lambda kv: (kv[1], kv[0]))[:6]
    for sym, n in top:
        print(f"  символ {chr(sym)!r:6} довжина {n:2}  код {codes[sym][0]:0{n}b}")
    out = {"input_bytes": len(text), "output_bytes": len(packed), "symbols": len(lengths),
           "max_code_len": max(lengths.values()), "kraft_sum": kraft, "decode_equals_input": back == text}
    print(f"  {len(text)} B -> {len(packed)} B (коефіцієнт {len(packed) / len(text):.3f}), "
          f"сума Крафта {kraft:.6f}, decode == вхід: {back == text}")
    return out


def demo_planner(seed: int = 42) -> dict:
    rnd = random.Random(seed)
    n = 9
    values = [rnd.choice((1.0, 2.0, 5.0)) for _ in range(n)]
    weights = [rnd.randint(20, 60) for _ in range(n)]
    util = [[0.0, v, v * 1.4, v * 1.6] for v in values]
    cost = [[0, w, 2 * w, 3 * w] for w in weights]
    budget = int(sum(weights) * 1.2)
    dp, greedy = solve_dp(util, cost, budget), solve_greedy(util, cost, budget)
    best = max((total(util, cost, list(c)) for c in itertools.product(range(4), repeat=n)
                if total(util, cost, list(c))[1] <= budget), key=lambda t: t[0])
    res = {"records": n, "budget_bits": budget, "dp": total(util, cost, dp), "greedy": total(util, cost, greedy),
           "brute_force_best": best}
    print(f"  ДП:        копії {dp} -> корисність {res['dp'][0]:.2f}, біт {res['dp'][1]}")
    print(f"  жадібний:  копії {greedy} -> корисність {res['greedy'][0]:.2f}, біт {res['greedy'][1]}")
    print(f"  перебір 4^{n}: найкраща корисність {best[0]:.2f}; ДП оптимальний: {abs(res['dp'][0] - best[0]) < 1e-9}")
    assert abs(res["dp"][0] - best[0]) < 1e-9 and res["dp"][1] <= budget
    return res


def main(argv: list[str]) -> int:
    which = argv[0] if argv else "huffman"
    out = demo_huffman() if which == "huffman" else demo_planner()
    print(json.dumps({"ok": True, "demo": which, **out}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
