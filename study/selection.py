"""Fixed-budget frame policies. Indices refer to the shared candidate pool."""
import numpy as np
from sklearn.cluster import KMeans

MODES = ("ktv6", "hybrid4_neighbor", "hybrid4_top2", "query6")


def ktv_frames(features, k):
    if len(features) < k:
        raise ValueError("Fewer candidates than requested frames")
    km = KMeans(n_clusters=k, random_state=0, init="k-means++", n_init=10).fit(features)
    distances = np.linalg.norm(features[None] - km.cluster_centers_[:, None], axis=2)
    selected = list(dict.fromkeys(np.argmin(distances, axis=1).tolist()))
    # Degenerate/identical frames can produce duplicate representatives.
    # Fill deterministically by maximal distance from the selected timestamps.
    while len(selected) < k:
        remaining = [i for i in range(len(features)) if i not in selected]
        selected.append(max(remaining, key=lambda i: (min(abs(i-j) for j in selected), -i)))
    return sorted(selected)


def select(mode, base4, base6, scores, times, question, neighbor_seconds=1.0):
    scores, times = np.asarray(scores), np.asarray(times)
    if len(scores) < 6 or not np.isfinite(scores).all():
        raise ValueError("Need at least six finite candidate scores")
    ranked = np.argsort(-scores, kind="stable").tolist()
    anchor, neighbor = ranked[0], None
    if mode == "ktv6":
        chosen = list(base6)
    elif mode == "query6":
        chosen = ranked[:6]
    elif mode == "hybrid4_top2":
        chosen = list(base4)
    elif mode == "hybrid4_neighbor":
        chosen = list(base4)
        # Prefer the side explicitly named in the question. Otherwise compare
        # CLIP at fixed +/-1 s offsets; never inspect answer choices or labels.
        import re
        words = set(re.findall(r"[a-z]+", question.lower()))
        sides = [-1] if "before" in words and "after" not in words else [1] if "after" in words and "before" not in words else [-1, 1]
        def candidates(directions):
            result = []
            for side in directions:
                eligible = [i for i in range(len(times)) if (times[i]-times[anchor])*side > 0 and i not in chosen]
                if eligible:
                    result.append(min(eligible, key=lambda i: (abs(times[i]-(times[anchor]+side*neighbor_seconds)), i)))
            return result
        near = candidates(sides) or candidates([-1, 1])
        if near:
            neighbor = max(near, key=lambda i: (scores[i], -i))
        chosen += [anchor] + ([] if neighbor is None else [neighbor])
    else:
        raise ValueError(mode)
    chosen = list(dict.fromkeys(chosen))
    for i in ranked:
        if len(chosen) == 6:
            break
        if i not in chosen:
            chosen.append(i)
    chosen.sort()
    assert len(chosen) == len(set(chosen)) == 6
    # KTV-normal: 288,144,144,144,144,72 = 936, in query-relevance order.
    relevance = sorted(chosen, key=lambda i: (-scores[i], i))
    ranks = [relevance.index(i) for i in chosen]
    return {"indices": chosen, "ranks": ranks, "anchor": anchor, "neighbor": neighbor,
            "anchor_in_base4": anchor in base4,
            "tokens_per_frame": [[288,144,144,144,144,72][r] for r in ranks]}
