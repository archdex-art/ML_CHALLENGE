"""Pair features (stage 1) and group-context features over stage-1 probabilities (stage 2)."""
import math
import os
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

import common as C

AUX = ["s1", "cand", "fnum_c"]  # carried along, never fed to the model
WORKERS = os.cpu_count()


def idf_tables(rec):
    def idf(col):
        df = Counter(t for s in col for t in set(s.split()))
        return {t: math.log(len(col) / c) for t, c in df.items()}
    return idf(rec.nc.values), idf(rec.af.values)


FREQ = ["fq_n1", "fq_n23", "fq_a1", "fq_a23"]


def add_freq(rec):
    """Per record: how many S1 / S2-S3 records in its country share its core name (n) or address (a)."""
    s1 = rec.src == 1
    for tag, key in (("n", rec.country + "|" + rec.nc), ("a", (rec.country + "|" + rec.af).where(rec.af != ""))):
        for name, m in ((f"fq_{tag}1", s1), (f"fq_{tag}23", ~s1)):
            rec[name] = key.map(key[m].value_counts()).fillna(0).where(key.notna()).astype(np.float32)


def _jacc(a, b):
    u = a | b
    return len(a & b) / len(u) if u else np.nan


def _wjacc(a, b, idf, dflt):
    u = a | b
    if not u:
        return np.nan, np.nan
    inter = sum(idf.get(t, dflt) for t in a & b)
    return inter / sum(idf.get(t, dflt) for t in u), inter


_G = {}  # pair arrays + idf tables, shared with forked workers


def _loop(bounds):
    lo, hi = bounds
    g = _G
    idf_n, idf_a = g["idf_n"], g["idf_a"]
    dn, da = max(idf_n.values(), default=1.0), max(idf_a.values(), default=1.0)
    out = np.full((hi - lo, 11), np.nan, np.float32)
    for r, i in enumerate(range(lo, hi)):
        aj, bj = g["anj"][i].split(), g["bnj"][i].split()
        A, B = set(aj), set(bj)
        o = out[r]
        o[0] = _jacc(A, B)
        o[1] = _jacc(set(g["ans"][i].split()), set(g["bns"][i].split()))
        if aj and bj:
            o[2] = aj[0] == bj[0]
            o[3] = aj[0] in B
            o[4] = bj[0] in A
            o[5] = len(A - B)
            o[6] = len(B - A)
            x, y = _lead_int(aj[0]), _lead_int(bj[0])
            o[7] = math.log1p(abs(x - y))
        o[8], o[10] = _wjacc(set(g["anc"][i].split()), set(g["bnc"][i].split()), idf_n, dn)
        o[9] = _wjacc(set(g["aaf"][i].split()), set(g["baf"][i].split()), idf_a, da)[0]
    return out


def _lead_int(s):
    j = 0
    while j < len(s) and s[j].isdigit():
        j += 1
    return int(s[:j][:15] or 0)


LOOP_COLS = ["nj_jacc", "ns_jacc", "fn_eq", "fn_in", "bfn_in", "n_miss", "n_extra", "fn_diff",
             "n_widf", "a_widf", "n_idf_shared"]


# ---------------------------------------------------------------- sibling evidence
# The data holds "sibling" businesses: the S1 name plus a descriptor (Group, Holdings, Midtown, Exports,
# Participations, ...) or with one word swapped, a few doors down the same street. Which words mark a
# sibling is learned without labels, per country (so it transfers to countries absent from training):
# for every token that one name has and the other lacks, the rate at which the pair's house numbers agree.
# True-match noise words (Inc, Services, Center) agree often; sibling words almost never.
MARK = re.compile(r"^.*\b(?:dba|d b a|fka|f k a|aka|a k a|t a|formerly known as|formerly|trading as|"
                  r"doing business as|known as|nee)\b ?")
TOK_M = 10.0  # smoothing strength toward the country prior
TOK_COLS = ["tx_n", "tm_n", "tx_agree_min", "tx_agree_mean", "tm_agree_min", "tx_lcnt_min", "tm_lcnt_min",
            "tx_df_max", "tm_df_max"]


def tok_diff(a, b):
    """Tokens only in b (extra) / only in a (missing), after dropping b's trade-name prefix
    ('Xyz dba <name>') and pairing typo variants (ratio >= 70)."""
    b = MARK.sub("", b)
    sa, sb = set(a.split()), set(b.split())
    ex = [t for t in dict.fromkeys(b.split()) if t not in sa and len(t) > 1]
    mi = [t for t in dict.fromkeys(a.split()) if t not in sb and len(t) > 1]
    if ex and mi:
        for e in list(ex):
            r = [fuzz.ratio(e, m) for m in mi]
            j = int(np.argmax(r))
            if r[j] >= 70:
                ex.remove(e)
                mi.pop(j)
                if not mi:
                    break
    return ex, mi


def num_agree(ha, hb, ja, jb):
    """House numbers equal? Falls back to the first joined number; None when either side lacks one."""
    if ha and hb:
        return ha == hb
    ja, jb = ja.split(" ", 1)[0], jb.split(" ", 1)[0]
    if ja and jb:
        return ja == jb
    return None


def _tok_count(bounds):
    lo, hi = bounds
    g = _G
    ex_c, mi_c = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    for i in range(lo, hi):
        eq = num_agree(g["ahn"][i], g["bhn"][i], g["anj"][i], g["bnj"][i])
        if eq is None:
            continue
        ex, mi = tok_diff(g["anf"][i], g["bnf"][i])
        cty = g["cty"][i]
        for store, toks in ((ex_c, ex), (mi_c, mi)):
            for t in toks:
                s = store[cty, t]
                s[0] += 1
                s[1] += eq
    return dict(ex_c), dict(mi_c)


def token_stats(rec, pairs):
    """Per (country, token): [n, n_agree] as extra / missing token over all candidate pairs, the
    country prior agreement rate, and S2/S3 document frequencies of name tokens."""
    ia, ib = rec.index.get_indexer(pairs.s1.values), rec.index.get_indexer(pairs.cand.values)
    _G.update(anf=rec.nf.values[ia], bnf=rec.nf.values[ib], ahn=rec.hn.values[ia], bhn=rec.hn.values[ib],
              anj=rec.nj.values[ia], bnj=rec.nj.values[ib], cty=rec.country.values[ia])
    n = len(pairs)
    step = max(1, -(-n // (WORKERS * 8)))
    ex, mi = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    with C.fork_pool(WORKERS) as p:
        for e, m in p.imap_unordered(_tok_count, [(i, min(i + step, n)) for i in range(0, n, step)]):
            for store, part in ((ex, e), (mi, m)):
                for k, (c, a) in part.items():
                    s = store[k]
                    s[0] += c
                    s[1] += a
    _G.clear()
    prior = {}
    for k, (c, a) in ex.items():
        s = prior.setdefault(k[0], [0, 0])
        s[0] += c
        s[1] += a
    prior = {k: a / max(c, 1) for k, (c, a) in prior.items()}
    s23 = rec[rec.src != 1]
    df = {}
    for cty, names in s23.groupby("country").nf:
        for t, c in Counter(t for s in names.values for t in set(s.split())).items():
            df[cty, t] = c
    return {"ex": dict(ex), "mi": dict(mi), "prior": prior, "df": df}


def _tok_feats(o, ex, mi, eq, cty, ts):
    """Fill TOK_COLS into o. Leave-one-out: the pair's own contribution is removed from the rates."""
    pri = ts["prior"].get(cty, 0.5)
    own = 0 if eq is None else 1
    o[0], o[1] = len(ex), len(mi)
    for toks, store, ia, il, idf in ((ex, ts["ex"], 2, 5, 7), (mi, ts["mi"], 4, 6, 8)):
        if not toks:
            continue
        rates, cnts, dfs = [], [], []
        for t in toks:
            c, a = store.get((cty, t), (0, 0))
            c, a = c - own, a - (eq or 0)
            rates.append((a + TOK_M * pri) / (c + TOK_M))
            cnts.append(c)
            dfs.append(ts["df"].get((cty, t), 0))
        o[ia] = min(rates)
        if ia == 2:
            o[3] = sum(rates) / len(rates)
        o[il] = math.log1p(max(min(cnts), 0))
        o[idf] = math.log1p(max(dfs))


def _tok_loop(bounds):
    lo, hi = bounds
    g = _G
    out = np.full((hi - lo, len(TOK_COLS)), np.nan, np.float32)
    for r, i in enumerate(range(lo, hi)):
        eq = num_agree(g["ahn"][i], g["bhn"][i], g["anj"][i], g["bnj"][i])
        ex, mi = tok_diff(g["anf"][i], g["bnf"][i])
        _tok_feats(out[r], ex, mi, eq, g["cty"][i], g["ts"])
    return out


def hn_relation(a, b):
    """0 equal, 1 one digit dropped, 2 within 50, 3 further, 4 one side missing, 5 both missing."""
    if not a or not b:
        return 5 if a == b else 4
    if a == b:
        return 0
    if abs(len(a) - len(b)) == 1:
        lo_, sh = (a, b) if len(a) > len(b) else (b, a)
        if any(lo_[:i] + lo_[i + 1:] == sh for i in range(len(lo_))):
            return 1
    return 2 if abs(int(a) - int(b)) <= 50 else 3


def _cp(scorer, x, y):
    return process.cpdist(x, y, scorer=scorer, workers=WORKERS, dtype=np.float32)


def build(rec, pairs, idf, ts):
    """rec: records indexed by int id. pairs: s1, cand + retrieval columns (whole S1 groups).
    idf: idf_tables(rec); ts: token_stats(rec, all candidate pairs of the split)."""
    ia, ib = rec.index.get_indexer(pairs.s1.values), rec.index.get_indexer(pairs.cand.values)
    col = {c: (rec[c].values[ia], rec[c].values[ib]) for c in ["nf", "nc", "af", "nj", "ns"]}
    (an, bn), (ac, bc), (aa, ba) = col["nf"], col["nc"], col["af"]
    f = pd.DataFrame({"s1": pairs.s1.values, "cand": pairs.cand.values})
    f["fnum_c"] = [s.split(" ", 1)[0] for s in col["nj"][1]]

    # retrieval
    for c in ["cos", "frank", "rrank", "s1_top", "c_top"]:
        f[c] = pairs[c].values.astype(np.float32)
    f["gap_s1"] = f.s1_top - f.cos
    f["gap_c"] = f.c_top - f.cos

    # names
    f["n_ratio"] = _cp(fuzz.ratio, an, bn)
    f["n_tsort"] = _cp(fuzz.token_sort_ratio, an, bn)
    f["n_tset"] = _cp(fuzz.token_set_ratio, an, bn)
    f["c_ratio"] = _cp(fuzz.ratio, ac, bc)
    f["c_tset"] = _cp(fuzz.token_set_ratio, ac, bc)
    f["c_partial"] = _cp(fuzz.partial_ratio, ac, bc)
    f["c_jw"] = _cp(JaroWinkler.normalized_similarity, ac, bc)
    acp, bcp = [s.replace(" ", "") for s in ac], [s.replace(" ", "") for s in bc]
    f["cp_ratio"] = _cp(fuzz.ratio, acp, bcp)
    f["cp_partial"] = _cp(fuzz.partial_ratio, acp, bcp)

    # addresses (NaN when either side is empty)
    empty = (aa == "") | (ba == "")
    for name, sc in [("a_ratio", fuzz.ratio), ("a_tset", fuzz.token_set_ratio),
                     ("a_tsort", fuzz.token_sort_ratio), ("a_partial", fuzz.partial_ratio)]:
        f[name] = np.where(empty, np.nan, _cp(sc, aa, ba))
    ad, bd = [s.replace(" ", "") for s in col["ns"][0]], [s.replace(" ", "") for s in col["ns"][1]]
    f["d_ratio"] = np.where((np.array(ad) == "") | (np.array(bd) == ""), np.nan, _cp(fuzz.ratio, ad, bd))
    af_ = [s.split(" ", 1)[0] for s in col["nj"][0]]
    f["fn_lev"] = np.where((np.array(af_) == "") | (f.fnum_c.values == ""), np.nan,
                           _cp(Levenshtein.distance, af_, f.fnum_c.values))

    # set / idf features (python loop, forked workers)
    _G.update(anj=col["nj"][0], bnj=col["nj"][1], ans=col["ns"][0], bns=col["ns"][1], anc=ac, bnc=bc,
              aaf=aa, baf=ba, idf_n=idf[0], idf_a=idf[1])
    n = len(f)
    step = max(1, -(-n // (WORKERS * 4)))
    with C.fork_pool(WORKERS) as p:
        loop = np.concatenate(p.map(_loop, [(i, min(i + step, n)) for i in range(0, n, step)]))
    _G.clear()
    for j, c in enumerate(LOOP_COLS):
        f[c] = loop[:, j]

    # house number relation + sibling-token evidence
    ha, hb = rec.hn.values[ia], rec.hn.values[ib]
    f["hn_rel"] = [hn_relation(x, y) for x, y in zip(ha, hb)]
    f["hn_ldiff"] = [math.log1p(abs(int(x) - int(y))) if x and y else np.nan for x, y in zip(ha, hb)]
    _G.update(anf=an, bnf=bn, ahn=ha, bhn=hb, anj=col["nj"][0], bnj=col["nj"][1],
              cty=rec.country.values[ia], ts=ts)
    with C.fork_pool(WORKERS) as p:
        tok = np.concatenate(p.map(_tok_loop, [(i, min(i + step, n)) for i in range(0, n, step)]))
    _G.clear()
    for j, c in enumerate(TOK_COLS):
        f[c] = tok[:, j]

    # record-level flags
    for side, idx in (("s1", ia), ("c", ib)):
        f[f"aemp_{side}"] = (rec.af.values[idx] == "").astype(np.float32)
        f[f"dom_{side}"] = rec.dom.values[idx].astype(np.float32)
        f[f"ntok_{side}"] = [s.count(" ") + 1 for s in rec.nc.values[idx]]
        f[f"alen_{side}"] = [s.count(" ") + 1 if s else 0 for s in rec.af.values[idx]]
    for c in FREQ:
        f[f"{c}_s1"], f[f"{c}_c"] = rec[c].values[ia], rec[c].values[ib]
    f["src_c"] = rec.src.values[ib].astype(np.float32)
    f["nl_name_c"] = rec.nl_name.values[ib].astype(np.float32)
    f["nl_addr_c"] = rec.nl_addr.values[ib].astype(np.float32)

    # group context within each S1's candidate list
    g = f.groupby("s1")
    f["g_size"] = g.cand.transform("size").astype(np.float32)
    f["cos_rank"] = g.cos.rank(ascending=False, method="first").astype(np.float32)
    for c in ["c_tset", "n_ratio", "a_tset", "cp_partial"]:
        f[f"{c}_gap"] = g[c].transform("max") - f[c]
    f["g_hn_eq"] = (f.hn_rel == 0).groupby(f.s1).transform("sum").astype(np.float32)
    has = f.fnum_c != ""
    key = [f.s1, f.fnum_c]
    f["g_same_fn"] = (f.groupby(key).cand.transform("size") - 1).where(has).astype(np.float32)
    top = (f.frank <= 10).astype(np.float32)
    f["g_same_fn_top"] = (top.groupby(key).transform("sum") - top).where(has)
    f["g_s1fn"] = (f.fn_eq == 1).groupby(f.s1).transform("sum").astype(np.float32)
    return f.astype({c: np.float32 for c in f.columns if c not in AUX})


def stage2(f, p1):
    """Context over stage-1 probabilities: is this candidate consistent with the S1's likely cluster?"""
    f = f.copy()
    f["p1"] = p1
    g = f.groupby("s1").p1
    f["p_max"] = g.transform("max")
    f["p_sum"] = g.transform("sum")
    f["p_rank"] = g.rank(ascending=False, method="first").astype(np.float32)
    f["p_rel"] = f.p1 / f.p_max.clip(lower=1e-6)
    f["p_n50"] = (f.p1 > 0.5).groupby(f.s1).transform("sum").astype(np.float32)
    f["p_rank_src"] = f.groupby(["s1", "src_c"]).p1.rank(ascending=False, method="first").astype(np.float32)
    has = f.fnum_c != ""
    w = f.p1.where(has, 0.0).groupby([f.s1, f.fnum_c]).transform("sum")
    f["fn_support"] = ((w - f.p1) / (f.p_sum - f.p1).clip(lower=1e-6)).where(has)
    f["fn_top"] = (w == w.groupby(f.s1).transform("max")).astype(np.float32).where(has)
    f["s1fn_support"] = (f.p1 * (f.fn_eq == 1)).groupby(f.s1).transform("sum") / f.p_sum.clip(lower=1e-6)
    return f
