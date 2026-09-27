"""Shared I/O, text normalization and the F0.5 metric."""
import csv
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from anyascii import anyascii

def device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def fork_pool(n):
    """fork (not macOS's default spawn) so workers inherit module globals (maps, idf tables)."""
    import multiprocessing
    return multiprocessing.get_context("fork").Pool(n)


PREFIX = "query: "  # e5 convention; used for every text on both sides
SRC_BASE = 10**10    # int id = src * SRC_BASE + numeric part (ids have no leading zeros, <= 9 digits)


# ---------------------------------------------------------------- I/O
def read_tsv(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)


def id_to_int(s):
    s = pd.Series(s, dtype=str)
    return s.str[1].astype(np.int64) * SRC_BASE + s.str[3:].astype(np.int64)


def int_to_id(x):
    x = np.asarray(x, dtype=np.int64)
    return [f"S{a}-{b}" for a, b in zip(x // SRC_BASE, x % SRC_BASE)]


# ---------------------------------------------------------------- normalization tables
ZW = dict.fromkeys(map(ord, "​‌‍﻿"), None)
NONLATIN = re.compile(r"[^\x00-˿ -⁯€№]")  # anything beyond Latin + punctuation

HONOR = {"mr", "mrs", "ms", "smt", "shri", "sri", "dr", "the", "messrs", "mme", "m"}
NAME_ABBR = {
    "pvt": "private", "pvte": "private", "ltd": "limited", "ltda": "limited", "inc": "incorporated",
    "corp": "corporation", "co": "company", "cos": "companies", "intl": "international",
    "mfg": "manufacturing", "bros": "brothers", "assoc": "associates", "svc": "services", "svcs": "services",
    "mgmt": "management", "ctr": "center", "centre": "center", "natl": "national", "cie": "compagnie",
    "ste": "societe", "sté": "societe", "and": "and",
}
# legal forms and generic words the noise generator appends/drops; removed for the "core" name
LEGAL = {
    "private", "limited", "llc", "llp", "lp", "pc", "pllc", "plc", "incorporated", "corporation", "company",
    "sarl", "sas", "sasu", "eurl", "sa", "sci", "ei", "snc", "compagnie", "societe", "services", "service",
    "group", "center", "holdings", "partners", "associates", "and", "of", "india", "france", "usa", "us",
}
DOMAIN = re.compile(r"(?:www\.)?([a-z0-9\-]+)\.(?:com|net|org|co\.in|in|co|fr|us|biz|info|io)")
DOTTED = re.compile(r"\b(?:[a-z]\.){2,}")

US_STATES = dict(
    al="alabama", ak="alaska", az="arizona", ar="arkansas", ca="california", co="colorado", ct="connecticut",
    de="delaware", fl="florida", ga="georgia", hi="hawaii", id="idaho", il="illinois", ia="iowa", ks="kansas",
    ky="kentucky", la="louisiana", me="maine", md="maryland", ma="massachusetts", mi="michigan",
    mn="minnesota", ms="mississippi", mo="missouri", mt="montana", ne="nebraska", nv="nevada",
    nh="new hampshire", nj="new jersey", nm="new mexico", ny="new york", nc="north carolina",
    nd="north dakota", oh="ohio", ok="oklahoma", pa="pennsylvania", ri="rhode island", sc="south carolina",
    sd="south dakota", tn="tennessee", tx="texas", ut="utah", vt="vermont", va="virginia", wa="washington",
    wv="west virginia", wi="wisconsin", wy="wyoming", dc="district of columbia", pr="puerto rico",
    mp="northern mariana islands", gu="guam", vi="virgin islands",
)
IN_STATES = dict(
    ap="andhra pradesh", ar="arunachal pradesh", **{"as": "assam"}, br="bihar", cg="chhattisgarh",
    ct="chhattisgarh", ga="goa", gj="gujarat", hr="haryana", hp="himachal pradesh", jh="jharkhand",
    ka="karnataka", kl="kerala", mp="madhya pradesh", mh="maharashtra", mn="manipur", ml="meghalaya",
    mz="mizoram", nl="nagaland", od="odisha", **{"or": "odisha"}, pb="punjab", rj="rajasthan", sk="sikkim",
    tn="tamil nadu", tg="telangana", ts="telangana", tr="tripura", up="uttar pradesh", uk="uttarakhand",
    ut="uttarakhand", wb="west bengal", an="andaman and nicobar islands", ch="chandigarh",
    dn="dadra and nagar haveli", dd="daman and diu", dl="delhi", jk="jammu and kashmir", la="ladakh",
    ld="lakshadweep", py="puducherry", orissa="odisha", pondicherry="puducherry", uttaranchal="uttarakhand",
)
STATES = {"US": US_STATES, "India": IN_STATES}  # unknown countries: no state mapping

ADDR_COMMON = {
    "rd": "road", "st": "street", "str": "street", "ave": "avenue", "av": "avenue", "blvd": "boulevard",
    "dr": "drive", "ln": "lane", "ct": "court", "cir": "circle", "pl": "place", "hwy": "highway",
    "pkwy": "parkway", "sq": "square", "bldg": "building", "flr": "floor", "fl": "floor",
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5", "sixth": "6", "seventh": "7",
    "eighth": "8", "ninth": "9", "tenth": "10", "eleventh": "11", "twelfth": "12",
}
ADDR_BY_COUNTRY = {
    "US": {"n": "north", "s": "south", "e": "east", "w": "west", "ne": "northeast", "nw": "northwest",
           "se": "southeast", "sw": "southwest", "ter": "terrace", "trl": "trail", "mt": "mount",
           "ft": "fort", "pt": "point", "cv": "cove", "xing": "crossing", "expy": "expressway"},
    "India": {"nr": "near", "opp": "opposite", "apts": "apartment", "apt": "apartment", "dist": "district",
              "ngr": "nagar", "tq": "taluk", "tal": "taluk", "bombay": "mumbai", "calcutta": "kolkata",
              "madras": "chennai", "bangalore": "bengaluru", "gurgaon": "gurugram", "poona": "pune",
              "hissar": "hisar", "ahmadabad": "ahmedabad", "baroda": "vadodara", "cochin": "kochi",
              "calicut": "kozhikode", "trivandrum": "thiruvananthapuram", "mysore": "mysuru"},
    "France": {"r": "rue", "bd": "boulevard", "boul": "boulevard", "all": "allee", "imp": "impasse",
               "che": "chemin", "chem": "chemin", "q": "quai", "rte": "route", "st": "saint", "ste": "sainte",
               "fg": "faubourg", "fbg": "faubourg", "crs": "cours", "dr": "docteur", "gal": "general",
               "mal": "marechal", "pres": "president"},
}
ADDR_DROP = {"city", "cty", "unit", "apt", "ste", "suite", "pmb", "po", "box", "no", "hno", "door", "hn",
             "null", "none", "nan", "na"}
NULLS = re.compile(r"<\s*null\s*>|\bn/a\b|\bnull\b")
NUMSIGN = re.compile(r"\bn\s*[°º]|№|#|\bh\.?\s*no\b\.?|\bno\.")
ORDINAL = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b")
JOIN = re.compile(r"(?<=[0-9a-z])[-/](?=[0-9a-z])")
NUM_J = re.compile(r"\d+[a-z]?(?![a-z0-9])")
LEAD0 = re.compile(r"^0+(?=\d)")


# ---------------------------------------------------------------- normalizers
def translit(text, tok_map):
    """Native-script tokens -> learned Latin token when known; the rest via anyascii."""
    text = text.translate(ZW)
    if NONLATIN.search(text):
        text = " ".join(tok_map.get(t, t) for t in text.split())
    return anyascii(text)


def norm_name(s, tok_map):
    """-> (full, core, is_domain)."""
    s = translit(s, tok_map).lower().strip()
    m = DOMAIN.fullmatch(s)
    if m:
        s = m.group(1).replace("-", " ")
    s = re.sub(r"\bm/s\b", " ", s)
    s = re.sub(r"\d{7,}", " ", s)  # phone numbers glued to names
    s = s.replace("&", " and ").replace("+", " plus ")
    s = DOTTED.sub(lambda x: x.group(0).replace(".", ""), s)  # l.l.c. -> llc
    toks = [NAME_ABBR.get(t, t) for t in re.findall(r"[a-z0-9]+", s) if t not in HONOR]
    toks = [t for i, t in enumerate(toks) if i == 0 or t != toks[i - 1]]
    full = " ".join(toks)
    core = " ".join(t for t in toks if t not in LEGAL) or full
    return full, core, m is not None


def norm_addr(s, country, comp_map, tok_map):
    """-> (address, joined-number tokens, split-number tokens)."""
    states = STATES.get(country, {})
    comps = []
    for c in s.translate(ZW).split(","):
        c = c.strip()
        if not c:
            continue
        if NONLATIN.search(c):
            c = comp_map.get(c) or translit(c, tok_map)
        comps.append(states.get(c.lower().strip(". "), c))
    s = NUMSIGN.sub(" ", ", ".join(comps).lower())
    s = NULLS.sub(" ", anyascii(s))
    s = ORDINAL.sub(r"\1", s)
    nums_j = [LEAD0.sub("", n) for n in NUM_J.findall(JOIN.sub("", s))]
    nums_s = [LEAD0.sub("", n) for n in re.findall(r"\d+", s)]
    abbr = ADDR_BY_COUNTRY.get(country, {})
    toks = []
    for t in re.findall(r"[a-z0-9]+", s):
        t = abbr.get(t) or ADDR_COMMON.get(t) or t
        if t.isdigit():
            t = LEAD0.sub("", t)
        if t not in ADDR_DROP:
            toks.extend(t.split())
    return " ".join(toks), " ".join(nums_j), " ".join(nums_s)


def normalize_row(args):
    """Worker: (name, addr, country) -> record columns. Maps come via module globals (fork)."""
    name, addr, country = args
    nf, nc, dom = norm_name(name, _MAPS["tok"])
    af, nj, ns = norm_addr(addr, country, _MAPS["comp"], _MAPS["tok"])
    return nf, nc, dom, af, nj, ns, bool(NONLATIN.search(name)), bool(NONLATIN.search(addr))


_MAPS = {"tok": {}, "comp": {}}


# ---------------------------------------------------------------- learned transliteration
def learn_maps(pairs):
    """pairs: iterable of (s1_name, s1_addr, cand_name, cand_addr) for true matches.

    Name tokens: positional alignment when the native name and the normalized S1 name
    have the same token count. Address components: Dice-style co-occurrence of a
    native component with the S1's address components.
    """
    tok_votes = defaultdict(Counter)
    c_t, c_w, c_tw = Counter(), Counter(), Counter()
    for s1n, s1a, cn, ca in pairs:
        cn = cn.translate(ZW)
        if NONLATIN.search(cn):
            ct = cn.lower().split()
            st = norm_name(s1n, {})[0].split()
            if len(ct) == len(st):
                for a, b in zip(ct, st):
                    if NONLATIN.search(a):
                        tok_votes[a][b] += 1
        native = [c.strip() for c in ca.translate(ZW).split(",") if NONLATIN.search(c)]
        if native:
            ws = {c.strip() for c in s1a.split(",") if c.strip()}
            for w in ws:
                c_w[w] += 1
            for t in set(native):
                c_t[t] += 1
                for w in ws:
                    c_tw[t, w] += 1
    tok = {}
    for a, votes in tok_votes.items():
        b, n = votes.most_common(1)[0]
        if n >= 2 and n >= 0.5 * sum(votes.values()):
            tok[a] = b
    best = {}
    for (t, w), n in c_tw.items():
        if n >= 3:
            sc = n * n / (c_t[t] * c_w[w])
            if sc > best.get(t, (0, None))[0]:
                best[t] = (sc, w)
    return tok, {t: w for t, (_, w) in best.items()}


# ---------------------------------------------------------------- metric
def f05_per_entity(pred, truth, s1_ids):
    """pred/truth: DataFrame[s1, cand] (int ids). Returns per-S1 F0.5 Series (README rules)."""
    d = pd.DataFrame(index=pd.Index(np.unique(s1_ids), name="s1"))
    d["tp"] = pred.merge(truth, on=["s1", "cand"]).groupby("s1").size()
    d["np"] = pred.groupby("s1").size()
    d["nt"] = truth.groupby("s1").size()
    d = d.fillna(0)
    p = d.tp / d.np.clip(lower=1)
    r = d.tp / d.nt.clip(lower=1)
    f = (1.25 * p * r / (0.25 * p + r).clip(lower=1e-12)).where(d.tp > 0, 0.0)
    f[(d.np == 0) & (d.nt == 0)] = 1.0
    return f


def decide(df, a=1.0, b=0.0, mode="expf", thr=0.5):
    """df: s1, cand, p -> selected DataFrame[s1, cand].

    1. calibrate: p' = sigmoid(a * logit(p) + b)
    2. one owner: each S2/S3 id stays only with its highest-p S1 (ground truth never shares one)
    3. per S1, either threshold (mode='thr') or pick the top-k maximizing expected F0.5,
       E[F] ~ 1.25 * sum(p_top_k) / (0.25 * sum(p_all) + k), vs. predicting nothing, E = prod(1 - p).
    """
    p = df.p.to_numpy(np.float64).clip(1e-7, 1 - 1e-7)
    p = 1 / (1 + np.exp(-(a * np.log(p / (1 - p)) + b)))
    d = pd.DataFrame({"s1": df.s1.values, "cand": df.cand.values, "p": p})
    d = d.sort_values("p", ascending=False, kind="stable").drop_duplicates("cand")
    if mode == "thr":
        return d.loc[d.p >= thr, ["s1", "cand"]]
    d = d.sort_values(["s1", "p"], ascending=[True, False], kind="stable")
    g = d.groupby("s1").p
    k = d.groupby("s1").cumcount() + 1
    ef = 1.25 * g.cumsum() / (0.25 * g.transform("sum") + k)
    best = ef.groupby(d.s1).transform("max")
    kbest = k.where(ef == best).groupby(d.s1).transform("min")
    empty = np.exp(np.log1p(-d.p).groupby(d.s1).transform("sum"))
    return d.loc[(k <= kbest) & (best > empty), ["s1", "cand"]]


if __name__ == "__main__":  # self-check
    print(norm_name("Mr >> Dholi Mótors Pvt. Ltd. (LLC)", {}))
    print(norm_name("speedholdingsgroup.com", {}))
    print(norm_addr("00207 Marston St, <NULL>, Copperas Cove, TX", "US", {}, {}))
    print(norm_addr("No 5-7-A, Mysore Colony, Chembur (E), Bombay, MH", "India", {}, {}))
    print(norm_addr("N° 41 R. DU VAL SAINT-MARTIN, PORNIC", "France", {}, {}))
    assert norm_name("Ear Nose & Throat P.C.", {})[1] == "ear nose throat"
    assert norm_addr("8-2-293/83A, Road No. 44", "India", {}, {})[1] == "8229383a 44"
    t = pd.DataFrame({"s1": [1, 1, 2], "cand": [10, 11, 20]})
    p = pd.DataFrame({"s1": [1, 1, 1, 3], "cand": [10, 11, 12, 30]})
    f = f05_per_entity(p, t, [1, 2, 3, 4])
    assert abs(f[1] - 0.714) < 1e-3 and f[2] == 0 and f[3] == 0 and f[4] == 1, f
    d = pd.DataFrame({"s1": [1, 1, 1, 2, 2, 3], "cand": [10, 11, 12, 12, 20, 30], "p": [.95, .9, .3, .6, .02, .1]})
    sel = decide(d)
    assert set(map(tuple, sel.values)) == {(1, 10), (1, 11), (2, 12)}, sel  # 12 owned by S1 2, S1 3 empty
    print("ok")
