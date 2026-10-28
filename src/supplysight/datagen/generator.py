"""Seeded synthetic SAP MM/PP data generator for a cosmetics manufacturer.

The simulation is causal, not random noise:

1. Finished-goods (FERT) demand follows category seasonality (Indian festive season,
   wedding season, summer/winter) plus trend and noise.
2. Weekly production orders (AFKO/AFPO) are planned from that demand.
3. Raw-material and packaging consumption (MSEG movement 261) is *derived* from
   production orders x bill of materials (STKO/STPO) plus scrap noise.
4. A weekly inventory simulation per material/plant raises purchase requisitions
   (EBAN) and purchase orders (EKKO/EKPO) when the inventory position drops below a
   reorder level; goods receipts (MSEG 101) arrive after vendor-specific lead times.
5. Labelled anomalies are injected into ~3-5 % of PO items (ground truth goes to the
   separate ``anomaly_labels`` table which models never see).

All names, GSTINs and bank references are invented. Column and table names are
lower-case SAP names so the same SQL works on SQLite and PostgreSQL.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from supplysight.datagen import catalog as C

ANOMALY_TYPES = [
    "price_spike",
    "quantity_outlier",
    "split_po",
    "offhours_new_vendor",
    "short_lead_time",
    "duplicate_vendor",
]
_ANOMALY_MIX = {
    "price_spike": 0.24,
    "quantity_outlier": 0.20,
    "split_po": 0.16,
    "offhours_new_vendor": 0.14,
    "short_lead_time": 0.14,
    "duplicate_vendor": 0.12,
}
_GST_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass
class GenConfig:
    """Configuration for one generator run."""

    seed: int = 42
    start: str = "2024-10-01"
    end: str = "2026-09-30"
    trend_anchor: str = "2024-10-01"
    n_vendors: int = 60
    anomaly_rate: float = 0.04
    approval_threshold: float = 500_000.0
    material_scale: float = 1.0  # <1.0 builds a smaller catalogue (CI / tests)
    id_offset: int = 0  # shifts document numbers (used for drift quarters)
    drift: dict = field(default_factory=dict)  # see scripts/simulate_drift.py


# --------------------------------------------------------------------------- helpers
def gstin_checksum(first14: str) -> str:
    """Return the GSTIN check character for the first 14 characters (mod-36 scheme)."""
    total = 0
    for i, ch in enumerate(first14):
        val = _GST_CHARS.index(ch)
        prod = val * (2 if i % 2 else 1)
        total += prod // 36 + prod % 36
    return _GST_CHARS[(36 - total % 36) % 36]


def make_gstin(rng: np.random.Generator, state_code: str, pan: str | None = None,
               entity: str = "1") -> str:
    """Build a format-valid but fake GSTIN (state code + PAN + entity + 'Z' + checksum)."""
    if pan is None:
        letters = "".join(rng.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), 5))
        letters = letters[:3] + "C" + letters[4]  # 4th char C = company
        pan = letters + f"{rng.integers(0, 10000):04d}" + str(rng.choice(list("ABCDEFGHJKLMNPQRSTUVWXYZ")))
    first14 = f"{state_code}{pan}{entity}Z"
    return first14 + gstin_checksum(first14)


def is_valid_gstin(gstin: str) -> bool:
    """Validate GSTIN format and checksum."""
    if not re.fullmatch(r"\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]", gstin or ""):
        return False
    return gstin_checksum(gstin[:14]) == gstin[14]


def _fake_bank_ref(rng: np.random.Generator) -> str:
    bank = rng.choice(["HDFC", "ICIC", "SBIN", "UTIB", "KKBK", "PUNB", "BARB"])
    return f"{bank}0{rng.integers(0, 999999):06d}/{rng.integers(10**13, 10**14 - 1)}"


def _unit_mass_kg(desc: str) -> float:
    m = re.search(r"(\d+)\s*(g|ml)\b", desc)
    if m:
        return int(m.group(1)) / 1000.0
    if "Lipstick" in desc or "Lip Balm" in desc:
        return 0.0045
    if "Kajal" in desc or "Mascara" in desc:
        return 0.006
    return 0.03


def season_factor(profile: str, d: date) -> float:
    """Demand multiplier for a seasonality profile on a given (sales) date."""
    m = d.month
    table = {
        "festive": {9: 1.15, 10: 1.50, 11: 1.35, 12: 1.05, 3: 1.05},
        "wedding": {11: 1.30, 12: 1.30, 1: 1.25, 2: 1.20, 4: 1.15, 5: 1.15, 7: 0.85, 8: 0.85},
        "summer": {3: 1.20, 4: 1.60, 5: 1.75, 6: 1.45, 7: 1.10, 11: 0.70, 12: 0.65, 1: 0.70},
        "winter": {10: 1.15, 11: 1.40, 12: 1.55, 1: 1.50, 2: 1.20, 4: 0.75, 5: 0.70, 6: 0.70, 7: 0.80},
        "flat": {},
    }
    f = table.get(profile, {}).get(m, 1.0)
    if profile != "summer" and m in (10, 11):
        f *= 1.08  # general Diwali uplift across the portfolio
    return f


def _variants(base: list[tuple], suffixes: list[str], n: int, rng: np.random.Generator) -> list[tuple]:
    idx = rng.choice(len(base), size=min(n, len(base)), replace=False)
    out = []
    for k, i in enumerate(sorted(idx)):
        row = list(base[i])
        row[0] = f"{row[0]} {suffixes[k % len(suffixes)]}"
        out.append(tuple(row))
    return out


# --------------------------------------------------------------------------- master data
def build_master_data(cfg: GenConfig) -> dict[str, pd.DataFrame]:
    """Create vendor, material, plant, info-record and BOM master data."""
    rng = np.random.default_rng(cfg.seed)
    start = pd.Timestamp(cfg.start)

    raw = list(C.RAW_MATERIALS) + _variants(C.RAW_MATERIALS, ["(Imported)", "- Organic", "- Ecocert", "Premium Grade"], 46, rng)
    pack = list(C.PACKAGING) + _variants(C.PACKAGING, ["- Amber", "- Printed", "- Recycled PCR", "- Matte"], 30, rng)
    fg = list(C.FINISHED) + [
        (f"{d.rsplit(' ', 2)[0]} Mini", g, u, int(b * 0.45), p)
        for (d, g, u, b, p) in [C.FINISHED[i] for i in sorted(rng.choice(len(C.FINISHED), 12, replace=False))]
    ]
    if cfg.material_scale < 1.0:
        k = cfg.material_scale
        raw = raw[: max(20, int(len(raw) * k))]
        pack = pack[: max(10, int(len(pack) * k))]
        fg = fg[: max(8, int(len(fg) * k))]

    # ---- MARA / MAKT
    mats = []
    for i, (d, g, u) in enumerate(raw):
        mats.append({"matnr": str(100001 + i), "mtart": "ROH", "matkl": g, "meins": u, "maktx": d,
                     "base_price": C.GROUP_PRICE[g] * float(rng.lognormal(0, 0.30)) * (1.3 if "Imported" in d or "Organic" in d else 1.0)})
    for i, (d, g, u) in enumerate(pack):
        mats.append({"matnr": str(200001 + i), "mtart": "VERP", "matkl": g, "meins": u, "maktx": d,
                     "base_price": C.GROUP_PRICE[g] * float(rng.lognormal(0, 0.25))})
    fg_meta = {}
    for i, (d, g, u, base, prof) in enumerate(fg):
        m = str(300001 + i)
        mats.append({"matnr": m, "mtart": "FERT", "matkl": g, "meins": u, "maktx": d, "base_price": np.nan})
        fg_meta[m] = {"base": base, "profile": prof, "group": g, "mass": _unit_mass_kg(d) * (0.4 if "Mini" in d else 1.0)}
    mat_df = pd.DataFrame(mats)
    mara = mat_df[["matnr", "mtart", "matkl", "meins"]].copy()
    mara["ersda"] = "2018-04-01"
    makt = mat_df[["matnr", "maktx"]].copy()
    makt["spras"] = "EN"

    # ---- vendors (LFA1)
    n_pk = max(4, int(round(cfg.n_vendors * 0.3)))
    n_rm = cfg.n_vendors - n_pk
    rm_groups = sorted({g for _, g, _ in raw})
    pk_groups = sorted({g for _, g, _ in pack})
    prefixes = list(rng.permutation(C.VENDOR_PREFIX))
    vendors = []
    for i in range(cfg.n_vendors):
        is_pk = i >= n_rm
        city, state, code = C.CITIES[rng.integers(len(C.CITIES))]
        suffix = rng.choice(C.VENDOR_SUFFIX_PK if is_pk else C.VENDOR_SUFFIX_RM)
        name = f"{prefixes[i % len(prefixes)]} {suffix} {rng.choice(C.LEGAL_FORMS)}"
        groups = list(rng.choice(pk_groups if is_pk else rm_groups, size=int(rng.integers(2, 5)), replace=False))
        created = pd.Timestamp("2014-01-01") + pd.Timedelta(days=int(rng.integers(0, 3600)))
        vendors.append({
            "lifnr": str(500001 + i), "name1": name,
            "stras": f"Plot {rng.integers(1, 400)}, {rng.choice(C.STREETS)}",
            "ort01": city, "regio": state, "pstlz": f"{rng.integers(110001, 799999)}",
            "stcd3": make_gstin(rng, code), "bank_ref": _fake_bank_ref(rng),
            "created_on": min(created, start - pd.Timedelta(days=30)).date().isoformat(),
            "vendor_type": "PK" if is_pk else "RM", "groups": groups,
            "price_level": float(rng.lognormal(0, 0.07)),
            "lt_mean": float(rng.uniform(5, 15) if is_pk else rng.uniform(7, 28)),
            "reliability_sd": float(rng.uniform(0.5, 4.0)),
        })
    # three legitimate new vendors onboarded during the period (normal behaviour)
    for j, idx in enumerate(rng.choice(cfg.n_vendors, 3, replace=False)):
        vendors[idx]["created_on"] = (start + pd.Timedelta(days=int(120 + 160 * j))).date().isoformat()
    # two legitimate branch registrations: same PAN, different state GSTIN (normal in India)
    for idx in rng.choice(cfg.n_vendors, 2, replace=False):
        parent = dict(vendors[idx])
        city, state, code = next(c for c in C.CITIES[::-1] if c[2] != parent["stcd3"][:2])
        parent.update({"lifnr": str(500001 + len(vendors)), "name1": parent["name1"] + f" ({city} Branch)",
                       "stras": f"Unit {rng.integers(1, 90)}, {rng.choice(C.STREETS)}", "ort01": city,
                       "regio": state, "stcd3": make_gstin(rng, code, pan=parent["stcd3"][2:12]),
                       "bank_ref": _fake_bank_ref(rng)})
        vendors.append(parent)
    vend_df = pd.DataFrame(vendors)

    # ---- plants for finished goods
    fg_plants = {}
    for m in fg_meta:
        r = rng.random()
        fg_plants[m] = ["1000", "2000"] if r < 0.4 else (["1000"] if r < 0.7 else ["2000"])

    # ---- BOM (STKO/STPO): base quantity 1000 EA of FG
    mats_by_group: dict[str, list[str]] = mat_df.groupby("matkl")["matnr"].apply(list).to_dict()
    popularity = {m: float(rng.pareto(1.5) + 1) for m in mat_df.matnr}
    stko, stpo = [], []
    used: set[str] = set()
    for n, (m, meta) in enumerate(fg_meta.items()):
        stlnr = f"{10000001 + n}"
        stko.append({"stlnr": stlnr, "matnr": m, "bmeng": 1000.0, "bmein": "EA"})
        comps: list[tuple[str, float]] = []
        for g in C.FG_RECIPES[meta["group"]]:
            if g not in mats_by_group:
                continue
            cands = mats_by_group[g]
            p = np.array([popularity[c] for c in cands])
            k = 2 if (g in ("RM-EMOLL", "RM-ACTIVE", "RM-PIGM") and rng.random() < 0.5 and len(cands) > 1) else 1
            for c in rng.choice(cands, size=k, replace=False, p=p / p.sum()):
                small = g in ("RM-ACTIVE", "RM-FRAG", "RM-PRESV", "RM-PH", "RM-PIGM", "RM-UVFIL")
                frac = rng.uniform(0.002, 0.012) if small else rng.uniform(0.01, 0.09)
                comps.append((str(c), round(frac * meta["mass"] * 1000.0, 4)))
        for g in C.FG_PACK[meta["group"]]:
            if g not in mats_by_group:
                continue
            c = str(rng.choice(mats_by_group[g]))
            comps.append((c, 1000.0 * (2 if g == "PK-LABEL" and rng.random() < 0.3 else 1)))
        seen = set()
        for pos, (c, q) in enumerate([x for x in comps if not (x[0] in seen or seen.add(x[0]))]):
            stpo.append({"stlnr": stlnr, "posnr": f"{(pos + 1) * 10:04d}", "idnrk": c, "menge": q,
                         "meins": "KG" if c.startswith("1") else "EA"})
            used.add(c)
    # make sure every purchased material is consumed by at least one FG
    purchased = mat_df[mat_df.mtart != "FERT"]
    fg_by_group: dict[str, list[str]] = {}
    for m, meta in fg_meta.items():
        fg_by_group.setdefault(meta["group"], []).append(m)
    stl_of = {r["matnr"]: r["stlnr"] for r in stko}
    for _, row in purchased[~purchased.matnr.isin(used)].iterrows():
        compatible = [g for g in C.FG_RECIPES if row.matkl in C.FG_RECIPES[g] + C.FG_PACK[g] and g in fg_by_group]
        g = str(rng.choice(compatible)) if compatible else str(rng.choice(list(fg_by_group)))
        target = str(rng.choice(fg_by_group[g]))
        mass = fg_meta[target]["mass"]
        q = 1000.0 if row.mtart == "VERP" else round(rng.uniform(0.003, 0.03) * mass * 1000, 4)
        nxt = sum(1 for s in stpo if s["stlnr"] == stl_of[target]) + 1
        stpo.append({"stlnr": stl_of[target], "posnr": f"{nxt * 10:04d}", "idnrk": row.matnr, "menge": q,
                     "meins": "KG" if row.mtart == "ROH" else "EA"})
    stko_df, stpo_df = pd.DataFrame(stko), pd.DataFrame(stpo)

    # ---- info records (EINA / EINE)
    eina, eine = [], []
    infnr = 5300000001
    for _, mrow in purchased.iterrows():
        typ = "PK" if mrow.mtart == "VERP" else "RM"
        cands = vend_df[(vend_df.vendor_type == typ) & vend_df.groups.apply(lambda gs, g=mrow.matkl: g in gs)]
        if cands.empty:
            cands = vend_df[vend_df.vendor_type == typ]
        k = int(min(len(cands), rng.choice([1, 2, 2, 3])))
        chosen = cands.sample(n=k, random_state=int(rng.integers(1 << 30)))
        for rank, (_, v) in enumerate(chosen.iterrows()):
            lt = max(2, int(round(v.lt_mean * rng.uniform(0.85, 1.2) * (1.6 if "Imported" in str(mrow.maktx) else 1.0))))
            price = round(float(mrow.base_price) * v.price_level * rng.uniform(0.96, 1.05), 2)
            eina.append({"infnr": str(infnr), "matnr": mrow.matnr, "lifnr": v.lifnr,
                         "created_on": max(pd.Timestamp(v.created_on), pd.Timestamp("2018-01-01")).date().isoformat(),
                         "is_primary": int(rank == 0)})
            eine.append({"infnr": str(infnr), "ekorg": C.PURCH_ORG, "netpr": price, "peinh": 1,
                         "waers": "INR", "aplfz": lt,
                         "valid_from": max(pd.Timestamp(v.created_on), pd.Timestamp(cfg.start) - pd.Timedelta(days=365)).date().isoformat()})
            infnr += 1
    eina_df, eine_df = pd.DataFrame(eina), pd.DataFrame(eine)

    # ---- which plants use each purchased material
    comp_fg = stpo_df.merge(stko_df[["stlnr", "matnr"]], on="stlnr").rename(columns={"matnr": "fg"})
    mat_plants: dict[str, set[str]] = {}
    for _, r in comp_fg.iterrows():
        mat_plants.setdefault(r.idnrk, set()).update(fg_plants[r.fg])

    return {
        "mat_df": mat_df, "mara": mara, "makt": makt, "vendors": vend_df, "fg_meta": fg_meta,
        "fg_plants": fg_plants, "stko": stko_df, "stpo": stpo_df, "eina": eina_df, "eine": eine_df,
        "mat_plants": mat_plants,
    }


# --------------------------------------------------------------------------- transactions
def _weeks(cfg: GenConfig, extra_days: int = 0) -> list[pd.Timestamp]:
    s = pd.Timestamp(cfg.start)
    s = s - pd.Timedelta(days=s.weekday())  # Monday
    return list(pd.date_range(s, pd.Timestamp(cfg.end) + pd.Timedelta(days=extra_days), freq="W-MON"))


PLAN_HORIZON_DAYS = 28  # frozen production-plan horizon published as planned orders (PLAF)


def simulate_production(cfg: GenConfig, md: dict, rng: np.random.Generator
                        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Weekly production orders per FG and plant (AFKO header, AFPO item) plus planned
    orders (PLAF) for the frozen 4-week horizon after the data end date."""
    weeks = _weeks(cfg, PLAN_HORIZON_DAYS)
    end = pd.Timestamp(cfg.end)
    plaf = []
    plnum = 7000001 + cfg.id_offset
    anchor = pd.Timestamp(cfg.trend_anchor)
    surge = cfg.drift.get("demand_surge", {})
    afko, afpo = [], []
    aufnr = 1000001 + cfg.id_offset
    for m, meta in md["fg_meta"].items():
        for plant in md["fg_plants"][m]:
            share = 0.55 if len(md["fg_plants"][m]) == 2 and plant == "1000" else (0.45 if len(md["fg_plants"][m]) == 2 else 1.0)
            ar, carry = 0.0, 0.0
            for w in weeks:
                sales_date = (w + pd.Timedelta(days=21)).date()  # produce ~3 weeks before sale
                years = (w - anchor).days / 365.25
                ar = 0.6 * ar + rng.normal(0, 0.07)
                plan = meta["base"] * share * season_factor(meta["profile"], sales_date) * (1.06 ** years) * np.exp(ar)
                plan *= surge.get(meta["group"], 1.0)
                plan = max(0.0, round(plan / 50) * 50)
                if plan <= 0:
                    continue
                # execution deviates from plan: yield/batch-size noise, slipped orders, rush batches
                actual = plan * rng.lognormal(0.0, 0.12) + carry
                carry = 0.0
                r = rng.random()
                if r < 0.06:
                    carry, actual = actual, 0.0  # slipped into next week
                elif r < 0.10:
                    actual *= rng.uniform(1.4, 1.9)  # rush / extra batch
                actual = max(0.0, round(actual / 10) * 10)
                start_day = w + pd.Timedelta(days=int(rng.integers(0, 5)))
                if start_day > end:
                    if start_day <= end + pd.Timedelta(days=PLAN_HORIZON_DAYS):
                        plaf.append({"plnum": str(plnum), "matnr": m, "plwrk": plant, "gsmng": plan,
                                     "psttr": start_day.date().isoformat(), "paart": "LA"})
                        plnum += 1
                    continue
                afko.append({"aufnr": str(aufnr), "werks": plant, "plnbez": m, "gamng": plan,
                             "gstrp": start_day.date().isoformat(),
                             "gltrp": (start_day + pd.Timedelta(days=2)).date().isoformat(), "auart": "PP01"})
                afpo.append({"aufnr": str(aufnr), "posnr": "0001", "matnr": m, "psmng": plan, "wemng": actual,
                             "werks": plant})
                aufnr += 1
    plaf_cols = ["plnum", "matnr", "plwrk", "gsmng", "psttr", "paart"]
    return pd.DataFrame(afko), pd.DataFrame(afpo), pd.DataFrame(plaf, columns=plaf_cols)


def derive_consumption(afko: pd.DataFrame, afpo: pd.DataFrame, md: dict, rng: np.random.Generator,
                       cfg: GenConfig) -> pd.DataFrame:
    """Component issues (movement 261) = actual production quantity x BOM x scrap noise."""
    bom = md["stpo"].merge(md["stko"][["stlnr", "matnr", "bmeng"]], on="stlnr")
    po = afko.merge(afpo[["aufnr", "wemng"]], on="aufnr")
    x = po.merge(bom, left_on="plnbez", right_on="matnr", suffixes=("", "_bom"))
    scrap = rng.normal(0.012, 0.015, len(x)).clip(-0.02, 0.08)
    x["qty"] = x.wemng / x.bmeng * x.menge * (1 + scrap)
    x["qty"] = np.where(x.meins == "EA", np.ceil(x.qty), x.qty.round(3))
    out = pd.DataFrame({
        "aufnr": x.aufnr, "matnr": x.idnrk, "werks": x.werks, "menge": x.qty, "meins": x.meins,
        "budat": x.gstrp, "bwart": "261",
    })
    return out[out.menge > 0].reset_index(drop=True)


def simulate_procurement(cfg: GenConfig, md: dict, issues: pd.DataFrame,
                         rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    """Weekly inventory simulation that raises PRs/POs and books goods receipts."""
    weeks = _weeks(cfg)
    end = pd.Timestamp(cfg.end)
    anchor = pd.Timestamp(cfg.trend_anchor)
    price_shock = float(cfg.drift.get("price_shock", 1.0))
    new_vendor_share = float(cfg.drift.get("new_vendor_share", 0.0))
    vendors = md["vendors"].set_index("lifnr")
    info = md["eina"].merge(md["eine"], on="infnr")
    mat = md["mat_df"].set_index("matnr")

    iss = issues.copy()
    iss["week"] = pd.to_datetime(iss.budat) - pd.to_timedelta(pd.to_datetime(iss.budat).dt.weekday, unit="D")
    weekly = iss.groupby(["matnr", "werks", "week"]).menge.sum()

    pos, gr, eban, stock_end = [], [], [], []
    ebeln = 4500000001 + cfg.id_offset
    banfn = 10000001 + cfg.id_offset
    for (m, plant), series in weekly.groupby(level=[0, 1]):
        s = series.droplevel([0, 1]).reindex(weeks, fill_value=0.0)
        recs = info[info.matnr == m]
        if recs.empty:
            continue
        primary = recs[recs.is_primary == 1].iloc[0]
        lt_w = max(1.0, primary.aplfz / 7)
        avg0 = max(s.iloc[:8].mean(), 1e-3)
        stock = avg0 * (lt_w + 4)
        pipeline: list[tuple[pd.Timestamp, float]] = []
        cover = rng.uniform(3, 7)
        pack = 25.0 if mat.loc[m, "meins"] == "KG" else 500.0
        for i, w in enumerate(weeks):
            arrived = [q for (d, q) in pipeline if d < w + pd.Timedelta(days=7)]
            pipeline = [(d, q) for (d, q) in pipeline if d >= w + pd.Timedelta(days=7)]
            stock += sum(arrived) - s.iloc[i]
            hist = s.iloc[max(0, i - 8): i + 1]
            avg = max(hist.mean(), avg0 * 0.3)
            position = stock + sum(q for _, q in pipeline)
            if position >= avg * (lt_w + 1.5):
                continue
            # --- raise a PR and convert to a PO
            qty = max(pack, np.ceil(avg * cover * rng.uniform(0.85, 1.2) / pack) * pack)
            if rng.random() < 0.04:  # legit festive pre-buy / campaign stock
                qty = np.ceil(qty * rng.uniform(1.8, 3.5) / pack) * pack
            cand = recs.copy()
            w_valid = cand.lifnr.map(lambda lf: pd.Timestamp(vendors.loc[lf, "created_on"]) <= w)
            cand = cand[w_valid.values]
            if cand.empty:
                cand = recs[recs.is_primary == 1]
            p = np.where(cand.is_primary == 1, 0.7, 0.3 / max(1, len(cand) - 1))
            if len(cand) == 1:
                p = np.array([1.0])
            rec = cand.iloc[int(rng.choice(len(cand), p=p / p.sum()))]
            lifnr = rec.lifnr
            if new_vendor_share and rng.random() < new_vendor_share:
                lifnr = "DRIFT"  # replaced later by a drift vendor
            weekday = int(rng.choice([0, 1, 2, 3, 4, 5], p=[0.21, 0.21, 0.2, 0.18, 0.15, 0.05]))
            hour = int(np.clip(rng.normal(13, 2.8), 7, 21))
            created = w + pd.Timedelta(days=weekday, hours=hour, minutes=int(rng.integers(0, 60)))
            if created > end:
                continue
            years = (w - anchor).days / 365.25
            volatile = mat.loc[m, "matkl"] in ("RM-FRAG", "RM-BUTTER", "RM-OIL", "RM-ACTIVE")
            price = rec.netpr * (1.055 ** years) * price_shock * (
                rng.uniform(0.9, 1.12) if volatile else rng.uniform(0.97, 1.035))
            if rng.random() < 0.06:
                price *= rng.uniform(1.06, 1.32)  # legit market move / spot buy
            lt = max(2, int(round(rec.aplfz * rng.uniform(0.85, 1.2))))
            if rng.random() < 0.05:
                lt = max(1, int(round(rec.aplfz * rng.uniform(0.25, 0.6))))  # legit expedite
            eindt = created.normalize() + pd.Timedelta(days=lt)
            vend_delay = vendors.loc[rec.lifnr, "reliability_sd"]
            gr_date = eindt + pd.Timedelta(days=int(max(-2, round(rng.normal(1, vend_delay)))))
            buyer = C.BUYERS[int(rng.integers(len(C.BUYERS)))]
            pr_date = created - pd.Timedelta(days=int(rng.integers(0, 4)))
            eban.append({"banfn": str(banfn), "bnfpo": "00010", "matnr": m, "werks": plant,
                         "menge": float(qty), "requested_by": C.REQUESTERS[int(rng.integers(len(C.REQUESTERS)))],
                         "created_on": pr_date.date().isoformat(), "ebeln": str(ebeln), "ebelp": "00010"})
            pos.append({"ebeln": str(ebeln), "ebelp": "00010", "lifnr": lifnr, "matnr": m, "werks": plant,
                        "menge": float(qty), "netpr": round(float(price), 2), "created_at": created,
                        "eindt": eindt, "gr_date": gr_date, "created_by": buyer, "banfn": str(banfn),
                        "meins": mat.loc[m, "meins"], "matkl": mat.loc[m, "matkl"], "mtart": mat.loc[m, "mtart"]})
            pipeline.append((gr_date, float(qty)))
            ebeln += 1
            banfn += 1
            if rng.random() < 0.03:  # legit: second delivery split the same day by the same buyer
                half = np.ceil(qty / 2 / pack) * pack
                pos[-1]["menge"] = float(half)
                twin = dict(pos[-1])
                twin.update({"ebeln": str(ebeln), "created_at": created + pd.Timedelta(minutes=int(rng.integers(2, 30))),
                             "eindt": eindt + pd.Timedelta(days=14), "gr_date": gr_date + pd.Timedelta(days=14),
                             "banfn": str(banfn)})
                pos.append(twin)
                eban.append({**eban[-1], "banfn": str(banfn), "menge": float(half), "ebeln": str(ebeln)})
                ebeln += 1
                banfn += 1
        stock_end.append({"matnr": m, "werks": plant, "lgort": "0001",
                          "labst": round(max(0.0, stock), 3),
                          "open_po_qty": round(sum(q for _, q in pipeline), 3)})
    return {"po": pd.DataFrame(pos), "eban": pd.DataFrame(eban), "mard": pd.DataFrame(stock_end)}


# --------------------------------------------------------------------------- anomalies
def _dup_name(name: str, rng: np.random.Generator) -> str:
    subs = [("Shree", "Shri"), ("Pvt Ltd", "Pvt. Ltd."), ("Private Limited", "Pvt Ltd"),
            ("Chemicals", "Chemical"), ("Industries", "Inds."), ("& Co", "and Company"),
            ("Packaging", "Packagings"), ("LLP", "L.L.P."), ("Enterprises", "Enterprise")]
    out = name
    for a, b in rng.permutation(subs):
        if a in out:
            out = out.replace(a, b, 1)
            break
    if out == name:
        out = name + " Unit-II"
    return out


def inject_anomalies(cfg: GenConfig, md: dict, po: pd.DataFrame, eban: pd.DataFrame,
                     rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Inject labelled anomalies; returns (po, eban, vendors, labels)."""
    po = po.copy()
    vendors = md["vendors"].copy()
    start, end = pd.Timestamp(cfg.start), pd.Timestamp(cfg.end)
    n_target = int(round(len(po) * cfg.anomaly_rate))
    counts = {k: max(1, int(round(n_target * v))) for k, v in _ANOMALY_MIX.items()}
    labels: list[dict] = []
    taken: set[str] = set()
    eligible = po[po.created_at > start + pd.Timedelta(days=60)].index.to_list()
    rng.shuffle(eligible)
    pool = iter(eligible)

    def take() -> int:
        while True:
            i = next(pool)
            if po.at[i, "ebeln"] not in taken:
                taken.add(po.at[i, "ebeln"])
                return i

    # 1. price spike: 40-200 % above normal
    for _ in range(counts["price_spike"]):
        i = take()
        po.at[i, "netpr"] = round(po.at[i, "netpr"] * rng.uniform(1.35, 2.4), 2)
        labels.append({"ebeln": po.at[i, "ebeln"], "ebelp": "00010", "anomaly_type": "price_spike"})
    # 2. quantity outlier
    for _ in range(counts["quantity_outlier"]):
        i = take()
        po.at[i, "menge"] = float(np.ceil(po.at[i, "menge"] * rng.uniform(2.5, 6.0)))
        labels.append({"ebeln": po.at[i, "ebeln"], "ebelp": "00010", "anomaly_type": "quantity_outlier"})
    # 3. impossible delivery date
    for _ in range(counts["short_lead_time"]):
        i = take()
        normal = (po.at[i, "eindt"] - po.at[i, "created_at"].normalize()).days
        po.at[i, "eindt"] = po.at[i, "created_at"].normalize() + pd.Timedelta(
            days=int(max(1, round(normal * rng.uniform(0.05, 0.3)))))
        po.at[i, "gr_date"] = po.at[i, "eindt"] + pd.Timedelta(days=int(rng.integers(0, 3)))
        labels.append({"ebeln": po.at[i, "ebeln"], "ebelp": "00010", "anomaly_type": "short_lead_time"})

    next_lifnr = int(vendors.lifnr.astype(int).max()) + 1
    # 4. duplicate vendors: re-created master records with the same bank/PAN, POs moved to them
    n_dup_vendors = max(2, counts["duplicate_vendor"] // 4)
    originals = vendors[vendors.created_on < cfg.start].sample(n=n_dup_vendors, random_state=int(rng.integers(1 << 30)))
    dup_map = {}
    for _, v in originals.iterrows():
        created = start + pd.Timedelta(days=int(rng.integers(90, max(91, (end - start).days - 120))))
        pan = v.stcd3[2:12]
        mode = rng.choice(["same_bank", "same_pan", "fresh_ids"], p=[0.35, 0.3, 0.35])
        new = v.to_dict()
        new.update({
            "lifnr": str(next_lifnr), "name1": _dup_name(v.name1, rng),
            "stras": v.stras.replace("Plot", "Plot No.") if rng.random() < 0.6 else v.stras,
            "stcd3": v.stcd3 if mode == "same_bank" else make_gstin(
                rng, v.stcd3[:2], pan=pan if mode == "same_pan" else None, entity="2"),
            "bank_ref": v.bank_ref if mode == "same_bank" else (
                v.bank_ref[:-3] + f"{rng.integers(0, 999):03d}" if mode == "same_pan" else _fake_bank_ref(rng)),
            "created_on": created.date().isoformat(),
        })
        vendors = pd.concat([vendors, pd.DataFrame([new])], ignore_index=True)
        dup_map[v.lifnr] = (str(next_lifnr), created)
        next_lifnr += 1
    n_dup = 0
    for orig, (dup, created) in dup_map.items():
        idx = [i for i in po.index[(po.lifnr == orig) & (po.created_at > created)] if po.at[i, "ebeln"] not in taken]
        k = max(1, counts["duplicate_vendor"] // len(dup_map))
        for i in idx[:k]:
            taken.add(po.at[i, "ebeln"])
            po.at[i, "lifnr"] = dup
            po.at[i, "netpr"] = round(po.at[i, "netpr"] * rng.uniform(1.04, 1.15), 2)
            labels.append({"ebeln": po.at[i, "ebeln"], "ebelp": "00010", "anomaly_type": "duplicate_vendor"})
            n_dup += 1

    # 5. off-hours / weekend creation by a brand-new vendor
    n_new = max(2, counts["offhours_new_vendor"] // 4)
    shell_vendors = []
    shell_names = [f"{a} {b} {c}" for a in ["Quick", "Swift", "Prime", "Global", "Metro", "Zenith"]
                   for b in ["Trading", "Traders", "Supplies", "Impex"] for c in ["Co", "Enterprises", "Agencies"]]
    shell_names = list(rng.permutation(shell_names))
    for _ in range(n_new):
        city, state, code = C.CITIES[int(rng.integers(len(C.CITIES)))]
        created = start + pd.Timedelta(days=int(rng.integers(90, max(91, (end - start).days - 60))))
        typ = "RM" if rng.random() < 0.7 else "PK"
        new = {"lifnr": str(next_lifnr),
               "name1": str(shell_names.pop()),
               "stras": f"Shop {rng.integers(1, 90)}, {rng.choice(['Main Bazaar', 'Market Road', 'Civil Lines'])}",
               "ort01": city, "regio": state, "pstlz": f"{rng.integers(110001, 799999)}",
               "stcd3": make_gstin(rng, code), "bank_ref": _fake_bank_ref(rng),
               "created_on": created.date().isoformat(), "vendor_type": typ, "groups": [],
               "price_level": 1.1, "lt_mean": 10.0, "reliability_sd": 3.0}
        vendors = pd.concat([vendors, pd.DataFrame([new])], ignore_index=True)
        shell_vendors.append((str(next_lifnr), created, typ, rng.random() < 0.5))
        next_lifnr += 1
    for j in range(counts["offhours_new_vendor"]):
        lifnr, created, typ, with_info = shell_vendors[j % len(shell_vendors)]
        cands = [i for i in po.index[(po.created_at > created + pd.Timedelta(days=2)) &
                                     (po.created_at < created + pd.Timedelta(days=75)) &
                                     (po.mtart == ("ROH" if typ == "RM" else "VERP"))]
                 if po.at[i, "ebeln"] not in taken]
        if not cands:
            continue
        i = cands[int(rng.integers(len(cands)))]
        taken.add(po.at[i, "ebeln"])
        ts = po.at[i, "created_at"]
        if rng.random() < 0.55:
            ts = ts.normalize() + pd.Timedelta(days=(5 - ts.weekday()) % 7 + int(rng.integers(0, 2)),
                                               hours=int(rng.integers(10, 17)))
        else:
            ts = ts.normalize() + pd.Timedelta(hours=int(rng.choice([21, 22, 23, 0, 1, 2, 5])))
        po.at[i, "created_at"] = ts
        po.at[i, "lifnr"] = lifnr
        po.at[i, "netpr"] = round(po.at[i, "netpr"] * rng.uniform(0.98, 1.12), 2)
        po.at[i, "eindt"] = ts.normalize() + pd.Timedelta(days=int(rng.integers(5, 15)))
        po.at[i, "gr_date"] = po.at[i, "eindt"]
        if with_info:  # the fraudster also created a purchasing info record
            infnr = str(int(md["eina"].infnr.astype(int).max()) + 1)
            md["eina"] = pd.concat([md["eina"], pd.DataFrame([{
                "infnr": infnr, "matnr": po.at[i, "matnr"], "lifnr": lifnr, "created_on": created.date().isoformat(),
                "is_primary": 0}])], ignore_index=True)
            md["eine"] = pd.concat([md["eine"], pd.DataFrame([{
                "infnr": infnr, "ekorg": C.PURCH_ORG, "netpr": po.at[i, "netpr"], "peinh": 1, "waers": "INR",
                "aplfz": int(rng.integers(5, 15)), "valid_from": created.date().isoformat()}])], ignore_index=True)
        labels.append({"ebeln": po.at[i, "ebeln"], "ebelp": "00010", "anomaly_type": "offhours_new_vendor"})

    # 6. split POs: one requirement split into several POs just under the approval limit
    thr = cfg.approval_threshold
    next_ebeln = int(po.ebeln.astype(int).max()) + 1
    next_banfn = int(eban.banfn.astype(int).max()) + 1 if len(eban) else 10000001
    new_rows, new_pr = [], []
    n_groups = max(1, counts["split_po"] // 3)
    for _ in range(n_groups):
        i = take()
        base = po.loc[i].to_dict()
        n_parts = int(rng.choice([2, 3, 3, 4]))
        buyer = C.BUYERS[int(rng.integers(len(C.BUYERS)))]
        day = base["created_at"].normalize() + pd.Timedelta(hours=int(rng.integers(10, 17)))
        spread = rng.random() < 0.4  # smarter splitting: parts spread over consecutive days
        for k in range(n_parts):
            target_val = thr * rng.uniform(0.8, 0.99)
            row = dict(base)
            row.update({"ebeln": str(next_ebeln), "menge": float(max(1.0, np.floor(target_val / base["netpr"]))),
                        "created_at": day + pd.Timedelta(days=k if spread else 0,
                                                         minutes=int(7 * k + rng.integers(1, 6))),
                        "created_by": buyer, "banfn": str(next_banfn)})
            new_rows.append(row)
            new_pr.append({"banfn": str(next_banfn), "bnfpo": "00010", "matnr": base["matnr"],
                           "werks": base["werks"], "menge": row["menge"],
                           "requested_by": C.REQUESTERS[int(rng.integers(len(C.REQUESTERS)))],
                           "created_on": day.date().isoformat(), "ebeln": str(next_ebeln), "ebelp": "00010"})
            labels.append({"ebeln": str(next_ebeln), "ebelp": "00010", "anomaly_type": "split_po"})
            next_ebeln += 1
            next_banfn += 1
    po = pd.concat([po, pd.DataFrame(new_rows)], ignore_index=True)
    eban = pd.concat([eban, pd.DataFrame(new_pr)], ignore_index=True)
    lab = pd.DataFrame(labels).drop_duplicates(["ebeln", "ebelp"])
    return po, eban, vendors, lab


# --------------------------------------------------------------------------- assembly
def _assemble_documents(cfg: GenConfig, po: pd.DataFrame, issues: pd.DataFrame,
                        rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    end = pd.Timestamp(cfg.end)
    po = po.sort_values("created_at").reset_index(drop=True)
    po["netwr"] = (po.menge * po.netpr).round(2)
    ekko = pd.DataFrame({
        "ebeln": po.ebeln, "bukrs": C.COMPANY_CODE, "bsart": "NB", "lifnr": po.lifnr,
        "bedat": po.created_at.dt.date.astype(str),
        "created_at": po.created_at.dt.strftime("%Y-%m-%d %H:%M:%S"),
        "ekorg": C.PURCH_ORG, "ekgrp": np.where(po.mtart == "ROH", "P01", "P02"),
        "created_by": po.created_by, "waers": "INR",
        "approval_status": np.where(po.netwr >= cfg.approval_threshold, "APPROVED_MGR", "AUTO_RELEASED"),
        "status": np.where(po.gr_date <= end, "CLOSED", "OPEN"),
    })
    ekpo = pd.DataFrame({
        "ebeln": po.ebeln, "ebelp": po.ebelp, "matnr": po.matnr, "werks": po.werks,
        "menge": po.menge, "meins": po.meins, "netpr": po.netpr, "peinh": 1, "netwr": po.netwr,
        "eindt": po.eindt.dt.date.astype(str), "banfn": po.banfn, "matkl": po.matkl,
    })
    # goods movements
    grs = po[po.gr_date <= end]
    mkpf, mseg = [], []
    mblnr = 4900000001 + cfg.id_offset
    for r in grs.itertuples():
        mkpf.append({"mblnr": str(mblnr), "mjahr": r.gr_date.year, "budat": r.gr_date.date().isoformat(), "usnam": "STORE01"})
        mseg.append({"mblnr": str(mblnr), "zeile": "0001", "bwart": "101", "matnr": r.matnr, "werks": r.werks,
                     "menge": r.menge, "meins": r.meins, "budat": r.gr_date.date().isoformat(),
                     "ebeln": r.ebeln, "aufnr": None, "lifnr": r.lifnr})
        mblnr += 1
    for aufnr, grp in issues.groupby("aufnr", sort=False):
        b = grp.budat.iloc[0]
        mkpf.append({"mblnr": str(mblnr), "mjahr": int(b[:4]), "budat": b, "usnam": "PROD_SUP1"})
        for k, r in enumerate(grp.itertuples()):
            mseg.append({"mblnr": str(mblnr), "zeile": f"{k + 1:04d}", "bwart": "261", "matnr": r.matnr,
                         "werks": r.werks, "menge": r.menge, "meins": r.meins, "budat": b,
                         "ebeln": None, "aufnr": aufnr, "lifnr": None})
        mblnr += 1
    return {"ekko": ekko, "ekpo": ekpo, "mkpf": pd.DataFrame(mkpf), "mseg": pd.DataFrame(mseg)}


def _build_marc(cfg: GenConfig, md: dict, issues: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Plant-level MRP data. MINBE is set from the first 6 months only, so it is
    deliberately stale for growing / seasonal materials (that's what the forecaster fixes)."""
    info = md["eina"].merge(md["eine"], on="infnr")
    prim = info[info.is_primary == 1].set_index("matnr")
    first = issues[pd.to_datetime(issues.budat) < pd.Timestamp(cfg.start) + pd.Timedelta(days=182)]
    wk = first.groupby(["matnr", "werks"]).menge.sum() / 26.0
    rows = []
    for m, plants in md["mat_plants"].items():
        for p in sorted(plants):
            avg = float(wk.get((m, p), 0.0))
            lt = int(prim.loc[m, "aplfz"]) if m in prim.index else 14
            minbe = avg * (lt / 7 + 1.0) * rng.uniform(0.75, 1.15)
            rows.append({"matnr": m, "werks": p, "plifz": lt, "eisbe": round(avg * rng.uniform(0.5, 1.0), 3),
                         "minbe": round(minbe, 3), "dispo": C.MRP_CONTROLLERS[int(rng.integers(4))],
                         "dismm": "VB"})
    for m, plants in md["fg_plants"].items():
        for p in plants:
            rows.append({"matnr": m, "werks": p, "plifz": 0, "eisbe": 0.0, "minbe": 0.0,
                         "dispo": "010", "dismm": "PD"})
    return pd.DataFrame(rows)


def generate(cfg: GenConfig | None = None) -> dict[str, pd.DataFrame]:
    """Run the full generator and return all tables as DataFrames (lower-case SAP names)."""
    cfg = cfg or GenConfig()
    md = build_master_data(cfg)
    rng = np.random.default_rng(cfg.seed + 1 + cfg.id_offset)
    afko, afpo, plaf = simulate_production(cfg, md, rng)
    issues = derive_consumption(afko, afpo, md, rng, cfg)
    proc = simulate_procurement(cfg, md, issues, rng)
    po, eban = proc["po"], proc["eban"]

    vendors = md["vendors"]
    if (po.lifnr == "DRIFT").any():  # drift scenario: new vendor mix
        n_new = 6
        start_id = int(vendors.lifnr.astype(int).max()) + 1
        newv = []
        for k in range(n_new):
            city, state, code = C.CITIES[int(rng.integers(len(C.CITIES)))]
            newv.append({"lifnr": str(start_id + k), "name1": f"New Age Supplier {k + 1} Pvt Ltd",
                         "stras": f"Plot {k + 10}, Sector 5", "ort01": city, "regio": state,
                         "pstlz": "110001", "stcd3": make_gstin(rng, code), "bank_ref": _fake_bank_ref(rng),
                         "created_on": (pd.Timestamp(cfg.start) - pd.Timedelta(days=10)).date().isoformat(),
                         "vendor_type": "RM", "groups": [], "price_level": 1.0, "lt_mean": 9.0,
                         "reliability_sd": 2.0})
        md["vendors"] = vendors = pd.concat([vendors, pd.DataFrame(newv)], ignore_index=True)
        mask = po.lifnr == "DRIFT"
        po.loc[mask, "lifnr"] = rng.choice([v["lifnr"] for v in newv], mask.sum())

    if cfg.anomaly_rate > 0:
        po, eban, vendors, labels = inject_anomalies(cfg, md, po, eban, rng)
    else:
        labels = pd.DataFrame(columns=["ebeln", "ebelp", "anomaly_type"])
    docs = _assemble_documents(cfg, po, issues, rng)
    marc = _build_marc(cfg, md, issues, rng)

    lfa1 = vendors.drop(columns=["groups", "price_level", "lt_mean", "reliability_sd", "vendor_type"]).copy()
    lfa1["land1"] = "IN"
    afko_out = afko.copy()
    tables = {
        "lfa1": lfa1, "mara": md["mara"], "makt": md["makt"], "marc": marc, "mard": proc["mard"],
        "eina": md["eina"], "eine": md["eine"], "eban": eban, "ekko": docs["ekko"], "ekpo": docs["ekpo"],
        "mkpf": docs["mkpf"], "mseg": docs["mseg"], "afko": afko_out, "afpo": afpo,
        "stko": md["stko"], "stpo": md["stpo"], "plaf": plaf, "anomaly_labels": labels,
    }
    for k, df in tables.items():
        tables[k] = df.reset_index(drop=True)
    return tables


def write_csv(tables: dict[str, pd.DataFrame], out_dir) -> None:
    """Write every table to ``out_dir/<table>.csv``."""
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False)


__all__ = ["GenConfig", "generate", "write_csv", "is_valid_gstin", "make_gstin", "season_factor",
           "ANOMALY_TYPES", "timedelta"]
