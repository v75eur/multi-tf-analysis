import os, json, hashlib, secrets, time, bisect
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, redirect, url_for, session, render_template
import requests
import websocket
import numpy as np

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", secrets.token_hex(32))
ADMIN_HASH = os.getenv("ADMIN_HASH", "d0695d2f4b6487fb81c7047ba01d06d5065aa1a9f18f89633389e1e9b5d85fd6")

BENIN = timezone(timedelta(hours=1))
JOURS = ["lun", "mar", "mer", "jeu", "ven", "sam", "dim"]

YAHOO_PAIRS = {"EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "XAUUSD": "GC=F"}
DERIV_PAIRS = {"V75": "R_75"}
DECIMALES = {"EURUSD": 5, "GBPUSD": 5, "XAUUSD": 2, "V75": 4}
# Taille d'un lot standard (compte en dollars). V75 : non calculé (dépend de ta plateforme).
CONTRAT = {"EURUSD": 100000, "GBPUSD": 100000, "XAUUSD": 100, "V75": None}
# Spread supposé (en prix), déduit de chaque trade du test. MODIFIE selon ton broker.
SPREAD = {"EURUSD": 0.00010, "GBPUSD": 0.00015, "XAUUSD": 0.30, "V75": 0.0}
# tf: (interval Yahoo, période Yahoo, durée d'une bougie en secondes)
TIMEFRAMES = {
    "H4": ("1h", "60d", 14400),
    "H1": ("1h", "60d", 3600),
    "M30": ("30m", "30d", 1800),
    "M15": ("15m", "15d", 900),
    "M5": ("5m", "30d", 300),
}
DERIV_COUNT = {"H4": 300, "H1": 400, "M30": 600, "M15": 1200, "M5": 3000}
PERIODE = 10
EMA_COURT = 5
EMA_LONG = 25
DERIV_WS = "wss://api.derivws.com/trading/v1/options/ws/public"
CACHE_SEC = {"1h": 300, "30m": 180, "15m": 120, "5m": 60}
_cache_yahoo = {}
_cache_bt = {}

MODES = {
    "strict": "Strict (7 conditions)",
    "strict_f": "Strict + filtres",
    "souple": "Souple (H1 + M15 + EMA M5)",
    "souple_f": "Souple + filtres",
}
MIN_TRADES = 12


def logged():
    return session.get("admin") is True


def fp(pair, x):
    return f"{x:.{DECIMALES.get(pair, 5)}f}"


# ───────────── RÉCUPÉRATION DES BOUGIES ─────────────

def recuperer_bougies_deriv(symbol, granularity, count=300):
    try:
        ws = websocket.create_connection(DERIV_WS, timeout=20)
        ws.send(json.dumps({"ticks_history": symbol, "count": count, "end": "latest",
                            "style": "candles", "granularity": granularity}))
        r = json.loads(ws.recv())
        ws.close()
        if "candles" in r:
            return [{"t": c["epoch"], "open": float(c["open"]), "high": float(c["high"]),
                     "low": float(c["low"]), "close": float(c["close"])} for c in r["candles"]]
        return []
    except Exception as e:
        print(f"Deriv {symbol}: {e}")
        return []


def recuperer_bougies_yahoo(symbol, interval, range_):
    now = time.time()
    key = f"{symbol}_{interval}_{range_}"
    ttl = CACHE_SEC.get(interval, 120)
    if key in _cache_yahoo:
        ts, data = _cache_yahoo[key]
        if now - ts < ttl:
            return data
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval={interval}&range={range_}"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        data = r.json()["chart"]["result"][0]
        ts_list = data["timestamp"]
        q = data["indicators"]["quote"][0]
        bougies = []
        for i in range(len(ts_list)):
            if q["open"][i] and q["high"][i] and q["low"][i] and q["close"][i]:
                bougies.append({"t": ts_list[i], "open": q["open"][i], "high": q["high"][i],
                                "low": q["low"][i], "close": q["close"][i]})
        _cache_yahoo[key] = (now, bougies)
        return bougies
    except Exception as e:
        print(f"Yahoo {symbol}: {e}")
        if key in _cache_yahoo:
            return _cache_yahoo[key][1]
        return []


def regrouper(bougies, gran):
    """Fabrique des grosses bougies (ex: H4) à partir de petites (H1)."""
    groupes = {}
    for b in bougies:
        k = b["t"] // gran * gran
        if k not in groupes:
            groupes[k] = {"t": k, "open": b["open"], "high": b["high"], "low": b["low"], "close": b["close"]}
        else:
            g = groupes[k]
            g["high"] = max(g["high"], b["high"])
            g["low"] = min(g["low"], b["low"])
            g["close"] = b["close"]
    return [groupes[k] for k in sorted(groupes)]


def bougies_fermees(bougies, gran):
    """Retire la dernière bougie si elle n'est pas encore clôturée."""
    if bougies and bougies[-1]["t"] + gran > time.time():
        return bougies[:-1]
    return bougies


# ───────────── CALCULS (bougies uniquement) ─────────────

def calc_ema_serie(clotures, periode):
    if len(clotures) < periode:
        return []
    k = 2 / (periode + 1)
    ema = float(np.mean(clotures[:periode]))
    serie = [ema]
    for c in clotures[periode:]:
        ema = c * k + ema * (1 - k)
        serie.append(ema)
    return serie


def calc_ema_signal(clotures):
    s5 = calc_ema_serie(clotures, EMA_COURT)
    s25 = calc_ema_serie(clotures, EMA_LONG)
    if not s25:
        return None, None, None, None
    s5 = s5[-len(s25):]
    ecarts = [a - b for a, b in zip(s5, s25)]
    dernier = ecarts[-1]
    if dernier > 0:
        sig = "haussier"
    elif dernier < 0:
        sig = "baissier"
    else:
        sig = "neutre"
    depuis = 0
    for e in reversed(ecarts):
        if e != 0 and (e > 0) == (dernier > 0):
            depuis += 1
        else:
            break
    return round(float(s5[-1]), 5), round(float(s25[-1]), 5), sig, depuis


def calc_tendance(clotures):
    x = np.arange(1, len(clotures) + 1)
    pente = np.polyfit(x, np.array(clotures), 1)[0]
    pct = float(min(100, abs(pente) / np.mean(clotures) * 10000))
    if pente > 0:
        return "haussier", round(pct, 1)
    if pente < 0:
        return "baissier", round(pct, 1)
    return "neutre", 0.0


def calc_acheteurs(bougies, periode=10):
    d = bougies[-periode:]
    v = sum(1 for b in d if b["close"] > b["open"])
    r = sum(1 for b in d if b["close"] < b["open"])
    t = len(d) or 1
    return v, r, round(v / t * 100, 1), round(r / t * 100, 1)


def calc_agitation(bougies, periode=10):
    """Part du mouvement perdue en allers-retours (haut = marché indécis)."""
    d = bougies[-periode:]
    if len(d) < 2:
        return 0, 0.0
    total = sum(b["high"] - b["low"] for b in d)
    if total == 0:
        return 0, 0.0
    net = abs(d[-1]["close"] - d[0]["open"])
    efficacite = min(100.0, net / total * 100)
    return 0, round(100 - efficacite, 1)


def trouver_creux(bougies, fenetre=50, largeur=3):
    n = len(bougies)
    creux = []
    for i in range(max(largeur, n - fenetre), n - largeur):
        if all(bougies[j]["low"] > bougies[i]["low"] for j in range(i - largeur, i + largeur + 1) if j != i):
            creux.append(i)
    return creux


def trouver_sommets(bougies, fenetre=50, largeur=3):
    n = len(bougies)
    sommets = []
    for i in range(max(largeur, n - fenetre), n - largeur):
        if all(bougies[j]["high"] < bougies[i]["high"] for j in range(i - largeur, i + largeur + 1) if j != i):
            sommets.append(i)
    return sommets


def calc_divergence(bougies, creux, sommets):
    div_h = False
    if len(creux) >= 2:
        c1, c2 = creux[-2], creux[-1]
        if bougies[c2]["low"] < bougies[c1]["low"]:
            r1 = calc_acheteurs(bougies[:c1 + 1])[2]
            r2 = calc_acheteurs(bougies[:c2 + 1])[2]
            if r2 > r1:
                div_h = True
    div_b = False
    if len(sommets) >= 2:
        s1, s2 = sommets[-2], sommets[-1]
        if bougies[s2]["high"] > bougies[s1]["high"]:
            r1 = calc_acheteurs(bougies[:s1 + 1])[2]
            r2 = calc_acheteurs(bougies[:s2 + 1])[2]
            if r2 < r1:
                div_b = True
    if div_h:
        return "haussiere"
    if div_b:
        return "baissiere"
    return "aucune"


def mouvement_moyen(bougies, n=24):
    d = bougies[-n:]
    if not d:
        return 0.0
    return float(np.mean([b["high"] - b["low"] for b in d]))


# ───────────── MOTIFS DE BOUGIES / CASSURE ─────────────

def calc_motifs(bougies):
    """Mèches de rejet, englobante, bougie anormale (dernière bougie fermée)."""
    b, p = bougies[-1], bougies[-2]
    rng = b["high"] - b["low"]
    motifs, mh, mb = [], False, False
    if rng > 0:
        corps = abs(b["close"] - b["open"])
        bas = min(b["open"], b["close"]) - b["low"]
        haut = b["high"] - max(b["open"], b["close"])
        if bas / rng >= 0.6 and corps / rng <= 0.3:
            motifs.append({"txt": "Mèche basse", "cls": "vert"})
            mh = True
        if haut / rng >= 0.6 and corps / rng <= 0.3:
            motifs.append({"txt": "Mèche haute", "cls": "rouge"})
            mb = True
    if p["close"] < p["open"] and b["close"] > b["open"] and b["open"] <= p["close"] and b["close"] >= p["open"]:
        motifs.append({"txt": "Englobante ↑", "cls": "vert"})
        mh = True
    if p["close"] > p["open"] and b["close"] < b["open"] and b["open"] >= p["close"] and b["close"] <= p["open"]:
        motifs.append({"txt": "Englobante ↓", "cls": "rouge"})
        mb = True
    recent = bougies[-21:-1]
    moy = float(np.mean([x["high"] - x["low"] for x in recent])) if recent else 0.0
    taille_x = round(rng / moy, 1) if moy > 0 else 0.0
    grande = taille_x >= 2
    if grande:
        motifs.append({"txt": f"Grande ×{taille_x}", "cls": "jaune"})
    return motifs, grande, taille_x, mh, mb


def calc_cassure(bougies, n=20):
    """Cassure du plus haut / plus bas des 20 bougies d'avant (sur les 3 dernières bougies)."""
    for k in range(0, 3):
        idx = len(bougies) - 1 - k
        if idx < n:
            break
        prev = bougies[idx - n:idx]
        c = bougies[idx]["close"]
        if c > max(x["high"] for x in prev):
            return {"sens": "haussiere", "il_y_a": k}
        if c < min(x["low"] for x in prev):
            return {"sens": "baissiere", "il_y_a": k}
    return {"sens": "aucune", "il_y_a": None}


def analyser(tf, bougies):
    if len(bougies) < 30:
        return None
    clotures = [b["close"] for b in bougies]
    tendance, pct_tendance = calc_tendance(clotures[-PERIODE:])
    v, r, pa, pv = calc_acheteurs(bougies)
    c, ag = calc_agitation(bougies)
    creux = trouver_creux(bougies)
    sommets = trouver_sommets(bougies)
    divergence = calc_divergence(bougies, creux, sommets)
    ema5, ema25, ema_sig, ema_depuis = calc_ema_signal(clotures)
    motifs, grande, taille_x, mh, mb = calc_motifs(bougies)
    return {
        "tf": tf, "tendance": tendance, "pct_tendance": pct_tendance,
        "pct_acheteurs": pa, "pct_vendeurs": pv,
        "pct_agitation": ag,
        "divergence": divergence,
        "ema5": ema5, "ema25": ema25, "ema_signal": ema_sig, "ema_depuis": ema_depuis,
        "motifs": motifs, "grande": grande, "taille_x": taille_x,
        "motif_haussier": mh, "motif_baissier": mb,
        "cassure": calc_cassure(bougies),
        "prix": clotures[-1]
    }


# ───────────── STATISTIQUES GÉNÉRALES PAR DEVISE ─────────────

def calc_stats(pair, h1, tfs):
    if len(h1) < 30:
        return None
    cl = h1[-1]["close"]

    def var(n):
        ref = h1[-min(n + 1, len(h1))]["close"]
        return round((cl - ref) / ref * 100, 2)

    j, s = h1[-24:], h1[-120:]
    haut24, bas24 = max(b["high"] for b in j), min(b["low"] for b in j)
    haut5, bas5 = max(b["high"] for b in s), min(b["low"] for b in s)
    pos = round((cl - bas24) / (haut24 - bas24) * 100) if haut24 > bas24 else 50

    def mouvement(liste):
        return float(np.mean([(b["high"] - b["low"]) / b["close"] * 100 for b in liste]))

    vol24, vol5 = mouvement(j), mouvement(s)
    if vol24 > vol5 * 1.3:
        vol_etat = "forte"
    elif vol24 < vol5 * 0.7:
        vol_etat = "faible"
    else:
        vol_etat = "normale"

    d100 = h1[-100:]
    vertes = round(sum(1 for b in d100 if b["close"] > b["open"]) / len(d100) * 100, 1)

    sens, n_serie = 0, 0
    for b in reversed(h1):
        c = 1 if b["close"] > b["open"] else -1 if b["close"] < b["open"] else 0
        if n_serie == 0:
            if c == 0:
                break
            sens, n_serie = c, 1
        elif c == sens:
            n_serie += 1
        else:
            break

    h = sum(1 for t in tfs.values() if t["tendance"] == "haussier")
    b_ = sum(1 for t in tfs.values() if t["tendance"] == "baissier")
    tot = len(tfs)
    if h > b_:
        align_txt, align_cls = f"{h}/{tot} haussiers", "vert"
    elif b_ > h:
        align_txt, align_cls = f"{b_}/{tot} baissiers", "rouge"
    else:
        align_txt, align_cls = f"Partagé {h}-{b_}", ""

    return {
        "var24": var(24), "var5j": var(120),
        "bas24": fp(pair, bas24), "haut24": fp(pair, haut24),
        "bas5j": fp(pair, bas5), "haut5j": fp(pair, haut5),
        "position": pos,
        "vol24": vol24, "vol_etat": vol_etat,
        "vertes": vertes,
        "serie_n": n_serie, "serie_sens": "hausse" if sens == 1 else "baisse" if sens == -1 else "—",
        "align_txt": align_txt, "align_cls": align_cls,
    }


# ───────────── SUPPORTS / RÉSISTANCES ─────────────

def calc_niveaux(pair, prix, m15, h1):
    res, sup = [], []
    for bl in (m15, h1):
        for i in trouver_sommets(bl, 80, 3):
            res.append(bl[i]["high"])
        for i in trouver_creux(bl, 80, 3):
            sup.append(bl[i]["low"])
    r = min([x for x in res if x > prix], default=None)
    s = max([x for x in sup if x < prix], default=None)
    return {
        "res": fp(pair, r) if r is not None else None,
        "res_brut": r,
        "res_pct": round((r - prix) / prix * 100, 3) if r is not None else None,
        "sup": fp(pair, s) if s is not None else None,
        "sup_brut": s,
        "sup_pct": round((prix - s) / prix * 100, 3) if s is not None else None,
    }


# ───────────── STOP LOSS / OBJECTIF ─────────────

def cote_sltp(pair, prix, stop, obj, sens):
    if stop is None or obj is None:
        return None
    if sens == "achat" and not (stop < prix < obj):
        return None
    if sens == "vente" and not (obj < prix < stop):
        return None
    risque, gain = abs(prix - stop), abs(obj - prix)
    if risque <= 0:
        return None
    ratio = gain / risque
    return {
        "stop": fp(pair, stop), "objectif": fp(pair, obj),
        "stop_brut": stop, "obj_brut": obj,
        "risque_pct": round(risque / prix * 100, 3),
        "gain_pct": round(gain / prix * 100, 3),
        "ratio": round(ratio, 2), "ok": ratio >= 1.5,
    }


def calc_sltp(pair, prix, m15):
    tampon = mouvement_moyen(m15, 20) * 0.1
    creux = [m15[i]["low"] for i in trouver_creux(m15, 80, 3)]
    sommets = [m15[i]["high"] for i in trouver_sommets(m15, 80, 3)]
    bas20 = min(b["low"] for b in m15[-20:])
    haut20 = max(b["high"] for b in m15[-20:])
    bas50 = min(b["low"] for b in m15[-50:])
    haut50 = max(b["high"] for b in m15[-50:])

    st = [x for x in creux if x < prix]
    stop_a = (st[-1] if st else bas20) - tampon
    tg = [x for x in sommets if x > prix]
    obj_a = tg[-1] if tg else (haut50 if haut50 > prix else None)

    sh = [x for x in sommets if x > prix]
    stop_v = (sh[-1] if sh else haut20) + tampon
    tc = [x for x in creux if x < prix]
    obj_v = tc[-1] if tc else (bas50 if bas50 < prix else None)

    return {"achat": cote_sltp(pair, prix, stop_a, obj_a, "achat"),
            "vente": cote_sltp(pair, prix, stop_v, obj_v, "vente")}


# ───────────── MEILLEURES HEURES (heure du Bénin) ─────────────

def calc_heures(h1):
    par = {}
    for b in h1:
        hh = datetime.fromtimestamp(b["t"], BENIN).hour
        par.setdefault(hh, []).append((b["high"] - b["low"]) / b["close"] * 100)
    moy = {h: float(np.mean(v)) for h, v in par.items() if len(v) >= 3}
    if len(moy) < 8:
        return None
    glob = float(np.mean(list(moy.values())))
    top = sorted(moy, key=moy.get, reverse=True)[:3]
    plate = max(moy.values()) / min(moy.values()) < 1.25
    maintenant = datetime.now(BENIN).hour
    if maintenant in moy and glob > 0:
        rel = moy[maintenant] / glob
        etat = "forte" if rel >= 1.2 else "faible" if rel <= 0.8 else "normale"
    else:
        rel, etat = 0.0, "inconnue"
    return {
        "top": [f"{h:02d}h–{(h + 1) % 24:02d}h" for h in top],
        "plate": plate, "now_etat": etat, "now_ratio": round(rel, 2),
    }


# ───────────── MODES DE SIGNAL (live) ─────────────

def etat_live(t):
    if t["tendance"] == "haussier" and t["pct_acheteurs"] > t["pct_vendeurs"]:
        return 1
    if t["tendance"] == "baissier" and t["pct_vendeurs"] > t["pct_acheteurs"]:
        return -1
    return 0


def signal_live(mode, tfs, sltp):
    """Renvoie 'ACHAT', 'VENTE' ou None pour un mode donné, avec les données actuelles."""
    if any(k not in tfs for k in ("H1", "M30", "M15", "M5")):
        return None
    h1s, m30s, m15s = etat_live(tfs["H1"]), etat_live(tfs["M30"]), etat_live(tfs["M15"])
    if mode.startswith("strict"):
        base = h1s if (h1s != 0 and h1s == m30s == m15s) else 0
    else:
        base = h1s if (h1s != 0 and h1s == m15s) else 0
    m5 = tfs["M5"]
    ema = 1 if m5["ema_signal"] == "haussier" else -1 if m5["ema_signal"] == "baissier" else 0
    if base == 0 or ema != base:
        return None
    if mode.endswith("_f"):
        if (m5["ema_depuis"] or 0) > 12 or m5["grande"]:
            return None
        cote = sltp["achat" if base == 1 else "vente"] if sltp else None
        if not cote or cote["ratio"] < 1.5:
            return None
    return "ACHAT" if base == 1 else "VENTE"


# ───────────── TEST SUR LE PASSÉ ─────────────

def etat_bt(sl):
    d = sl[-PERIODE:]
    cl = [b["close"] for b in d]
    p = np.polyfit(np.arange(len(cl)), np.array(cl), 1)[0]
    v = sum(1 for b in d if b["close"] > b["open"])
    r = sum(1 for b in d if b["close"] < b["open"])
    if p > 0 and v > r:
        return 1
    if p < 0 and r > v:
        return -1
    return 0


def est_grande(m5, i):
    rng = m5[i]["high"] - m5[i]["low"]
    recent = m5[max(0, i - 20):i]
    if not recent:
        return False
    moy = float(np.mean([x["high"] - x["low"] for x in recent]))
    return moy > 0 and rng / moy >= 2


def simuler(m5, i, sens, cote, spread, max_trade):
    """Simule un trade avec stop et objectif. Résultat en R, spread déduit."""
    entree = m5[i]["close"]
    st, ob = cote["stop_brut"], cote["obj_brut"]
    risque = abs(entree - st)
    if risque <= 0:
        return None
    cout = spread / risque
    fin = min(len(m5) - 1, i + max_trade)
    for j in range(i + 1, fin + 1):
        bj = m5[j]
        if sens == "ACHAT":
            if bj["low"] <= st:
                return -1.0 - cout
            if bj["high"] >= ob:
                return cote["ratio"] - cout
        else:
            if bj["high"] >= st:
                return -1.0 - cout
            if bj["low"] <= ob:
                return cote["ratio"] - cout
    if fin == i + max_trade:
        dernier = m5[fin]["close"]
        brut = ((dernier - entree) if sens == "ACHAT" else (entree - dernier)) / risque
        return brut - cout
    return None


def cotes_au_temps(pair, series_m15, ends15, t, entree):
    k = bisect.bisect_right(ends15, t)
    sl = series_m15[max(0, k - 100):k]
    if len(sl) < 30:
        return {"achat": None, "vente": None}
    return calc_sltp(pair, entree, sl)


def backtest(series, pair, test=2500, min_avant=12, max_trade=48):
    """Rejoue 4 modes de signal sur les bougies passées, avec stop, objectif et spread."""
    if any(len(series.get(tf, [])) < 30 for tf in ("H1", "M30", "M15", "M5")):
        return None
    m5 = series["M5"]
    ends = {tf: [b["t"] + g for b in series[tf]] for tf, g in (("H1", 3600), ("M30", 1800), ("M15", 900))}
    debut = max(40, len(m5) - test)
    fin_boucle = len(m5) - min_avant
    spread = SPREAD.get(pair, 0.0)
    prev = {m: None for m in MODES}
    trades = {m: [] for m in MODES}

    for i in range(debut, fin_boucle):
        if m5[i + min_avant]["t"] - m5[i]["t"] > min_avant * 300 * 1.5:
            prev = {m: None for m in MODES}
            continue
        t = m5[i]["t"] + 300
        etats, ok = {}, True
        for tf in ("H1", "M30", "M15"):
            k = bisect.bisect_right(ends[tf], t)
            sl = series[tf][max(0, k - 60):k]
            if len(sl) < 30:
                ok = False
                break
            etats[tf] = etat_bt(sl)
        if not ok:
            prev = {m: None for m in MODES}
            continue
        h1s, m30s, m15s = etats["H1"], etats["M30"], etats["M15"]
        s_souple = h1s if (h1s != 0 and h1s == m15s) else 0
        s_strict = h1s if (h1s != 0 and h1s == m30s == m15s) else 0

        ema_sens, depuis, grande = 0, 0, False
        if s_souple != 0:
            cl = [b["close"] for b in m5[max(0, i - 100):i + 1]]
            _, _, sg, dep = calc_ema_signal(cl)
            ema_sens = 1 if sg == "haussier" else -1 if sg == "baissier" else 0
            depuis = dep or 0
            grande = est_grande(m5, i)

        entree = m5[i]["close"]
        cotes = None
        for mode in MODES:
            base = s_strict if mode.startswith("strict") else s_souple
            sig, cote = None, None
            if base != 0 and ema_sens == base:
                ok_filtre = True
                if mode.endswith("_f"):
                    if depuis > 12 or grande:
                        ok_filtre = False
                    else:
                        if cotes is None:
                            cotes = cotes_au_temps(pair, series["M15"], ends["M15"], t, entree)
                        cote = cotes["achat" if base == 1 else "vente"]
                        if not cote or cote["ratio"] < 1.5:
                            ok_filtre = False
                if ok_filtre:
                    sig = "ACHAT" if base == 1 else "VENTE"
            if sig and sig != prev[mode]:
                if cote is None:
                    if cotes is None:
                        cotes = cotes_au_temps(pair, series["M15"], ends["M15"], t, entree)
                    cote = cotes["achat" if sig == "ACHAT" else "vente"]
                if cote:
                    r = simuler(m5, i, sig, cote, spread, max_trade)
                    if r is not None:
                        trades[mode].append((i, r))
            prev[mode] = sig

    coupe = debut + int((fin_boucle - debut) * 0.6)
    modes, meilleur, meilleur_r = [], None, None
    for mode, label in MODES.items():
        rs = [r for _, r in trades[mode]]
        tr = [r for idx, r in trades[mode] if idx < coupe]
        te = [r for idx, r in trades[mode] if idx >= coupe]
        n = len(rs)
        avg = round(float(np.mean(rs)), 2) if n else None
        wr = round(sum(1 for r in rs if r > 0) / n * 100, 1) if n else None
        train = round(float(np.mean(tr)), 2) if tr else None
        test_r = round(float(np.mean(te)), 2) if te else None
        valide = bool(n >= MIN_TRADES and avg is not None and avg > 0.1
                      and len(tr) >= 4 and len(te) >= 4
                      and train is not None and test_r is not None and train > 0 and test_r > 0)
        if valide:
            couleur = "vert"
        elif n >= 8 and avg is not None and avg < -0.1:
            couleur = "rouge"
        else:
            couleur = ""
        modes.append({"key": mode, "label": label, "n": n, "avg": avg, "wr": wr,
                      "train": train, "n_train": len(tr), "test": test_r, "n_test": len(te),
                      "valide": valide, "couleur": couleur})
        if valide and (meilleur_r is None or avg > meilleur_r):
            meilleur, meilleur_r = mode, avg

    heures = round((m5[-1]["t"] - m5[debut]["t"]) / 3600)
    return {"modes": modes, "meilleur": meilleur, "heures": heures,
            "spread": fp(pair, spread), "min_trades": MIN_TRADES}


def backtest_cache(pair, series):
    now = time.time()
    if pair in _cache_bt and now - _cache_bt[pair][0] < 600:
        return _cache_bt[pair][1]
    res = backtest(series, pair)
    _cache_bt[pair] = (now, res)
    return res


# ───────────── FRAÎCHEUR DES DONNÉES ─────────────

def fraicheur(m5):
    if not m5:
        return {"texte": "Aucune donnée reçue (M5)", "etat": "ferme"}
    fin = m5[-1]["t"] + 300
    age = int((time.time() - fin) / 60)
    dt = datetime.fromtimestamp(fin, BENIN)
    h = f"{JOURS[dt.weekday()]} {dt.strftime('%H:%M')}"
    if age <= 15:
        return {"texte": f"🟢 Données à jour · dernière bougie M5 : {h}", "etat": "ok"}
    if age <= 90:
        return {"texte": f"🟡 Données en retard de {age} min · dernière bougie M5 : {h}", "etat": "warn"}
    return {"texte": f"🌙 Marché fermé ou pas de données · dernière bougie M5 : {h}", "etat": "ferme"}


# ───────────── SIGNAL STRICT (affichage des 7 conditions) ─────────────

def conditions_paire(tfs, ferme=False):
    manque = [k for k in ["H1", "M30", "M15", "M5"] if k not in tfs]
    if manque:
        return {"signal": "?", "couleur": "jaune", "bonus": False,
                "conseil": "Données manquantes : " + ", ".join(manque), "details": {}}

    h1, m30, m15, m5 = tfs["H1"], tfs["M30"], tfs["M15"], tfs["M5"]
    h4 = tfs.get("H4")
    trois = (("H1", h1), ("M30", m30), ("M15", m15))

    def verifier(sens):
        if sens == "haussier":
            fl, a, b, lab = "↑", "pct_acheteurs", "pct_vendeurs", "Ach>Ven"
        else:
            fl, a, b, lab = "↓", "pct_vendeurs", "pct_acheteurs", "Ven>Ach"
        d = {}
        for nom, t in trois:
            d[f"{nom} {fl}"] = t["tendance"] == sens
        for nom, t in trois:
            d[f"{lab} {nom}"] = t[a] > t[b]
        d[f"M5 EMA{fl}"] = m5["ema_signal"] == sens
        bonus = bool(h4 and h4["tendance"] == sens)
        return d, bonus, fl

    res = {s: verifier(s) for s in ("haussier", "baissier")}
    base = None
    for sens, nom, couleur in (("haussier", "ACHAT", "vert"), ("baissier", "VENTE", "rouge")):
        d, bonus, fl = res[sens]
        if all(d.values()):
            conseil = f"Signal {nom} détecté (mode strict). Regarde le verdict final : seul un mode validé sur le passé compte."
            dep = m5["ema_depuis"] or 0
            if dep > 12:
                conseil += f" ⚠️ EMA croisée depuis {dep} bougies : entrée tardive."
            details = {**d, f"H4 {fl}": bonus}
            base = {"signal": nom, "couleur": couleur, "bonus": bonus, "conseil": conseil, "details": details}
            break

    if base is None:
        meilleur = max(("haussier", "baissier"), key=lambda s: sum(res[s][0].values()))
        d, bonus, fl = res[meilleur]
        nom = "ACHAT" if meilleur == "haussier" else "VENTE"
        manque_c = [k for k, v in d.items() if not v]
        ok = len(d) - len(manque_c)
        conseil = f"Mode strict : {ok}/{len(d)} conditions {nom}. Il manque : {', '.join(manque_c)}."
        base = {"signal": "ATTENDRE", "couleur": "jaune", "bonus": False, "conseil": conseil,
                "details": {**d, f"H4 {fl}": bonus}}

    if ferme:
        return {**base, "signal": "FERMÉ", "couleur": "gris", "bonus": False,
                "conseil": "🌙 Marché fermé : ces chiffres viennent de la dernière séance. N'entre pas."}
    return base


# ───────────── ALERTES, VERDICT, GUIDE ─────────────

def alertes_signal(sens, tfs, niv, sltp, mm_h1):
    al = []
    achat = sens == "ACHAT"
    m5, m15 = tfs["M5"], tfs["M15"]
    prix = m5["prix"]

    if achat and niv["res_brut"] is not None and niv["res_brut"] - prix < mm_h1:
        al.append({"txt": f"⚠️ Résistance proche ({niv['res']}) : le prix peut rebondir contre ce plafond.", "cls": "jaune"})
    if not achat and niv["sup_brut"] is not None and prix - niv["sup_brut"] < mm_h1:
        al.append({"txt": f"⚠️ Support proche ({niv['sup']}) : le prix peut rebondir contre ce plancher.", "cls": "jaune"})

    if m5["grande"]:
        al.append({"txt": f"⚠️ Grosse bougie M5 (×{m5['taille_x']}) : le mouvement est déjà fait, attends la prochaine.", "cls": "jaune"})

    cote = sltp["achat" if achat else "vente"]
    if cote is None:
        al.append({"txt": "⚠️ Pas d'objectif clair : le prix est au bord de la fourchette récente.", "cls": "jaune"})
    elif cote["ok"]:
        al.append({"txt": f"✅ Ratio gain/risque {cote['ratio']} : correct (au moins 1,5).", "cls": "vert"})
    else:
        al.append({"txt": f"⚠️ Ratio gain/risque {cote['ratio']} : sous 1,5, le trade ne vaut pas le coup.", "cls": "jaune"})

    cs = "haussiere" if achat else "baissiere"
    if m15["cassure"]["sens"] == cs:
        al.append({"txt": "💥 Cassure M15 dans le sens du signal (elle peut être fausse : vérifie que la bougie suivante tient).", "cls": "vert"})

    pour = "motif_haussier" if achat else "motif_baissier"
    contre = "motif_baissier" if achat else "motif_haussier"
    if m5[pour] or m15[pour]:
        al.append({"txt": "🕯️ Bougie de rejet ou d'englobement dans le même sens (un indice, pas une preuve).", "cls": "vert"})
    if m5[contre]:
        al.append({"txt": "⚠️ Bougie M5 de sens contraire (mèche ou englobante) : prudence.", "cls": "jaune"})
    return al


def construire_verdict(tfs, sltp, bt, ferme, cond):
    if ferme:
        return {"etat": "ferme", "court": "FERMÉ", "couleur": "gris", "sens": None, "mode": None,
                "titre": "🌙 Marché fermé", "txt": "N'entre pas : les chiffres viennent de la dernière séance."}
    if cond["signal"] == "?" or not bt:
        return {"etat": "donnees", "court": "DONNÉES", "couleur": "jaune", "sens": None, "mode": None,
                "titre": "⚠️ Données insuffisantes", "txt": "Il manque des bougies pour juger cette devise. Attends le prochain rafraîchissement."}
    best = bt["meilleur"]
    if not best:
        return {"etat": "aucun", "court": "AUCUN MODE", "couleur": "gris", "sens": None, "mode": None,
                "titre": "⛔ N'entre pas : aucun mode validé",
                "txt": "Sur le passé, aucun des 4 modes n'est validé pour cette devise. Si tu entres quand même, alors tu n'as aucune preuve que le signal marche ici."}
    label = MODES[best]
    sens = signal_live(best, tfs, sltp)
    if sens is None:
        return {"etat": "attente", "court": "ATTENDRE", "couleur": "jaune", "sens": None, "mode": label,
                "titre": "⏳ Attends",
                "txt": f"Le mode « {label} » est validé pour cette devise, mais il n'y a pas de signal maintenant."}
    return {"etat": "entrer", "court": sens, "couleur": "vert" if sens == "ACHAT" else "rouge",
            "sens": sens, "mode": label,
            "titre": f"✅ Entre : {sens}",
            "txt": f"Le mode « {label} » est validé sur le passé et son signal {sens} est actif. Lis le guide avant d'entrer."}


def construire_guide(pair, sens, mode, tfs, sltp):
    achat = sens == "ACHAT"
    cote = sltp["achat" if achat else "vente"] if sltp else None
    if not cote:
        return None
    prix = tfs["M5"]["prix"]
    nxt = (int(time.time() // 300) + 1) * 300
    return {
        "sens": sens, "mode": mode,
        "heure": datetime.fromtimestamp(nxt, BENIN).strftime("%H:%M"),
        "prix": fp(pair, prix),
        "contrat": CONTRAT.get(pair),
        "stop": cote["stop"], "objectif": cote["objectif"], "ratio": cote["ratio"],
        "dist": f"{abs(prix - cote['stop_brut']):.8f}",
        "dist_obj": f"{abs(cote['obj_brut'] - prix):.8f}",
        "moitie": fp(pair, prix + (cote["obj_brut"] - prix) / 2),
    }


# ───────────── ANALYSE D'UNE PAIRE ─────────────

def analyser_paire(pair, source):
    series, resultats = {}, {}
    for tf, (interval, range_, gran) in TIMEFRAMES.items():
        if source == "yahoo":
            brut = recuperer_bougies_yahoo(YAHOO_PAIRS[pair], interval, range_)
            if tf == "H4":
                brut = regrouper(brut, gran)
        else:
            brut = recuperer_bougies_deriv(DERIV_PAIRS[pair], gran, DERIV_COUNT[tf])
        bougies = bougies_fermees(brut, gran)
        series[tf] = bougies
        r = analyser(tf, bougies)
        if r:
            resultats[tf] = r

    fr = fraicheur(series.get("M5"))
    ferme = fr["etat"] == "ferme"
    stats = calc_stats(pair, series.get("H1", []), resultats)
    cond = conditions_paire(resultats, ferme)
    prix = fp(pair, resultats["M5"]["prix"]) if "M5" in resultats else "—"

    niv = sltp = bt = heures = guide = None
    alertes = []
    h1, m15 = series.get("H1", []), series.get("M15", [])
    if "M5" in resultats and len(h1) >= 30 and len(m15) >= 30:
        p = resultats["M5"]["prix"]
        niv = calc_niveaux(pair, p, m15, h1)
        sltp = calc_sltp(pair, p, m15)
        heures = calc_heures(h1)
        bt = backtest_cache(pair, series)

    verdict = construire_verdict(resultats, sltp, bt, ferme, cond)
    if verdict["etat"] == "entrer" and niv:
        alertes = alertes_signal(verdict["sens"], resultats, niv, sltp, mouvement_moyen(h1, 24))
        jaunes = [a["txt"] for a in alertes if a["cls"] == "jaune"]
        if jaunes:
            verdict = {"etat": "attente", "court": "ATTENDRE", "couleur": "jaune", "sens": None,
                       "mode": verdict["mode"], "titre": "⏳ Attends : le signal est actif mais pas idéal",
                       "txt": "Le mode validé donne un signal, mais : " + " ".join(jaunes)}
        else:
            guide = construire_guide(pair, verdict["sens"], verdict["mode"], resultats, sltp)
            if guide is None:
                verdict = {"etat": "attente", "court": "ATTENDRE", "couleur": "jaune", "sens": None,
                           "mode": verdict["mode"], "titre": "⏳ Attends",
                           "txt": "Pas de stop ni d'objectif clair pour le moment."}

    return {"tfs": resultats, "stats": stats, "fraicheur": fr, "cond": cond, "prix": prix,
            "niv": niv, "sltp": sltp, "bt": bt, "heures": heures, "guide": guide,
            "alertes": alertes, "verdict": verdict}


def analyser_securise(pair, source):
    try:
        return analyser_paire(pair, source)
    except Exception as e:
        print(f"Erreur {pair}: {e}")
        return {"tfs": {}, "stats": None,
                "fraicheur": {"texte": "Erreur de chargement", "etat": "ferme"},
                "cond": conditions_paire({}), "prix": "—",
                "niv": None, "sltp": None, "bt": None, "heures": None, "guide": None, "alertes": [],
                "verdict": {"etat": "donnees", "court": "ERREUR", "couleur": "jaune", "sens": None, "mode": None,
                            "titre": "⚠️ Erreur de chargement", "txt": "Réessaie au prochain rafraîchissement."}}


def analyser_tout():
    taches = [(p, "yahoo") for p in YAHOO_PAIRS] + [(p, "deriv") for p in DERIV_PAIRS]
    res = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        for (pair, _), data in zip(taches, ex.map(lambda t: analyser_securise(*t), taches)):
            res[pair] = data
    return res


# ───────────── ROUTES ─────────────

@app.route("/")
def index():
    if logged():
        return redirect(url_for("dashboard"))
    return render_template("login.html", error=None)


@app.route("/login", methods=["POST"])
def login():
    if hashlib.sha256(request.form.get("password", "").encode()).hexdigest() == ADMIN_HASH:
        session["admin"] = True
        return redirect(url_for("dashboard"))
    return render_template("login.html", error="Mot de passe incorrect")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


@app.route("/dashboard")
def dashboard():
    if not logged():
        return redirect(url_for("index"))
    paires = analyser_tout()
    return render_template("dashboard.html", paires=paires, now=datetime.now(BENIN).strftime("%H:%M"))


@app.route("/ping")
def ping():
    return "ok", 200


@app.route("/cron")
def cron():
    analyser_tout()
    return "ok", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
