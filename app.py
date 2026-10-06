import os, json, hashlib, secrets, time, bisect
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
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
CONTRAT = {"EURUSD": 100000, "GBPUSD": 100000, "XAUUSD": 100, "V75": None}
TIMEFRAMES = {
    "H4": ("1h", "60d", 14400),
    "H1": ("1h", "60d", 3600),
    "M30": ("30m", "30d", 1800),
    "M15": ("15m", "15d", 900),
    "M5": ("5m", "7d", 300),
}
DERIV_COUNT = {"H4": 300, "H1": 300, "M30": 300, "M15": 400, "M5": 1000}
EMA_COURT = 5
EMA_LONG = 25
DERIV_WS = "wss://api.derivws.com/trading/v1/options/ws/public"
CACHE_SEC = {"1h": 300, "30m": 180, "15m": 120, "5m": 60}
_cache_yahoo = {}

POIDS_TF = {"M5": 5, "M15": 4, "M30": 3, "H1": 2, "H4": 1}
ORDRE_TF = ["M5", "M15", "M30", "H1", "H4"]

RSI_PERIODE = {"M5": 7, "M15": 14, "M30": 14, "H1": 14, "H4": 14}
PERIODE_TENDANCE = {"M5": 30, "M15": 25, "M30": 20, "H1": 15, "H4": 10}

CORRELATIONS = {"EURUSD": ["GBPUSD"], "GBPUSD": ["EURUSD"], "XAUUSD": [], "V75": []}

SEUIL_AGITATION_MAX = 70
DUREE_SIGNAL_MIN = 30
JOURNAL_MAX = 200
TIMING_FRAIS_MAX = 5
TIMING_MOYEN_MAX = 12
HTTP_TIMEOUT = 6

# ═══════════════════════════════════════════════════════
# SEUILS DES MOTIFS DE CONTINUATION (basés sur Bulkowski)
# ═══════════════════════════════════════════════════════
SEUILS_MOTIFS = {
    "rising_three": 0.02,       # écart max entre les 3 petites bougies
    "separating": 0.015,        # écart max pour separating lines
    "deliberation": 0.01,       # petite bougie avant continuation
    "three_line_strike": 0.005, # écart max pour 3 line strike
}


def logged():
    return session.get("admin") is True


def fp(pair, x):
    return f"{x:.{DECIMALES.get(pair, 5)}f}"


# ───────────── RÉCUPÉRATION ─────────────

def recuperer_bougies_deriv(symbol, granularity, count=300):
    try:
        ws = websocket.create_connection(DERIV_WS, timeout=HTTP_TIMEOUT)
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
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=HTTP_TIMEOUT)
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
    if bougies and bougies[-1]["t"] + gran > time.time():
        return bougies[:-1]
    return bougies


# ───────────── CALCULS ─────────────

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
    if len(clotures) < 3:
        return "neutre", 0.0
    x = np.arange(1, len(clotures) + 1)
    try:
        pente = np.polyfit(x, np.array(clotures), 1)[0]
    except Exception:
        return "neutre", 0.0
    prix_moy = float(np.mean(clotures))
    if prix_moy <= 0:
        return "neutre", 0.0
    pct = round(pente / prix_moy * 10000, 2)
    if pct > 0.02:
        return "haussier", pct
    if pct < -0.02:
        return "baissier", pct
    return "neutre", pct


def calc_rsi(bougies, periode=14):
    if len(bougies) < periode + 1:
        return 50.0
    clotures = [b["close"] for b in bougies]
    gains = []
    pertes = []
    for i in range(1, len(clotures)):
        diff = clotures[i] - clotures[i - 1]
        if diff > 0:
            gains.append(diff)
            pertes.append(0.0)
        elif diff < 0:
            gains.append(0.0)
            pertes.append(abs(diff))
        else:
            gains.append(0.0)
            pertes.append(0.0)
    if len(gains) < periode:
        return 50.0
    gain_moy = float(np.mean(gains[:periode]))
    perte_moy = float(np.mean(pertes[:periode]))
    for i in range(periode, len(gains)):
        gain_moy = (gain_moy * (periode - 1) + gains[i]) / periode
        perte_moy = (perte_moy * (periode - 1) + pertes[i]) / periode
    if perte_moy == 0:
        return 100.0
    rs = gain_moy / perte_moy
    rsi = 100 - (100 / (1 + rs))
    return round(rsi, 1)


def calc_agitation(bougies, periode=10):
    d = bougies[-periode:]
    if len(d) < 2:
        return 0.0
    total = sum(b["high"] - b["low"] for b in d)
    if total == 0:
        return 0.0
    net = abs(d[-1]["close"] - d[0]["open"])
    efficacite = min(100.0, net / total * 100)
    return round(100 - efficacite, 1)


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
            r1 = calc_rsi(bougies[:c1 + 1], 14)
            r2 = calc_rsi(bougies[:c2 + 1], 14)
            if r2 > r1:
                div_h = True
    div_b = False
    if len(sommets) >= 2:
        s1, s2 = sommets[-2], sommets[-1]
        if bougies[s2]["high"] > bougies[s1]["high"]:
            r1 = calc_rsi(bougies[:s1 + 1], 14)
            r2 = calc_rsi(bougies[:s2 + 1], 14)
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


# ═══════════════════════════════════════════════════════
# MOTIFS DE CONTINUATION RÉPUTÉS (Bulkowski)
# ═══════════════════════════════════════════════════════

def motif_rising_three_methods(bougies):
    """Rising Three Methods (79% succès) : 1 grande verte, 3 petites rouges, 1 grande verte."""
    if len(bougies) < 5:
        return None
    b = bougies[-5:]
    # 1ère bougie : grande verte
    if b[0]["close"] <= b[0]["open"]:
        return None
    # 3 petites rouges
    for i in range(1, 4):
        if b[i]["close"] >= b[i]["open"]:
            return None
        # corps petit
        corps = abs(b[i]["close"] - b[i]["open"])
        rng = b[i]["high"] - b[i]["low"]
        if rng > 0 and corps / rng > 0.4:
            return None
    # Dernière grande verte qui dépasse la 1ère
    if b[4]["close"] <= b[4]["open"]:
        return None
    if b[4]["close"] > b[0]["close"]:
        return {"nom": "Rising Three Methods", "sens": "haussier", "fiabilite": 79,
                "desc": "3 petites bougies rouges après une grande verte, puis cassure haussière"}
    return None


def motif_falling_three_methods(bougies):
    """Falling Three Methods : 1 grande rouge, 3 petites vertes, 1 grande rouge."""
    if len(bougies) < 5:
        return None
    b = bougies[-5:]
    if b[0]["close"] >= b[0]["open"]:
        return None
    for i in range(1, 4):
        if b[i]["close"] <= b[i]["open"]:
            return None
        corps = abs(b[i]["close"] - b[i]["open"])
        rng = b[i]["high"] - b[i]["low"]
        if rng > 0 and corps / rng > 0.4:
            return None
    if b[4]["close"] >= b[4]["open"]:
        return None
    if b[4]["close"] < b[0]["close"]:
        return {"nom": "Falling Three Methods", "sens": "baissier", "fiabilite": 79,
                "desc": "3 petites bougies vertes après une grande rouge, puis cassure baissière"}
    return None


def motif_separating_lines(bougies):
    """Separating Lines (76% succès) : continuation après pullback."""
    if len(bougies) < 3:
        return None
    b = bougies[-3:]
    # Bougie 1 et 2 : même sens (tendance)
    if b[0]["close"] > b[0]["open"] and b[1]["close"] > b[1]["open"]:
        # Bougie 3 : rouge qui ouvre près de l'ouverture de la bougie 2
        if b[2]["close"] < b[2]["open"]:
            ecart = abs(b[2]["open"] - b[1]["open"]) / b[1]["open"]
            if ecart < SEUILS_MOTIFS["separating"]:
                return {"nom": "Bearish Separating Lines", "sens": "haussier", "fiabilite": 76,
                        "desc": "Ligne de séparation après 2 vertes, reprise haussière probable"}
    if b[0]["close"] < b[0]["open"] and b[1]["close"] < b[1]["open"]:
        if b[2]["close"] > b[2]["open"]:
            ecart = abs(b[2]["open"] - b[1]["open"]) / b[1]["open"]
            if ecart < SEUILS_MOTIFS["separating"]:
                return {"nom": "Bullish Separating Lines", "sens": "baissier", "fiabilite": 76,
                        "desc": "Ligne de séparation après 2 rouges, reprise baissière probable"}
    return None


def motif_deliberation(bougies):
    """Deliberation (75% succès) : grande bougie + 2 petites + grande bougie."""
    if len(bougies) < 4:
        return None
    b = bougies[-4:]
    # Grande bougie 1
    rng1 = b[0]["high"] - b[0]["low"]
    corps1 = abs(b[0]["close"] - b[0]["open"])
    if rng1 <= 0 or corps1 / rng1 < 0.7:
        return None
    # 2 petites bougies
    for i in range(1, 3):
        rng = b[i]["high"] - b[i]["low"]
        corps = abs(b[i]["close"] - b[i]["open"])
        if rng <= 0 or corps / rng > 0.3:
            return None
    # Dernière grande bougie même sens
    rng3 = b[3]["high"] - b[3]["low"]
    corps3 = abs(b[3]["close"] - b[3]["open"])
    if rng3 <= 0 or corps3 / rng3 < 0.7:
        return None
    if b[0]["close"] > b[0]["open"] and b[3]["close"] > b[3]["open"]:
        return {"nom": "Bullish Deliberation", "sens": "haussier", "fiabilite": 75,
                "desc": "Grande verte, 2 petites hésitations, grande verte de reprise"}
    if b[0]["close"] < b[0]["open"] and b[3]["close"] < b[3]["open"]:
        return {"nom": "Bearish Deliberation", "sens": "baissier", "fiabilite": 75,
                "desc": "Grande rouge, 2 petites hésitations, grande rouge de reprise"}
    return None


def motif_three_line_strike(bougies):
    """Three Line Strike : 3 bougies dans un sens + 1 grande bougie inverse."""
    if len(bougies) < 4:
        return None
    b = bougies[-4:]
    if b[0]["close"] > b[0]["open"] and b[1]["close"] > b[1]["open"] and b[2]["close"] > b[2]["open"]:
        if b[3]["close"] < b[3]["open"]:
            # La grande bougie rouge doit englober les 3 précédentes
            if b[3]["close"] < b[0]["open"] and b[3]["open"] > b[2]["close"]:
                return {"nom": "Bullish Three Line Strike", "sens": "haussier", "fiabilite": 65,
                        "desc": "3 vertes suivies d'une grande rouge (faux signal) → reprise haussière"}
    if b[0]["close"] < b[0]["open"] and b[1]["close"] < b[1]["open"] and b[2]["close"] < b[2]["open"]:
        if b[3]["close"] > b[3]["open"]:
            if b[3]["close"] > b[0]["open"] and b[3]["open"] < b[2]["close"]:
                return {"nom": "Bearish Three Line Strike", "sens": "baissier", "fiabilite": 65,
                        "desc": "3 rouges suivies d'une grande verte (faux signal) → reprise baissière"}
    return None


def detecter_motifs_continuation(bougies):
    """Retourne la liste des motifs de continuation détectés."""
    motifs = []
    for f in [motif_rising_three_methods, motif_falling_three_methods,
              motif_separating_lines, motif_deliberation, motif_three_line_strike]:
        try:
            m = f(bougies)
            if m:
                motifs.append(m)
        except Exception:
            pass
    return motifs


def calc_motifs_classiques(bougies):
    """Mèches de rejet, englobante, bougie anormale."""
    if len(bougies) < 2:
        return [], False, 0, False, False
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

    per_tend = PERIODE_TENDANCE.get(tf, 15)
    per_rsi = RSI_PERIODE.get(tf, 14)

    tendance, pct_tendance = calc_tendance(clotures[-per_tend:])
    rsi = calc_rsi(bougies, per_rsi)
    ag = calc_agitation(bougies)
    creux = trouver_creux(bougies)
    sommets = trouver_sommets(bougies)
    divergence = calc_divergence(bougies, creux, sommets)
    ema5, ema25, ema_sig, ema_depuis = calc_ema_signal(clotures)
    motifs, grande, taille_x, mh, mb = calc_motifs_classiques(bougies)
    continuation = detecter_motifs_continuation(bougies)

    pa = round(rsi, 1)
    pv = round(100 - rsi, 1)

    return {
        "tf": tf, "tendance": tendance, "pct_tendance": pct_tendance,
        "pct_acheteurs": pa, "pct_vendeurs": pv,
        "rsi": rsi, "rsi_periode": per_rsi,
        "pct_agitation": ag,
        "periode_tend": per_tend,
        "divergence": divergence,
        "ema5": ema5, "ema25": ema25, "ema_signal": ema_sig, "ema_depuis": ema_depuis,
        "motifs": motifs, "grande": grande, "taille_x": taille_x,
        "motif_haussier": mh, "motif_baissier": mb,
        "continuation": continuation,
        "cassure": calc_cassure(bougies),
        "prix": clotures[-1]
    }


def calc_timing_entree(m5):
    if not m5 or m5.get("ema_depuis") is None:
        return None
    dep = m5["ema_depuis"]
    if dep <= TIMING_FRAIS_MAX:
        return {"niveau": "FRAIS", "couleur": "vert", "emoji": "🔥",
                "txt": f"Croisement EMA M5 très récent ({dep} bougies). Tu entres au début du mouvement.",
                "lot_conseil": "LOT PLEIN", "lot_ratio": 1.0}
    if dep <= TIMING_MOYEN_MAX:
        return {"niveau": "EN COURS", "couleur": "jaune", "emoji": "⏳",
                "txt": f"Croisement EMA M5 il y a {dep} bougies. Le mouvement est déjà lancé.",
                "lot_conseil": "DEMI-LOT", "lot_ratio": 0.5}
    return {"niveau": "TARDIF", "couleur": "rouge", "emoji": "❌",
            "txt": f"Croisement EMA M5 il y a {dep} bougies. Le mouvement est trop avancé.",
            "lot_conseil": "ATTENDRE", "lot_ratio": 0.0}


def calc_tendance_generale(tfs):
    if not tfs:
        return None
    poids_total = 0
    score = 0.0
    somme_rsi = 0.0
    for tf, data in tfs.items():
        p = POIDS_TF.get(tf, 1)
        poids_total += p
        somme_rsi += data["rsi"] * p
        if data["tendance"] == "haussier":
            vote = 1.0
        elif data["tendance"] == "baissier":
            vote = -1.0
        else:
            vote = 0.0
        score += vote * p
    if poids_total == 0:
        return None
    rsi_gen = round(somme_rsi / poids_total, 1)
    pa_gen = rsi_gen
    pv_gen = round(100 - rsi_gen, 1)
    score_pct_brut = score / poids_total * 100
    sens_list = [t["tendance"] for t in tfs.values()]
    nb_haut = sum(1 for s in sens_list if s == "haussier")
    nb_bas = sum(1 for s in sens_list if s == "baissier")
    total_tf = len(sens_list)
    bonus = 0.0
    if nb_haut == total_tf:
        bonus = 10.0
    elif nb_bas == total_tf:
        bonus = -10.0
    score_pct = round(score_pct_brut + bonus, 1)
    if score_pct >= 60:
        tendance, couleur = "ACHAT FORT", "vert"
    elif score_pct >= 25:
        tendance, couleur = "ACHAT", "vert"
    elif score_pct <= -60:
        tendance, couleur = "VENTE FORTE", "rouge"
    elif score_pct <= -25:
        tendance, couleur = "VENTE", "rouge"
    else:
        tendance, couleur = "NEUTRE", ""
    alignement = {"haussier": 0, "baissier": 0, "detail": []}
    for sens in ("haussier", "baissier"):
        compte = 0
        for tf in ORDRE_TF:
            if tf not in tfs:
                break
            if tfs[tf]["tendance"] == sens:
                compte += 1
            else:
                break
        alignement[sens] = compte
    for tf in ORDRE_TF:
        if tf in tfs:
            alignement["detail"].append({"tf": tf, "sens": tfs[tf]["tendance"]})
    if alignement["haussier"] >= alignement["baissier"] and alignement["haussier"] > 0:
        align_sens, align_n = "haussier", alignement["haussier"]
    elif alignement["baissier"] > 0:
        align_sens, align_n = "baissier", alignement["baissier"]
    else:
        align_sens, align_n = None, 0
    confirmation = None
    trois = ["M15", "M30", "H1"]
    if all(tf in tfs for tf in trois) and "M5" in tfs:
        m15, m30, h1, m5 = tfs["M15"], tfs["M30"], tfs["H1"], tfs["M5"]
        if m15["tendance"] == m30["tendance"] == h1["tendance"] == "haussier":
            if m5["ema_signal"] == "haussier" and (m5["ema_depuis"] or 99) <= 3:
                confirmation = {"sens": "ACHAT", "couleur": "vert",
                                "txt": "M15+M30+H1 haussiers + croisement EMA M5 haussier récent."}
        elif m15["tendance"] == m30["tendance"] == h1["tendance"] == "baissier":
            if m5["ema_signal"] == "baissier" and (m5["ema_depuis"] or 99) <= 3:
                confirmation = {"sens": "VENTE", "couleur": "rouge",
                                "txt": "M15+M30+H1 baissiers + croisement EMA M5 baissier récent."}
    votes_detail = []
    for tf in ORDRE_TF:
        if tf in tfs:
            p = POIDS_TF.get(tf, 1)
            v = 1 if tfs[tf]["tendance"] == "haussier" else (-1 if tfs[tf]["tendance"] == "baissier" else 0)
            votes_detail.append({"tf": tf, "vote": v, "poids": p, "contribution": v * p,
                                 "rsi": tfs[tf]["rsi"]})
    # Collecter tous les motifs de continuation détectés
    continuations = []
    for tf in ORDRE_TF:
        if tf in tfs:
            for m in tfs[tf]["continuation"]:
                continuations.append({**m, "tf": tf})
    return {"rsi_gen": rsi_gen, "ach_gen": pa_gen, "ven_gen": pv_gen,
            "score_pct": score_pct, "score_brut": round(score_pct_brut, 1),
            "bonus": bonus,
            "tendance": tendance, "couleur": couleur,
            "align_sens": align_sens, "align_n": align_n,
            "align_detail": alignement["detail"],
            "confirmation": confirmation,
            "votes_detail": votes_detail,
            "continuations": continuations}


def session_actuelle():
    h = datetime.now(BENIN).hour
    if 1 <= h < 9:
        return {"heure": h, "qualite": "faible", "nom": "Asie", "txt": "Asie (peu de volume, signaux peu fiables)"}
    if 9 <= h < 15:
        return {"heure": h, "qualite": "forte", "nom": "Londres", "txt": "Londres (session active, bonne fiabilité)"}
    if 15 <= h < 18:
        return {"heure": h, "qualite": "excellente", "nom": "Londres+NY", "txt": "Chevauchement Londres + New York (meilleure fenêtre)"}
    if 18 <= h < 23:
        return {"heure": h, "qualite": "forte", "nom": "New York", "txt": "New York (session active, bonne fiabilité)"}
    return {"heure": h, "qualite": "faible", "nom": "Nuit", "txt": "Nuit (marché calme)"}


# ───────────── MOTEUR DE CONSEILS ─────────────

def generer_conseil(pair, d, session_act, autres_cond, historique):
    tfs = d["tfs"]
    cond = d["cond"]
    gen = d["generale"]
    niv = d["niv"]
    sltp = d["sltp"]
    bt = d["bt"]
    raisons = []
    sig = cond["signal"]
    if not tfs or not gen:
        return {"verdict": "PAS DE DONNÉES", "couleur": "gris", "action": "ATTENDRE",
                "message": f"Je n'ai pas assez de données sur {pair} pour te conseiller.",
                "raisons": ["Données insuffisantes ou marché fermé."]}
    if sig not in ("ACHAT", "VENTE"):
        manque = [k for k, v in cond.get("details", {}).items() if not v]
        msg = f"Pas de signal sur {pair}. Le marché n'est pas aligné."
        if manque:
            msg += f" Il manque : {', '.join(manque[:3])}{'...' if len(manque) > 3 else ''}."
        return {"verdict": "ATTENDRE", "couleur": "gris", "action": "NE RIEN FAIRE",
                "message": msg, "raisons": ["Aucun signal complet détecté."]}
    timing = cond.get("timing")
    achat = sig == "ACHAT"
    if session_act["qualite"] == "faible":
        raisons.append(f"Session {session_act['nom']} : peu de volume, signaux peu fiables.")
    if timing:
        if timing["niveau"] == "EN COURS":
            raisons.append(f"Timing d'entrée EN COURS ({tfs['M5']['ema_depuis']} bougies).")
        elif timing["niveau"] == "TARDIF":
            raisons.append(f"Timing d'entrée TARDIF ({tfs['M5']['ema_depuis']} bougies). Mouvement trop avancé.")
    h4 = tfs.get("H4")
    h1 = tfs.get("H1")
    contre = "baissier" if achat else "haussier"
    h4_contre = h4 and h4["tendance"] == contre
    h1_contre = h1 and h1["tendance"] == contre
    if h4_contre and h1_contre:
        raisons.append(f"H4 ET H1 sont {contre}s (contre ton signal). Pullback dans une tendance inverse.")
    elif h4_contre:
        raisons.append(f"H4 est {contre} (contre ton signal). Le fond est contre toi.")
    elif h1_contre:
        raisons.append(f"H1 est {contre} (contre ton signal). La tendance horaire n'est pas retournée.")
    h1_data = tfs.get("H1")
    if h1_data and h1_data.get("pct_agitation", 0) > SEUIL_AGITATION_MAX:
        raisons.append(f"Marché trop agité (agitation H1 {h1_data['pct_agitation']}%).")
    rsi_m5 = tfs["M5"]["rsi"]
    if achat and rsi_m5 > 80:
        raisons.append(f"RSI M5 en excès haussier ({rsi_m5}) : risque de retournement imminent.")
    if not achat and rsi_m5 < 20:
        raisons.append(f"RSI M5 en excès baissier ({rsi_m5}) : risque de rebond imminent.")
    cote = sltp["achat" if achat else "vente"] if sltp else None
    if cote is None:
        raisons.append("Pas d'objectif clair devant le prix.")
    elif not cote["ok"]:
        raisons.append(f"Ratio gain/risque faible ({cote['ratio']}, sous 1,5).")
    m5 = tfs["M5"]
    if m5["grande"]:
        raisons.append(f"Grosse bougie M5 (×{m5['taille_x']}). Le mouvement est déjà fait d'un coup.")
    if niv and h1_data:
        mm = h1_data["prix"] * 0.001
        if achat and niv["res_brut"] is not None and niv["res_brut"] - m5["prix"] < mm:
            raisons.append(f"Résistance proche ({niv['res']}).")
        if not achat and niv["sup_brut"] is not None and m5["prix"] - niv["sup_brut"] < mm:
            raisons.append(f"Support proche ({niv['sup']}).")
    if historique:
        maintenant = time.time()
        for h in reversed(historique):
            if h["pair"] == pair and h["sens"] == sig:
                age = (maintenant - h["t"]) / 60
                if age < DUREE_SIGNAL_MIN:
                    raisons.append(f"Signal {sig} identique il y a {int(age)} min. Même mouvement.")
                break
    for c in CORRELATIONS.get(pair, []):
        if c in autres_cond and autres_cond[c].get("signal") == sig:
            raisons.append(f"{c} a le même signal. Doubler ton risque.")
    if bt and bt["n"] >= 10 and bt["pct"] is not None and bt["pct"] < 45:
        raisons.append(f"Sur le passé, ce signal n'a réussi que {bt['pct']}% du temps.")
    bloquants = [r for r in raisons if "pullback" in r or "trop avancé" in r or "pas d'objectif" in r.lower() or "doubler ton risque" in r or "peu fiables" in r or "excès" in r]
    avertissements = [r for r in raisons if r not in bloquants]
    if timing and timing["niveau"] == "TARDIF":
        return {"verdict": "N'ENTRE PAS", "couleur": "rouge", "action": "ATTENDRE LE PROCHAIN CROISEMENT",
                "message": f"Signal {sig} sur {pair} trop tardif. Croisement EMA M5 il y a {tfs['M5']['ema_depuis']} bougies. Attends un nouveau croisement.",
                "raisons": raisons}
    if bloquants:
        return {"verdict": "N'ENTRE PAS", "couleur": "rouge", "action": "PASSER CE TRADE",
                "message": f"Signal {sig} sur {pair} MAIS quelque chose bloque.",
                "raisons": raisons}
    if avertissements:
        return {"verdict": "PRUDENCE — DEMI-LOT", "couleur": "jaune", "action": "RÉDUIRE LE LOT DE MOITIÉ",
                "message": f"Signal {sig} sur {pair}, points de vigilance. Réduis ton lot de moitié.",
                "raisons": raisons}
    lot_msg = "lot plein" if not timing or timing["niveau"] == "FRAIS" else "demi-lot"
    rsi_mention = f" RSI M5 : {rsi_m5}." if rsi_m5 else ""
    return {"verdict": "ENTRE MAINTENANT", "couleur": "vert", "action": f"ACHAT/VENTE AU MARCHÉ — {lot_msg.upper()}",
            "message": f"Setup propre sur {pair}. Signal {sig}, timing {timing['niveau'] if timing else 'OK'}.{rsi_mention} Entre à la clôture M5 avec le {lot_msg}.",
            "raisons": raisons or ["Tous les filtres au vert."]}


# ───────────── STATS ─────────────

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
    return {"var24": var(24), "var5j": var(120),
            "bas24": fp(pair, bas24), "haut24": fp(pair, haut24),
            "bas5j": fp(pair, bas5), "haut5j": fp(pair, haut5),
            "position": pos, "vol24": vol24, "vol_etat": vol_etat,
            "vertes": vertes, "serie_n": n_serie,
            "serie_sens": "hausse" if sens == 1 else "baisse" if sens == -1 else "—"}


def calc_niveaux(pair, prix, m15, h1):
    res, sup = [], []
    for bl in (m15, h1):
        for i in trouver_sommets(bl, 80, 3):
            res.append(bl[i]["high"])
        for i in trouver_creux(bl, 80, 3):
            sup.append(bl[i]["low"])
    r = min([x for x in res if x > prix], default=None)
    s = max([x for x in sup if x < prix], default=None)
    return {"res": fp(pair, r) if r is not None else None, "res_brut": r,
            "res_pct": round((r - prix) / prix * 100, 3) if r is not None else None,
            "sup": fp(pair, s) if s is not None else None, "sup_brut": s,
            "sup_pct": round((prix - s) / prix * 100, 3) if s is not None else None}


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
    return {"stop": fp(pair, stop), "objectif": fp(pair, obj),
            "stop_brut": stop, "obj_brut": obj,
            "risque_pct": round(risque / prix * 100, 3),
            "gain_pct": round(gain / prix * 100, 3),
            "ratio": round(ratio, 2), "ok": ratio >= 1.5}


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


def tendance_simple(cl):
    x = np.arange(len(cl))
    p = np.polyfit(x, np.array(cl), 1)[0]
    return 1 if p > 0 else -1 if p < 0 else 0


def backtest(series, pair="", horizon=12, test=500, max_trade=48):
    if any(len(series.get(tf, [])) < 30 for tf in ("H1", "M30", "M15", "M5")):
        return None
    m5 = series["M5"]
    ends = {tf: [b["t"] + g for b in series[tf]] for tf, g in (("H1", 3600), ("M30", 1800), ("M15", 900))}
    debut = max(40, len(m5) - test)
    n = {"ACHAT": 0, "VENTE": 0}
    w = {"ACHAT": 0, "VENTE": 0}
    sim = {"tp": 0, "sl": 0, "to": 0, "r": []}
    prev = None
    for i in range(debut, len(m5) - horizon):
        if m5[i + horizon]["t"] - m5[i]["t"] > horizon * 300 * 1.5:
            prev = None
            continue
        t = m5[i]["t"] + 300
        etats, ok = set(), True
        for tf in ("H1", "M30", "M15"):
            k = bisect.bisect_right(ends[tf], t)
            sl = series[tf][max(0, k - 60):k]
            if len(sl) < 30:
                ok = False
                break
            per_tend = PERIODE_TENDANCE.get(tf, 15)
            tend = tendance_simple([b["close"] for b in sl[-per_tend:]])
            per_rsi = RSI_PERIODE.get(tf, 14)
            rsi = calc_rsi(sl, per_rsi)
            if tend > 0 and rsi > 50:
                etats.add(1)
            elif tend < 0 and rsi < 50:
                etats.add(-1)
            else:
                etats.add(0)
        if not ok:
            prev = None
            continue
        sig = None
        if etats == {1} or etats == {-1}:
            s = next(iter(etats))
            cl = [b["close"] for b in m5[max(0, i - 100):i + 1]]
            e5, e25 = calc_ema_serie(cl, EMA_COURT), calc_ema_serie(cl, EMA_LONG)
            if e5 and e25:
                if s == 1 and e5[-1] > e25[-1]:
                    sig = "ACHAT"
                elif s == -1 and e5[-1] < e25[-1]:
                    sig = "VENTE"
        if sig and sig != prev:
            n[sig] += 1
            entree, sortie = m5[i]["close"], m5[i + horizon]["close"]
            if (sig == "ACHAT" and sortie > entree) or (sig == "VENTE" and sortie < entree):
                w[sig] += 1
            k15 = bisect.bisect_right(ends["M15"], t)
            sl15 = series["M15"][max(0, k15 - 100):k15]
            if len(sl15) >= 30:
                cote = calc_sltp(pair, entree, sl15)["achat" if sig == "ACHAT" else "vente"]
                if cote:
                    st, ob = cote["stop_brut"], cote["obj_brut"]
                    risque = abs(entree - st)
                    fin = min(len(m5) - 1, i + max_trade)
                    res_r = None
                    for j in range(i + 1, fin + 1):
                        bj = m5[j]
                        if sig == "ACHAT":
                            if bj["low"] <= st:
                                res_r = -1.0
                                sim["sl"] += 1
                                break
                            if bj["high"] >= ob:
                                res_r = cote["ratio"]
                                sim["tp"] += 1
                                break
                        else:
                            if bj["high"] >= st:
                                res_r = -1.0
                                sim["sl"] += 1
                                break
                            if bj["low"] <= ob:
                                res_r = cote["ratio"]
                                sim["tp"] += 1
                                break
                    if res_r is None and fin == i + max_trade:
                        dernier = m5[fin]["close"]
                        res_r = ((dernier - entree) if sig == "ACHAT" else (entree - dernier)) / risque
                        sim["to"] += 1
                    if res_r is not None:
                        sim["r"].append(res_r)
        prev = sig
    total, gagnes = n["ACHAT"] + n["VENTE"], w["ACHAT"] + w["VENTE"]
    pct = round(gagnes / total * 100, 1) if total else None
    if total < 8:
        verdict, couleur = "Trop peu de signaux pour conclure.", ""
    elif pct >= 55:
        verdict, couleur = "Plutôt bon sur cette période.", "vert"
    elif pct >= 45:
        verdict, couleur = "Proche du hasard (50 %).", ""
    else:
        verdict, couleur = "Mauvais sur cette période.", "rouge"
    sim_n = len(sim["r"])
    sim_r = round(float(np.mean(sim["r"])), 2) if sim_n else None
    if sim_n < 8:
        sim_verdict, sim_couleur = "Trop peu de trades simulés.", ""
    elif sim_r > 0.15:
        sim_verdict, sim_couleur = "Gain moyen positif sur cette période.", "vert"
    elif sim_r < -0.1:
        sim_verdict, sim_couleur = "Perte moyenne sur cette période.", "rouge"
    else:
        sim_verdict, sim_couleur = "Résultat proche de zéro.", ""
    heures = round((m5[-1]["t"] - m5[debut]["t"]) / 3600)
    return {"n": total, "wins": gagnes, "pct": pct, "couleur": couleur, "heures": heures,
            "achat_n": n["ACHAT"], "achat_w": w["ACHAT"], "vente_n": n["VENTE"], "vente_w": w["VENTE"],
            "verdict": verdict, "sim_n": sim_n, "sim_tp": sim["tp"], "sim_sl": sim["sl"],
            "sim_to": sim["to"], "sim_r": sim_r, "sim_couleur": sim_couleur, "sim_verdict": sim_verdict}


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


def conditions_paire(tfs, ferme=False, generale=None):
    manque = [k for k in ["H1", "M30", "M15", "M5"] if k not in tfs]
    if manque:
        return {"signal": "?", "couleur": "jaune", "bonus": False,
                "conseil": "Données manquantes : " + ", ".join(manque),
                "details": {}, "alertes": [], "generale": generale, "timing": None}
    h1, m30, m15, m5 = tfs["H1"], tfs["M30"], tfs["M15"], tfs["M5"]
    h4 = tfs.get("H4")
    trois = (("H1", h1), ("M30", m30), ("M15", m15))
    def verifier(sens):
        if sens == "haussier":
            fl, a, b, lab = "↑", "rsi", "rsi_bas", "RSI>50"
        else:
            fl, a, b, lab = "↓", "rsi_bas", "rsi", "RSI<50"
        d = {}
        for nom, t in trois:
            d[f"{nom} {fl}"] = t["tendance"] == sens
        for nom, t in trois:
            if sens == "haussier":
                d[f"{lab} {nom}"] = t["rsi"] > 50
            else:
                d[f"{lab} {nom}"] = t["rsi"] < 50
        d[f"M5 EMA{fl}"] = m5["ema_signal"] == sens
        bonus = bool(h4 and h4["tendance"] == sens)
        return d, bonus, fl
    res = {s: verifier(s) for s in ("haussier", "baissier")}
    timing = calc_timing_entree(m5)
    base = None
    for sens, nom, couleur in (("haussier", "ACHAT", "vert"), ("baissier", "VENTE", "rouge")):
        d, bonus, fl = res[sens]
        if all(d.values()):
            conseil = f"Signal {nom} détecté."
            if bonus:
                conseil = f"⭐ Signal {nom} renforcé (H4 dans le même sens)."
            details = {**d, f"H4 {fl}": bonus}
            base = {"signal": nom, "couleur": couleur, "bonus": bonus,
                    "conseil": conseil, "details": details, "timing": timing}
            break
    if base is None:
        meilleur = max(("haussier", "baissier"), key=lambda s: sum(res[s][0].values()))
        d, bonus, fl = res[meilleur]
        nom = "ACHAT" if meilleur == "haussier" else "VENTE"
        manque_c = [k for k, v in d.items() if not v]
        ok = len(d) - len(manque_c)
        conseil = f"⏸️ Attendre : {ok}/{len(d)} conditions {nom}."
        base = {"signal": "ATTENDRE", "couleur": "jaune", "bonus": False,
                "conseil": conseil, "details": {**d, f"H4 {fl}": bonus}, "timing": timing}
    base["alertes"] = []
    base["generale"] = generale
    if ferme:
        return {**base, "signal": "FERMÉ", "couleur": "gris", "bonus": False,
                "conseil": "🌙 Marché fermé."}
    return base


def alertes_signal(cond, tfs, niv, sltp, bt, mm_h1):
    sig = cond["signal"]
    if sig not in ("ACHAT", "VENTE"):
        return []
    al = []
    achat = sig == "ACHAT"
    m5, m15 = tfs["M5"], tfs["M15"]
    prix = m5["prix"]
    if achat and niv and niv["res_brut"] is not None and niv["res_brut"] - prix < mm_h1:
        al.append({"txt": f"Résistance proche ({niv['res']}).", "cls": "jaune"})
    if not achat and niv and niv["sup_brut"] is not None and prix - niv["sup_brut"] < mm_h1:
        al.append({"txt": f"Support proche ({niv['sup']}).", "cls": "jaune"})
    if m5["grande"]:
        al.append({"txt": f"Grosse bougie M5 (×{m5['taille_x']}).", "cls": "jaune"})
    if m5["rsi"] > 80:
        al.append({"txt": f"RSI M5 suracheté ({m5['rsi']}).", "cls": "jaune"})
    if m5["rsi"] < 20:
        al.append({"txt": f"RSI M5 survendu ({m5['rsi']}).", "cls": "jaune"})
    if sltp:
        cote = sltp["achat" if achat else "vente"]
        if cote is None:
            al.append({"txt": "Pas d'objectif clair.", "cls": "jaune"})
        elif cote["ok"]:
            al.append({"txt": f"Ratio {cote['ratio']}.", "cls": "vert"})
        else:
            al.append({"txt": f"Ratio {cote['ratio']} sous 1,5.", "cls": "jaune"})
    return al


def construire_guide(pair, cond, tfs, sltp):
    sig = cond["signal"]
    if sig not in ("ACHAT", "VENTE") or not sltp:
        return None
    achat = sig == "ACHAT"
    cote = sltp["achat" if achat else "vente"]
    prix = tfs["M5"]["prix"]
    nxt = (int(time.time() // 300) + 1) * 300
    timing = cond.get("timing")
    feu = "vert"
    lot_niveau = "PLEIN"
    lot_txt = "Lot plein autorisé."
    if timing:
        if timing["niveau"] == "EN COURS":
            feu = "orange"
            lot_niveau = "DEMI"
            lot_txt = "Demi-lot conseillé."
        elif timing["niveau"] == "TARDIF":
            feu = "orange"
            lot_niveau = "ATTENDRE"
            lot_txt = "N'ENTRE PAS : mouvement trop avancé."
    g = {"sens": sig, "feu": feu, "heure": datetime.fromtimestamp(nxt, BENIN).strftime("%H:%M"),
         "prix": fp(pair, prix), "contrat": CONTRAT.get(pair), "stop": None,
         "timing": timing, "lot_niveau": lot_niveau, "lot_txt": lot_txt,
         "rsi_m5": tfs["M5"]["rsi"]}
    if cote:
        g.update({"stop": cote["stop"], "objectif": cote["objectif"], "ratio": cote["ratio"],
                  "dist": f"{abs(prix - cote['stop_brut']):.8f}",
                  "dist_obj": f"{abs(cote['obj_brut'] - prix):.8f}",
                  "moitie": fp(pair, prix + (cote["obj_brut"] - prix) / 2)})
    else:
        g["feu"] = "orange"
    return g


# ───────────── ANALYSE PARALLÉLISÉE ─────────────

def charger_tf_pour_paire(pair, source, tf, interval, range_, gran):
    try:
        if source == "yahoo":
            brut = recuperer_bougies_yahoo(YAHOO_PAIRS[pair], interval, range_)
            if tf == "H4":
                brut = regrouper(brut, gran)
        else:
            brut = recuperer_bougies_deriv(DERIV_PAIRS[pair], gran, DERIV_COUNT[tf])
        return tf, bougies_fermees(brut, gran)
    except Exception as e:
        print(f"Erreur {pair} {tf}: {e}")
        return tf, []


def analyser_paire(pair, source, historique=None, autres_cond=None):
    series = {}
    taches = [(tf, *TIMEFRAMES[tf]) for tf in TIMEFRAMES]
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = [ex.submit(charger_tf_pour_paire, pair, source, *t) for t in taches]
        for f in as_completed(futures):
            try:
                tf, bougies = f.result()
                series[tf] = bougies
            except Exception as e:
                print(f"Erreur future {pair}: {e}")
    resultats = {}
    for tf, bougies in series.items():
        r = analyser(tf, bougies)
        if r:
            resultats[tf] = r
    fr = fraicheur(series.get("M5"))
    stats = calc_stats(pair, series.get("H1", []), resultats)
    generale = calc_tendance_generale(resultats)
    cond = conditions_paire(resultats, fr["etat"] == "ferme", generale)
    prix = fp(pair, resultats["M5"]["prix"]) if "M5" in resultats else "—"
    niv = sltp = bt = heures = guide = None
    h1, m15 = series.get("H1", []), series.get("M15", [])
    if "M5" in resultats and len(h1) >= 30 and len(m15) >= 30:
        p = resultats["M5"]["prix"]
        niv = calc_niveaux(pair, p, m15, h1)
        sltp = calc_sltp(pair, p, m15)
        bt = backtest(series, pair)
        cond["alertes"] = alertes_signal(cond, resultats, niv, sltp, bt, mouvement_moyen(h1, 24))
        guide = construire_guide(pair, cond, resultats, sltp)
    return {"tfs": resultats, "stats": stats, "fraicheur": fr, "cond": cond, "prix": prix,
            "niv": niv, "sltp": sltp, "bt": bt, "heures": heures, "guide": guide, "generale": generale}


def analyser_securise(pair, source, historique=None, autres_cond=None):
    try:
        return analyser_paire(pair, source, historique, autres_cond)
    except Exception as e:
        print(f"Erreur {pair}: {e}")
        return {"tfs": {}, "stats": None, "fraicheur": {"texte": "Erreur", "etat": "ferme"},
                "cond": conditions_paire({}), "prix": "—", "niv": None, "sltp": None,
                "bt": None, "heures": None, "guide": None, "generale": None}


def analyser_tout():
    taches = [(p, "yahoo") for p in YAHOO_PAIRS] + [(p, "deriv") for p in DERIV_PAIRS]
    res = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        for (pair, _), data in zip(taches, ex.map(lambda t: analyser_securise(*t), taches)):
            res[pair] = data
    autres_cond = {p: res[p]["cond"] for p in res}
    historique = session.get("historique", [])
    for pair, source in taches:
        try:
            res[pair] = analyser_paire(pair, source, historique, autres_cond)
        except Exception as e:
            print(f"Erreur passe 2 {pair}: {e}")
    session_act = session_actuelle()
    journal = session.get("journal", [])
    maintenant = time.time()
    maintenant_txt = datetime.now(BENIN).strftime("%H:%M")
    for pair, d in res.items():
        conseil = generer_conseil(pair, d, session_act, autres_cond, historique)
        d["conseil_bot"] = conseil
        dernier = None
        for j in reversed(journal):
            if j["pair"] == pair:
                dernier = j
                break
        nouveau_verdict = conseil["verdict"]
        if dernier is None or dernier["verdict"] != nouveau_verdict or (maintenant - dernier["t"]) > 3600:
            journal.append({"t": maintenant, "heure": maintenant_txt, "pair": pair,
                            "verdict": nouveau_verdict, "couleur": conseil["couleur"],
                            "message": conseil["message"], "action": conseil["action"],
                            "raisons": conseil["raisons"], "prix": d["prix"]})
    journal = journal[-JOURNAL_MAX:]
    session["journal"] = journal
    for pair, d in res.items():
        cond = d["cond"]
        if cond["signal"] in ("ACHAT", "VENTE"):
            historique.append({"pair": pair, "sens": cond["signal"], "t": maintenant, "prix": d["prix"]})
    historique = historique[-50:]
    session["historique"] = historique
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
    journal = session.get("journal", [])
    session_act = session_actuelle()
    return render_template("dashboard.html", paires=paires, now=datetime.now(BENIN).strftime("%H:%M"),
                           journal=journal[-30:], session_act=session_act)


@app.route("/ping")
def ping():
    return "ok", 200


@app.route("/cron")
def cron():
    analyser_tout()
    return "ok", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
