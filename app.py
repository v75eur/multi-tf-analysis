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
            gains.append(diff); pertes.append(0.0)
        elif diff < 0:
            gains.append(0.0); pertes.append(abs(diff))
        else:
            gains.append(0.0); pertes.append(0.0)
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
    return round(100 - (100 / (1 + rs)), 1)


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
# DÉCLENCHEUR : SIGNAUX VISUELS DE BOUGIES
# Basé sur les 5 dernières bougies M5
# ═══════════════════════════════════════════════════════

def detecter_signal_bougies(bougies):
    """Détecte un signal d'entrée par les bougies.
    Retourne: (sens, type_signal, fiabilite, description) ou None.
    Priorité: grande bougie > englobante > 3 bougies successives."""
    if len(bougies) < 5:
        return None

    # Taille moyenne des 20 dernières bougies (référence)
    recent = bougies[-21:-1]
    moy_rng = float(np.mean([b["high"] - b["low"] for b in recent])) if recent else 0.0

    # On regarde les 5 dernières bougies
    d = bougies[-5:]

    # PRIORITÉ 1 : Grande bougie (×2 la moyenne) dans les 2 dernières
    for k in range(0, 2):
        b = d[-(k + 1)]
        rng = b["high"] - b["low"]
        if moy_rng > 0 and rng / moy_rng >= 2:
            if b["close"] > b["open"]:
                return {"sens": "haussier", "type": "Grande bougie verte",
                        "fiabilite": 75, "taille": round(rng / moy_rng, 1),
                        "il_y_a": k, "desc": f"Grande bougie verte (×{round(rng/moy_rng,1)}) — signal fort haussier"}
            elif b["close"] < b["open"]:
                return {"sens": "baissier", "type": "Grande bougie rouge",
                        "fiabilite": 75, "taille": round(rng / moy_rng, 1),
                        "il_y_a": k, "desc": f"Grande bougie rouge (×{round(rng/moy_rng,1)}) — signal fort baissier"}

    # PRIORITÉ 2 : Englobante (avale la précédente) dans les 2 dernières
    for k in range(0, 2):
        idx = len(d) - 1 - k
        if idx < 1:
            continue
        b = d[idx]
        p = d[idx - 1]
        # Englobante haussière : verte qui avale une rouge
        if p["close"] < p["open"] and b["close"] > b["open"]:
            if b["open"] <= p["close"] and b["close"] >= p["open"]:
                return {"sens": "haussier", "type": "Englobante haussière",
                        "fiabilite": 63, "taille": 0, "il_y_a": k,
                        "desc": "Englobante haussière — la bougie verte avale la rouge précédente"}
        # Englobante baissière : rouge qui avale une verte
        if p["close"] > p["open"] and b["close"] < b["open"]:
            if b["open"] >= p["close"] and b["close"] <= p["open"]:
                return {"sens": "baissier", "type": "Englobante baissière",
                        "fiabilite": 63, "taille": 0, "il_y_a": k,
                        "desc": "Englobante baissière — la bougie rouge avale la verte précédente"}

    # PRIORITÉ 3 : 3 bougies successives dans le même sens
    trois = d[-3:]
    if len(trois) == 3:
        # 3 vertes d'affilée
        if all(b["close"] > b["open"] for b in trois):
            # Vérifier que chaque bougie ferme plus haut que la précédente
            if trois[0]["close"] < trois[1]["close"] < trois[2]["close"]:
                return {"sens": "haussier", "type": "3 bougies vertes",
                        "fiabilite": 78, "taille": 0, "il_y_a": 0,
                        "desc": "3 soldats blancs — 3 bougies vertes successives de plus en plus hautes"}
        # 3 rouges d'affilée
        if all(b["close"] < b["open"] for b in trois):
            if trois[0]["close"] > trois[1]["close"] > trois[2]["close"]:
                return {"sens": "baissier", "type": "3 bougies rouges",
                        "fiabilite": 78, "taille": 0, "il_y_a": 0,
                        "desc": "3 corbeaux noirs — 3 bougies rouges successives de plus en plus basses"}

    return None


def calc_motifs_classiques(bougies):
    """Mèches de rejet, englobante (info), grande bougie."""
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
    signal_bougies = detecter_signal_bougies(bougies) if tf == "M5" else None

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
        "signal_bougies": signal_bougies,
        "cassure": calc_cassure(bougies),
        "prix": clotures[-1]
    }


def calc_timing_entree(signal_bougies):
    """Le timing est basé sur le signal des bougies."""
    if not signal_bougies:
        return None
    il_y_a = signal_bougies.get("il_y_a", 0)
    if il_y_a <= 1:
        return {"niveau": "FRAIS", "couleur": "vert", "emoji": "🔥",
                "txt": f"Signal bougies très récent ({il_y_a} bougie(s)). Tu entres au début du mouvement.",
                "lot_conseil": "LOT PLEIN", "lot_ratio": 1.0}
    if il_y_a <= 2:
        return {"niveau": "EN COURS", "couleur": "jaune", "emoji": "⏳",
                "txt": f"Signal bougies il y a {il_y_a} bougies. Le mouvement est lancé.",
                "lot_conseil": "DEMI-LOT", "lot_ratio": 0.5}
    return {"niveau": "TARDIF", "couleur": "rouge", "emoji": "❌",
            "txt": f"Signal bougies il y a {il_y_a} bougies. Trop tardif.",
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
    votes_detail = []
    for tf in ORDRE_TF:
        if tf in tfs:
            p = POIDS_TF.get(tf, 1)
            v = 1 if tfs[tf]["tendance"] == "haussier" else (-1 if tfs[tf]["tendance"] == "baissier" else 0)
            votes_detail.append({"tf": tf, "vote": v, "poids": p, "contribution": v * p,
                                 "rsi": tfs[tf]["rsi"]})
    return {"rsi_gen": rsi_gen, "ach_gen": rsi_gen, "ven_gen": round(100 - rsi_gen, 1),
            "score_pct": score_pct, "score_brut": round(score_pct_brut, 1),
            "bonus": bonus,
            "tendance": tendance, "couleur": couleur,
            "align_sens": align_sens, "align_n": align_n,
            "align_detail": alignement["detail"],
            "votes_detail": votes_detail}


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


# ───────────── CONDITIONS D'ENTRÉE ─────────────

def conditions_paire(tfs, ferme=False, generale=None):
    """Conditions basées sur les SIGNAUX BOUGIES maintenant."""
    manque = [k for k in ["H1", "M30", "M15", "M5"] if k not in tfs]
    if manque:
        return {"signal": "?", "couleur": "jaune", "bonus": False,
                "conseil": "Données manquantes : " + ", ".join(manque),
                "details": {}, "alertes": [], "generale": generale, "timing": None,
                "signal_bougies": None}
    h1, m30, m15, m5 = tfs["H1"], tfs["M30"], tfs["M15"], tfs["M5"]
    h4 = tfs.get("H4")
    sig_b = m5.get("signal_bougies")

    # Pas de signal bougies → on ne peut pas entrer
    if not sig_b:
        conseil = "⏸️ Aucun signal de bougie visible sur M5. Attends 3 bougies successives, une grande bougie, ou une englobante."
        return {"signal": "ATTENDRE", "couleur": "jaune", "bonus": False,
                "conseil": conseil, "details": {}, "alertes": [],
                "generale": generale, "timing": None, "signal_bougies": None}

    sens = sig_b["sens"]
    achat = sens == "haussier"

    # Vérifier l'alignement des 3 TF avec le sens du signal bougies
    def verifier(s):
        if s == "haussier":
            fl, lab = "↑", "RSI>50"
        else:
            fl, lab = "↓", "RSI<50"
        d = {}
        for nom, t in (("H1", h1), ("M30", m30), ("M15", m15)):
            d[f"{nom} {fl}"] = t["tendance"] == s
        for nom, t in (("H1", h1), ("M30", m30), ("M15", m15)):
            if s == "haussier":
                d[f"{lab} {nom}"] = t["rsi"] > 50
            else:
                d[f"{lab} {nom}"] = t["rsi"] < 50
        # Signal bougies dans le bon sens
        d[f"M5 signal {fl}"] = m5["signal_bougies"] is not None and m5["signal_bougies"]["sens"] == s
        bonus = bool(h4 and h4["tendance"] == s)
        return d, bonus, fl

    d, bonus, fl = verifier(sens)
    timing = calc_timing_entree(sig_b)
    nom = "ACHAT" if achat else "VENTE"
    couleur = "vert" if achat else "rouge"

    if all(d.values()):
        conseil = f"✅ Signal {nom} par bougies : {sig_b['type']}. H1, M30, M15 alignés {sens}s + RSI confirme."
        if bonus:
            conseil = f"⭐ Signal {nom} renforcé : {sig_b['type']}. H4 aussi dans le même sens."
        return {"signal": nom, "couleur": couleur, "bonus": bonus,
                "conseil": conseil, "details": {**d, f"H4 {fl}": bonus},
                "timing": timing, "signal_bougies": sig_b}

    # Signal bougies existe mais les 3 TF ne sont pas alignés
    manque_c = [k for k, v in d.items() if not v]
    ok = len(d) - len(manque_c)
    conseil = f"⚠️ Signal {nom} par bougies MAIS {ok}/{len(d)} conditions. Manque : {', '.join(manque_c[:4])}."
    return {"signal": "ATTENDRE", "couleur": "jaune", "bonus": False,
            "conseil": conseil, "details": {**d, f"H4 {fl}": bonus},
            "timing": timing, "signal_bougies": sig_b}


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
            lot_txt = "N'ENTRE PAS : signal trop tardif."
    g = {"sens": sig, "feu": feu, "heure": datetime.fromtimestamp(nxt, BENIN).strftime("%H:%M"),
         "prix": fp(pair, prix), "contrat": CONTRAT.get(pair), "stop": None,
         "timing": timing, "lot_niveau": lot_niveau, "lot_txt": lot_txt,
         "signal_bougies": cond.get("signal_bougies"),
         "rsi_m5": tfs["M5"]["rsi"]}
    if cote:
        g.update({"stop": cote["stop"], "objectif": cote["objectif"], "ratio": cote["ratio"],
                  "dist": f"{abs(prix - cote['stop_brut']):.8f}",
                  "dist_obj": f"{abs(cote['obj_brut'] - prix):.8f}",
                  "moitie": fp(pair, prix + (cote["obj_brut"] - prix) / 2)})
    else:
        g["feu"] = "orange"
    return g


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
        # Vérifier le signal bougies sur les 5 dernières bougies M5
        sig_b = detecter_signal_bougies(m5[max(0, i - 4):i + 1])
        if not sig_b:
            prev = None
            continue
        s_sens = sig_b["sens"]
        s_sign = 1 if s_sens == "haussier" else -1
        # Vérifier alignement des 3 TF
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
        if not ok or etats != {s_sign}:
            prev = None
            continue
        sig = "ACHAT" if s_sign == 1 else "VENTE"
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
                                res_r = -1.0; sim["sl"] += 1; break
                            if bj["high"] >= ob:
                                res_r = cote["ratio"]; sim["tp"] += 1; break
                        else:
                            if bj["high"] >= st:
                                res_r = -1.0; sim["sl"] += 1; break
                            if bj["low"] <= ob:
                                res_r = cote["ratio"]; sim["tp"] += 1; break
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
    niv = sltp = bt = guide = None
    h1, m15 = series.get("H1", []), series.get("M15", [])
    if "M5" in resultats and len(h1) >= 30 and len(m15) >= 30:
        p = resultats["M5"]["prix"]
        niv = calc_niveaux(pair, p, m15, h1)
        sltp = calc_sltp(pair, p, m15)
        bt = backtest(series, pair)
        cond["alertes"] = alertes_signal(cond, resultats, niv, sltp, bt, mouvement_moyen(h1, 24))
        guide = construire_guide(pair, cond, resultats, sltp)
    return {"tfs": resultats, "stats": stats, "fraicheur": fr, "cond": cond, "prix": prix,
            "niv": niv, "sltp": sltp, "bt": bt, "guide": guide, "generale": generale}


def analyser_securise(pair, source, historique=None, autres_cond=None):
    try:
        return analyser_paire(pair, source, historique, autres_cond)
    except Exception as e:
        print(f"Erreur {pair}: {e}")
        return {"tfs": {}, "stats": None, "fraicheur": {"texte": "Erreur", "etat": "ferme"},
                "cond": conditions_paire({}), "prix": "—", "niv": None, "sltp": None,
                "bt": None, "guide": None, "generale": None}


def generer_conseil(pair, d, session_act, autres_cond, historique):
    """Conseil basé sur les signaux bougies."""
    tfs = d["tfs"]
    cond = d["cond"]
    gen = d["generale"]
    sig_b = cond.get("signal_bougies")
    sig = cond["signal"]

    if not tfs or not gen:
        return {"verdict": "PAS DE DONNÉES", "couleur": "gris", "action": "ATTENDRE",
                "message": f"Pas assez de données sur {pair}.", "raisons": ["Données insuffisantes."]}

    # Cas 1 : signal bougies + alignement OK
    if sig in ("ACHAT", "VENTE"):
        timing = cond.get("timing")
        raisons = []
        achat = sig == "ACHAT"
        if session_act["qualite"] == "faible":
            raisons.append(f"Session {session_act['nom']} : peu de volume.")
        if timing and timing["niveau"] == "EN COURS":
            raisons.append(f"Signal bougies il y a {sig_b.get('il_y_a', 0)} bougies (EN COURS).")
        elif timing and timing["niveau"] == "TARDIF":
            raisons.append(f"Signal bougies il y a {sig_b.get('il_y_a', 0)} bougies (TARDIF).")
        h4 = tfs.get("H4")
        contre = "baissier" if achat else "haussier"
        if h4 and h4["tendance"] == contre:
            raisons.append(f"H4 est {contre} (contre ton signal).")
        rsi_m5 = tfs["M5"]["rsi"]
        if achat and rsi_m5 > 80:
            raisons.append(f"RSI M5 suracheté ({rsi_m5}).")
        if not achat and rsi_m5 < 20:
            raisons.append(f"RSI M5 survendu ({rsi_m5}).")
        sltp = d["sltp"]
        cote = sltp["achat" if achat else "vente"] if sltp else None
        if cote is None:
            raisons.append("Pas d'objectif clair.")
        elif not cote["ok"]:
            raisons.append(f"Ratio faible ({cote['ratio']}).")
        bloquants = [r for r in raisons if "contre ton signal" in r or "pas d'objectif" in r.lower() or "TARDIF" in r]
        avertissements = [r for r in raisons if r not in bloquants]

        type_sig = sig_b["type"] if sig_b else "signal"
        fiab = sig_b["fiabilite"] if sig_b else 0

        if bloquants:
            return {"verdict": "N'ENTRE PAS", "couleur": "rouge", "action": "PASSER CE TRADE",
                    "message": f"Signal {sig} par bougies ({type_sig}) MAIS quelque chose bloque.",
                    "raisons": raisons}
        if avertissements:
            return {"verdict": "PRUDENCE — DEMI-LOT", "couleur": "jaune", "action": "RÉDUIRE LE LOT",
                    "message": f"Signal {sig} par bougies ({type_sig}, fiabilité {fiab}%). Points de vigilance.",
                    "raisons": raisons}
        lot_msg = "lot plein" if not timing or timing["niveau"] == "FRAIS" else "demi-lot"
        return {"verdict": "ENTRE MAINTENANT", "couleur": "vert",
                "action": f"{sig} AU MARCHÉ — {lot_msg.upper()}",
                "message": f"Signal {sig} par bougies : {type_sig} (fiabilité {fiab}%). H1+M30+M15 alignés. Entre à la clôture M5 avec le {lot_msg}.",
                "raisons": raisons or [f"Signal bougies valide : {type_sig}."]}

    # Cas 2 : signal bougies mais pas aligné
    if sig_b:
        return {"verdict": "ATTENDRE", "couleur": "gris", "action": "NE RIEN FAIRE",
                "message": cond["conseil"], "raisons": ["Signal bougies présent mais alignement des TF incomplet."]}

    # Cas 3 : pas de signal bougies
    return {"verdict": "ATTENDRE", "couleur": "gris", "action": "NE RIEN FAIRE",
            "message": "Aucun signal de bougie visible sur M5. Attends 3 bougies successives, une grande bougie, ou une englobante dans le sens de la tendance.",
            "raisons": ["Pas de signal bougies."]}


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
