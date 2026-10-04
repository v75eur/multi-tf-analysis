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
CONTRAT = {"EURUSD": 100000, "GBPUSD": 100000, "XAUUSD": 100, "V75": None}
TIMEFRAMES = {
    "H4": ("1h", "60d", 14400),
    "H1": ("1h", "60d", 3600),
    "M30": ("30m", "30d", 1800),
    "M15": ("15m", "15d", 900),
    "M5": ("5m", "7d", 300),
}
DERIV_COUNT = {"H4": 300, "H1": 300, "M30": 300, "M15": 400, "M5": 1000}
PERIODE = 10
EMA_COURT = 5
EMA_LONG = 25
DERIV_WS = "wss://api.derivws.com/trading/v1/options/ws/public"
CACHE_SEC = {"1h": 300, "30m": 180, "15m": 120, "5m": 60}
_cache_yahoo = {}

# Pondération : petits TF plus lourds (mouvement récent)
POIDS_TF = {"M5": 5, "M15": 4, "M30": 3, "H1": 2, "H4": 1}
ORDRE_TF = ["M5", "M15", "M30", "H1", "H4"]

# ───────────── SESSIONS DE TRADING (heure du Bénin) ─────────────
# Asie 01h-09h : éviter (peu de volume sur EURUSD/GBPUSD)
# Londres 09h-18h : optimale pour EURUSD/GBPUSD
# New York 15h-23h : optimale pour XAUUSD
# Chevauchement Londres/NY 15h-18h : meilleure fenêtre
SESSIONS = {
    "asie":     {"debut": 1,  "fin": 9,  "qualite": "faible",  "txt": "Asie (peu de volume, signaux peu fiables)"},
    "londres":  {"debut": 9,  "fin": 18, "qualite": "forte",   "txt": "Londres (session active, bonne fiabilité)"},
    "newyork":  {"debut": 15, "fin": 23, "qualite": "forte",   "txt": "New York (session active, bonne fiabilité)"},
    "nuit":     {"debut": 23, "fin": 1,  "qualite": "faible",  "txt": "Nuit (marché calme)"},
}

# Corrélations connues (si deux paires bougent ensemble)
CORRELATIONS = {
    "EURUSD": ["GBPUSD"],   # EUR et GBP très corrélés
    "GBPUSD": ["EURUSD"],
    "XAUUSD": [],           # Or souvent inverse du dollar
    "V75": [],
}

# Seuils de fiabilité
SEUIL_VOLATILITE_MIN = 0.05   # % mouvement moyen H1 minimum (sinon marché plat)
SEUIL_AGITATION_MAX = 70      # % agitation max (sinon marché trop indécis)
DUREE_SIGNAL_MIN = 30         # minutes avant qu'un nouveau signal identique soit valide


def logged():
    return session.get("admin") is True


def fp(pair, x):
    return f"{x:.{DECIMALES.get(pair, 5)}f}"


# ───────────── RÉCUPÉRATION DES BOUGIES ─────────────

def recuperer_bougies_deriv(symbol, granularity, count=300):
    try:
        ws = websocket.create_connection(DERIV_WS, timeout=15)
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
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
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


# ───────────── TENDANCE GÉNÉRALE ─────────────

def calc_tendance_generale(tfs):
    if not tfs:
        return None
    poids_total = 0
    somme_ach = 0.0
    somme_ven = 0.0
    for tf, data in tfs.items():
        p = POIDS_TF.get(tf, 1)
        poids_total += p
        somme_ach += data["pct_acheteurs"] * p
        somme_ven += data["pct_vendeurs"] * p
    if poids_total == 0:
        return None
    ach_gen = round(somme_ach / poids_total, 1)
    ven_gen = round(somme_ven / poids_total, 1)

    score = 0
    score_max = 0
    for tf, data in tfs.items():
        p = POIDS_TF.get(tf, 1)
        score_max += p
        if data["tendance"] == "haussier":
            score += p
        elif data["tendance"] == "baissier":
            score -= p
    score_pct = round(score / score_max * 100, 1) if score_max else 0.0

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
                                "txt": "✅ M15+M30+H1 haussiers + croisement EMA M5 haussier récent (après clôture) → ACHAT confirmé."}
        elif m15["tendance"] == m30["tendance"] == h1["tendance"] == "baissier":
            if m5["ema_signal"] == "baissier" and (m5["ema_depuis"] or 99) <= 3:
                confirmation = {"sens": "VENTE", "couleur": "rouge",
                                "txt": "✅ M15+M30+H1 baissiers + croisement EMA M5 baissier récent (après clôture) → VENTE confirmée."}

    return {
        "ach_gen": ach_gen, "ven_gen": ven_gen,
        "score_pct": score_pct, "tendance": tendance, "couleur": couleur,
        "align_sens": align_sens, "align_n": align_n,
        "align_detail": alignement["detail"],
        "confirmation": confirmation,
    }


# ───────────── FILTRES DE FIABILITÉ (NOUVEAU) ─────────────

def session_actuelle():
    """Retourne la session en cours (heure du Bénin)."""
    h = datetime.now(BENIN).hour
    if 1 <= h < 9:
        s = SESSIONS["asie"]
    elif 9 <= h < 15:
        s = SESSIONS["londres"]
    elif 15 <= h < 18:
        # Chevauchement Londres/NY = meilleure fenêtre
        s = {"qualite": "excellente", "txt": "Chevauchement Londres + New York (meilleure fenêtre de la journée)"}
    elif 18 <= h < 23:
        s = SESSIONS["newyork"]
    else:
        s = SESSIONS["nuit"]
    return {"heure": h, "qualite": s["qualite"], "txt": s["txt"]}


def filtre_volatilite(tfs):
    """Vérifie que le marché n'est pas plat."""
    h1 = tfs.get("H1")
    if not h1:
        return {"ok": False, "txt": "Pas de données H1 pour évaluer la volatilité.", "cls": "jaune"}
    mvt = mouvement_moyen([{"high": h1["prix"], "low": h1["prix"], "close": h1["prix"]}], 1)
    # On prend le % de mouvement H1 depuis les stats
    # Utilise directement le pct_tendance comme proxy + agitation
    agitation = h1.get("pct_agitation", 0)
    if agitation > SEUIL_AGITATION_MAX:
        return {"ok": False, "cls": "jaune",
                "txt": f"⚠️ Marché trop agité (agitation H1 = {agitation}%) : les bougies font des allers-retours, le signal est peu fiable. Attends un marché plus directionnel."}
    return {"ok": True, "cls": "vert",
            "txt": f"✅ Volatilité correcte (agitation H1 = {agitation}%)."}


def filtre_tendance_superieure(tfs, sens_signal):
    """Vérifie que H4 et H1 ne contredisent pas le signal.
    Si H4 + H1 sont contraires au signal, c'est un pullback = piège."""
    if not sens_signal or sens_signal not in ("haussier", "baissier"):
        return {"ok": True, "cls": "", "txt": ""}
    h4 = tfs.get("H4")
    h1 = tfs.get("H1")
    if not h4 or not h1:
        return {"ok": True, "cls": "", "txt": "Pas assez de données H4/H1 pour vérifier la tendance supérieure."}

    h4_sens = h4["tendance"]
    h1_sens = h1["tendance"]
    contre = "baissier" if sens_signal == "haussier" else "haussier"

    if h4_sens == contre and h1_sens == contre:
        return {"ok": False, "cls": "rouge",
                "txt": f"🚫 H4 ET H1 sont {contre}s, ton signal est {sens_signal}. C'est un pullback dans une tendance inverse : NE PRENDS PAS ce trade, tu irais contre la tendance de fond."}
    if h4_sens == contre:
        return {"ok": False, "cls": "jaune",
                "txt": f"⚠️ H4 est {contre} (contre ton signal {sens_signal}). Le signal peut être un simple rebond dans une tendance inverse. Réduis ton lot de moitié ou attends."}
    if h1_sens == contre:
        return {"ok": False, "cls": "jaune",
                "txt": f"⚠️ H1 est {contre} (contre ton signal {sens_signal}). Attention, la tendance horaire n'est pas encore retournée. Attends que H1 confirme."}
    return {"ok": True, "cls": "vert",
            "txt": f"✅ H4 et H1 vont dans le sens du signal ({sens_signal}). Tu trades avec la tendance de fond."}


def filtre_session():
    """Vérifie qu'on est dans une bonne session de trading."""
    s = session_actuelle()
    if s["qualite"] == "excellente":
        return {"ok": True, "cls": "vert", "txt": f"✅ {s['txt']}."}
    if s["qualite"] == "forte":
        return {"ok": True, "cls": "vert", "txt": f"✅ {s['txt']}."}
    return {"ok": False, "cls": "jaune",
            "txt": f"⚠️ {s['txt']}. Les signaux en dehors de Londres/New York sont souvent faux à cause du manque de volume. Évite de trader maintenant."}


def filtre_duplication(pair, sens, historique):
    """Vérifie qu'un signal identique n'a pas déjà été donné récemment."""
    if not historique or not sens:
        return {"ok": True, "cls": "", "txt": ""}
    maintenant = time.time()
    for h in reversed(historique):
        if h["pair"] == pair and h["sens"] == sens:
            age_min = (maintenant - h["t"]) / 60
            if age_min < DUREE_SIGNAL_MIN:
                return {"ok": False, "cls": "jaune",
                        "txt": f"⚠️ Un signal {sens} identique sur {pair} a déjà été donné il y a {int(age_min)} min. Ce n'est probablement pas un nouveau signal, c'est le même mouvement. Attends au moins {DUREE_SIGNAL_MIN} min."}
            break
    return {"ok": True, "cls": "", "txt": ""}


def filtre_correlation(pair, sens, autres_cond):
    """Vérifie que deux paires corrélées n'ont pas le même signal en même temps."""
    if not sens or sens not in ("ACHAT", "VENTE") or not autres_cond:
        return {"ok": True, "cls": "", "txt": ""}
    correlees = CORRELATIONS.get(pair, [])
    if not correlees:
        return {"ok": True, "cls": "", "txt": ""}
    conflits = []
    for c in correlees:
        if c in autres_cond and autres_cond[c].get("signal") == sens:
            conflits.append(c)
    if conflits:
        return {"ok": False, "cls": "jaune",
                "txt": f"⚠️ {', '.join(conflits)} a aussi un signal {sens} en même temps. Ces paires sont corrélées : prendre les deux = doubler ton risque. Choisis-en une seule."}
    return {"ok": True, "cls": "", "txt": ""}


def filtre_global(tfs, cond_base, pair, sens_signal, historique, autres_cond):
    """Applique tous les filtres et retourne un verdict global."""
    filtres = []

    # 1. Session
    f_session = filtre_session()
    filtres.append({"nom": "Session", **f_session})

    # 2. Volatilité
    f_vol = filtre_volatilite(tfs)
    filtres.append({"nom": "Volatilité", **f_vol})

    # 3. Tendance supérieure
    if sens_signal:
        f_tend = filtre_tendance_superieure(tfs, sens_signal)
        filtres.append({"nom": "Tendance H4/H1", **f_tend})

    # 4. Duplication
    f_dup = filtre_duplication(pair, sens_signal, historique)
    if f_dup["txt"]:
        filtres.append({"nom": "Signal récent", **f_dup})

    # 5. Corrélation
    f_corr = filtre_correlation(pair, sens_signal, autres_cond)
    if f_corr["txt"]:
        filtres.append({"nom": "Corrélation", **f_corr})

    # Verdict global
    bloquants = [f for f in filtres if f["ok"] is False and f["cls"] == "rouge"]
    avertissements = [f for f in filtres if f["ok"] is False and f["cls"] == "jaune"]

    if bloquants:
        verdict = "BLOQUÉ"
        couleur = "rouge"
        conseil = "🚫 NE PRENDS PAS ce trade. " + " ".join(f["txt"] for f in bloquants)
    elif avertissements:
        verdict = "PRUDENCE"
        couleur = "jaune"
        conseil = "⚠️ Trade possible mais risqué. " + " ".join(f["txt"] for f in avertissements)
    else:
        verdict = "FEU VERT"
        couleur = "vert"
        conseil = "✅ Tous les filtres sont au vert. Tu peux trader en respectant le guide."

    return {"filtres": filtres, "verdict": verdict, "couleur": couleur, "conseil": conseil}


# ───────────── STATISTIQUES GÉNÉRALES ─────────────

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


# ───────────── MEILLEURES HEURES ─────────────

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


# ───────────── TEST SUR LE PASSÉ ─────────────

def tendance_simple(cl):
    x = np.arange(len(cl))
    p = np.polyfit(x, np.array(cl), 1)[0]
    return 1 if p > 0 else -1 if p < 0 else 0


def backtest(series, pair="", horizon=12, test=1000, max_trade=48):
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
            tend = tendance_simple([b["close"] for b in sl[-PERIODE:]])
            d = sl[-PERIODE:]
            v = sum(1 for b in d if b["close"] > b["open"])
            r = sum(1 for b in d if b["close"] < b["open"])
            if tend > 0 and v > r:
                etats.add(1)
            elif tend < 0 and r > v:
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
        verdict = "Trop peu de signaux pour conclure : si tu vois moins de 8 signaux, alors le résultat peut être dû au hasard."
        couleur = ""
    elif pct >= 55:
        verdict = "Plutôt bon sur cette période : si la réussite reste au-dessus de 55 %, alors le signal a un petit avantage."
        couleur = "vert"
    elif pct >= 45:
        verdict = "Proche du hasard (50 %) : si c'est le cas, alors le signal seul ne suffit pas. Regarde le ratio gain/risque."
        couleur = ""
    else:
        verdict = "Mauvais sur cette période : si le signal perd plus qu'il ne gagne, alors évite-le sur cette devise."
        couleur = "rouge"

    sim_n = len(sim["r"])
    sim_r = round(float(np.mean(sim["r"])), 2) if sim_n else None
    if sim_n < 8:
        sim_verdict = "Trop peu de trades simulés pour conclure."
        sim_couleur = ""
    elif sim_r > 0.15:
        sim_verdict = "Avec stop et objectif, le signal gagne en moyenne sur cette période."
        sim_couleur = "vert"
    elif sim_r < -0.1:
        sim_verdict = "Avec stop et objectif, le signal perd en moyenne : évite-le sur cette devise."
        sim_couleur = "rouge"
    else:
        sim_verdict = "Avec stop et objectif, le résultat est proche de zéro : pas d'avantage net."
        sim_couleur = ""

    heures = round((m5[-1]["t"] - m5[debut]["t"]) / 3600)
    return {
        "n": total, "wins": gagnes, "pct": pct, "couleur": couleur, "heures": heures,
        "achat_n": n["ACHAT"], "achat_w": w["ACHAT"], "vente_n": n["VENTE"], "vente_w": w["VENTE"],
        "verdict": verdict,
        "sim_n": sim_n, "sim_tp": sim["tp"], "sim_sl": sim["sl"], "sim_to": sim["to"],
        "sim_r": sim_r, "sim_couleur": sim_couleur, "sim_verdict": sim_verdict,
    }


# ───────────── FRAÎCHEUR ─────────────

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


# ───────────── SIGNAL ─────────────

def conditions_paire(tfs, ferme=False, generale=None):
    manque = [k for k in ["H1", "M30", "M15", "M5"] if k not in tfs]
    if manque:
        return {"signal": "?", "couleur": "jaune", "bonus": False,
                "conseil": "Données manquantes : " + ", ".join(manque), "details": {}, "alertes": [],
                "generale": generale}

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
            conseil = f"✅ Signal {nom} détecté. Entre à la clôture M5 seulement si le guide ci-dessous est au vert."
            if bonus:
                conseil = f"⭐ Signal {nom} détecté, H4 dans le même sens. Entre à la clôture M5 seulement si le guide ci-dessous est au vert."
            dep = m5["ema_depuis"] or 0
            if dep <= 3:
                conseil += " 🔥 Croisement EMA récent."
            elif dep > 12:
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
        conseil = f"⏸️ Attendre : {ok}/{len(d)} conditions {nom}. Il manque : {', '.join(manque_c)}."
        base = {"signal": "ATTENDRE", "couleur": "jaune", "bonus": False, "conseil": conseil,
                "details": {**d, f"H4 {fl}": bonus}}

    base["alertes"] = []
    base["generale"] = generale
    if ferme:
        return {**base, "signal": "FERMÉ", "couleur": "gris", "bonus": False,
                "conseil": "🌙 Marché fermé : ces chiffres viennent de la dernière séance. N'entre pas. "
                           f"(Dernier verdict : {base['signal']})"}
    return base


def alertes_signal(cond, tfs, niv, sltp, bt, mm_h1):
    sig = cond["signal"]
    if sig not in ("ACHAT", "VENTE"):
        return []
    al = []
    achat = sig == "ACHAT"
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

    sens = "haussiere" if achat else "baissiere"
    if m15["cassure"]["sens"] == sens:
        al.append({"txt": "💥 Cassure M15 dans le sens du signal (elle peut être fausse : vérifie que la bougie suivante tient).", "cls": "vert"})

    pour = "motif_haussier" if achat else "motif_baissier"
    contre = "motif_baissier" if achat else "motif_haussier"
    if m5[pour] or m15[pour]:
        al.append({"txt": "🕯️ Bougie de rejet ou d'englobement dans le même sens (un indice, pas une preuve).", "cls": "vert"})
    if m5[contre]:
        al.append({"txt": "⚠️ Bougie M5 de sens contraire (mèche ou englobante) : prudence.", "cls": "jaune"})

    if bt and bt["n"] >= 10 and bt["pct"] is not None and bt["pct"] < 45:
        al.append({"txt": f"⚠️ Sur le passé, ce signal ne va dans le bon sens que {bt['pct']}% des fois sur cette devise.", "cls": "jaune"})
    if bt and bt["sim_n"] >= 8 and bt["sim_r"] is not None and bt["sim_r"] < 0:
        al.append({"txt": f"⚠️ Sur le passé, avec stop et objectif, ce signal perd en moyenne {abs(bt['sim_r'])} R par trade.", "cls": "jaune"})
    return al


# ───────────── GUIDE DU TRADE ─────────────

def construire_guide(pair, cond, tfs, sltp, filtres=None):
    sig = cond["signal"]
    if sig not in ("ACHAT", "VENTE") or not sltp:
        return None
    achat = sig == "ACHAT"
    cote = sltp["achat" if achat else "vente"]
    prix = tfs["M5"]["prix"]
    raisons = [a["txt"] for a in cond.get("alertes", []) if a["cls"] == "jaune"]
    nxt = (int(time.time() // 300) + 1) * 300
    g = {
        "sens": sig,
        "feu": "vert" if not raisons else "orange",
        "raisons": raisons,
        "heure": datetime.fromtimestamp(nxt, BENIN).strftime("%H:%M"),
        "prix": fp(pair, prix),
        "contrat": CONTRAT.get(pair),
        "stop": None,
    }
    if cote:
        g.update({
            "stop": cote["stop"], "objectif": cote["objectif"], "ratio": cote["ratio"],
            "dist": f"{abs(prix - cote['stop_brut']):.8f}",
            "dist_obj": f"{abs(cote['obj_brut'] - prix):.8f}",
            "moitie": fp(pair, prix + (cote["obj_brut"] - prix) / 2),
        })
    else:
        g["feu"] = "orange"
    return g


# ───────────── ANALYSE D'UNE PAIRE ─────────────

def analyser_paire(pair, source, historique=None, autres_cond=None):
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
    stats = calc_stats(pair, series.get("H1", []), resultats)
    generale = calc_tendance_generale(resultats)
    cond = conditions_paire(resultats, fr["etat"] == "ferme", generale)
    prix = fp(pair, resultats["M5"]["prix"]) if "M5" in resultats else "—"

    niv = sltp = bt = heures = guide = filtres = None
    h1, m15 = series.get("H1", []), series.get("M15", [])
    if "M5" in resultats and len(h1) >= 30 and len(m15) >= 30:
        p = resultats["M5"]["prix"]
        niv = calc_niveaux(pair, p, m15, h1)
        sltp = calc_sltp(pair, p, m15)
        heures = calc_heures(h1)
        bt = backtest(series, pair)
        cond["alertes"] = alertes_signal(cond, resultats, niv, sltp, bt, mouvement_moyen(h1, 24))

        # Filtres de fiabilité
        sens_sig = None
        if cond["signal"] == "ACHAT":
            sens_sig = "haussier"
        elif cond["signal"] == "VENTE":
            sens_sig = "baissier"
        filtres = filtre_global(resultats, cond, pair, sens_sig, historique or [], autres_cond or {})

        guide = construire_guide(pair, cond, resultats, sltp, filtres)

    return {"tfs": resultats, "stats": stats, "fraicheur": fr, "cond": cond, "prix": prix,
            "niv": niv, "sltp": sltp, "bt": bt, "heures": heures, "guide": guide,
            "generale": generale, "filtres": filtres}


def analyser_securise(pair, source, historique=None, autres_cond=None):
    try:
        return analyser_paire(pair, source, historique, autres_cond)
    except Exception as e:
        print(f"Erreur {pair}: {e}")
        return {"tfs": {}, "stats": None,
                "fraicheur": {"texte": "Erreur de chargement", "etat": "ferme"},
                "cond": conditions_paire({}), "prix": "—",
                "niv": None, "sltp": None, "bt": None, "heures": None, "guide": None,
                "generale": None, "filtres": None}


def analyser_tout():
    taches = [(p, "yahoo") for p in YAHOO_PAIRS] + [(p, "deriv") for p in DERIV_PAIRS]

    # Passe 1 : analyse brute (sans filtres croisés)
    res = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        for (pair, _), data in zip(taches, ex.map(lambda t: analyser_securise(*t), taches)):
            res[pair] = data

    # Passe 2 : on recalcule les filtres avec les autres signaux connus
    autres_cond = {p: res[p]["cond"] for p in res}
    historique = session.get("historique", [])
    for pair, source in taches:
        try:
            res[pair] = analyser_paire(pair, source, historique, autres_cond)
        except Exception as e:
            print(f"Erreur passe 2 {pair}: {e}")

    # Mise à jour de l'historique
    maintenant = time.time()
    for pair, d in res.items():
        cond = d["cond"]
        if cond["signal"] in ("ACHAT", "VENTE"):
            historique.append({"pair": pair, "sens": cond["signal"], "t": maintenant,
                               "prix": d["prix"], "feu": d["guide"]["feu"] if d.get("guide") else "?"})
    # Garde seulement les 50 derniers
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
    historique = session.get("historique", [])
    session_act = session_actuelle()
    return render_template("dashboard.html", paires=paires, now=datetime.now(BENIN).strftime("%H:%M"),
                           historique=historique[-10:], session_act=session_act)


@app.route("/ping")
def ping():
    return "ok", 200


@app.route("/cron")
def cron():
    analyser_tout()
    return "ok", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
