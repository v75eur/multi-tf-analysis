import os, json, hashlib, secrets, time, bisect, copy
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, redirect, url_for, session, render_template, jsonify
import requests
import websocket
import numpy as np

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", secrets.token_hex(32))
ADMIN_HASH = os.getenv("ADMIN_HASH", "d0695d2f4b6487fb81c7047ba01d06d5065aa1a9f18f89633389e1e9b5d85fd6")

BENIN = timezone(timedelta(hours=1))
JOURS = ["lun", "mar", "mer", "jeu", "ven", "sam", "dim"]

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CONFIG_FILE = os.path.join(DATA_DIR, "config.json")

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
DERIV_WS = "wss://api.derivws.com/trading/v1/options/ws/public"
CACHE_SEC = {"1h": 300, "30m": 180, "15m": 120, "5m": 60}
_cache_yahoo = {}

CONFIG_DEFAUT = {
    "poids_tf": {"M5": 5, "M15": 4, "M30": 3, "H1": 2, "H4": 1},
    "ema_court": 5,
    "ema_long": 25,
    "periode_tendance": 10,
    "timing_frais_max": 5,
    "timing_moyen_max": 12,
    "seuil_agitation_max": 70,
    "duree_signal_min": 30,
    "ratio_min": 1.5,
    "risque_max_pct": 1.0,
    "journal_max": 100,
    "sessions": {
        "asie":    {"debut": 1,  "fin": 9,  "qualite": "faible",  "label": "Asie"},
        "londres": {"debut": 9,  "fin": 15, "qualite": "forte",   "label": "Londres"},
        "overlap": {"debut": 15, "fin": 18, "qualite": "excellente", "label": "Londres + New York"},
        "newyork": {"debut": 18, "fin": 23, "qualite": "forte",   "label": "New York"},
        "nuit":    {"debut": 23, "fin": 1,  "qualite": "faible",  "label": "Nuit"}
    },
    "paires": {
        "EURUSD": {"actif": True, "source": "yahoo", "correlees": ["GBPUSD"]},
        "GBPUSD": {"actif": True, "source": "yahoo", "correlees": ["EURUSD"]},
        "XAUUSD": {"actif": True, "source": "yahoo", "correlees": []},
        "V75":    {"actif": True, "source": "deriv", "correlees": []}
    },
    "filtres_actifs": {
        "session": True,
        "volatilite": True,
        "tendance_superieure": True,
        "timing": True,
        "duplication": True,
        "correlation": True
    },
    "conseil_mode": "detaille"
}


def charger_config():
    """Charge la config depuis le fichier. Crée le défaut si absent."""
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        if not os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(CONFIG_DEFAUT, f, indent=2, ensure_ascii=False)
            return copy.deepcopy(CONFIG_DEFAUT)
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        # Fusionne avec le défaut pour avoir toutes les clés
        final = copy.deepcopy(CONFIG_DEFAUT)
        for k, v in cfg.items():
            if isinstance(v, dict) and isinstance(final.get(k), dict):
                final[k].update(v)
            else:
                final[k] = v
        return final
    except Exception as e:
        print(f"Erreur chargement config : {e}")
        return copy.deepcopy(CONFIG_DEFAUT)


def sauver_config(cfg):
    """Sauve la config dans le fichier."""
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Erreur sauvegarde config : {e}")
        return False


def cfg():
    """Récupère la config (rechargée à chaque appel pour que les changements soient pris)."""
    return charger_config()


def logged():
    return session.get("admin") is True


def fp(pair, x):
    return f"{x:.{DECIMALES.get(pair, 5)}f}"


# ───────────── RÉCUPÉRATION ─────────────

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


def calc_ema_signal(clotures, c):
    s5 = calc_ema_serie(clotures, c["ema_court"])
    s25 = calc_ema_serie(clotures, c["ema_long"])
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


def calc_tendance(clotures, periode):
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


# ───────────── MOTIFS ─────────────

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


def analyser(tf, bougies, c):
    if len(bougies) < 30:
        return None
    clotures = [b["close"] for b in bougies]
    tendance, pct_tendance = calc_tendance(clotures[-c["periode_tendance"]:], c["periode_tendance"])
    v, r, pa, pv = calc_acheteurs(bougies)
    cc, ag = calc_agitation(bougies)
    creux = trouver_creux(bougies)
    sommets = trouver_sommets(bougies)
    divergence = calc_divergence(bougies, creux, sommets)
    ema5, ema25, ema_sig, ema_depuis = calc_ema_signal(clotures, c)
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


# ───────────── TIMING ─────────────

def calc_timing_entree(m5, c):
    if not m5 or m5.get("ema_depuis") is None:
        return None
    dep = m5["ema_depuis"]
    if dep <= c["timing_frais_max"]:
        return {"niveau": "FRAIS", "couleur": "vert", "emoji": "🔥",
                "txt": f"Croisement EMA M5 très récent ({dep} bougies). Tu entres au début du mouvement.",
                "lot_conseil": "LOT PLEIN", "lot_ratio": 1.0}
    if dep <= c["timing_moyen_max"]:
        return {"niveau": "EN COURS", "couleur": "jaune", "emoji": "⏳",
                "txt": f"Croisement EMA M5 il y a {dep} bougies. Mouvement déjà lancé.",
                "lot_conseil": "DEMI-LOT", "lot_ratio": 0.5}
    return {"niveau": "TARDIF", "couleur": "rouge", "emoji": "❌",
            "txt": f"Croisement EMA M5 il y a {dep} bougies. Trop tardif, attends le prochain.",
            "lot_conseil": "ATTENDRE", "lot_ratio": 0.0}


# ───────────── TENDANCE GÉNÉRALE ─────────────

def calc_tendance_generale(tfs, c):
    if not tfs:
        return None
    poids = c["poids_tf"]
    poids_total = 0
    somme_ach = 0.0
    somme_ven = 0.0
    for tf, data in tfs.items():
        p = poids.get(tf, 1)
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
        p = poids.get(tf, 1)
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

    ordre = ["M5", "M15", "M30", "H1", "H4"]
    alignement = {"haussier": 0, "baissier": 0, "detail": []}
    for sens in ("haussier", "baissier"):
        compte = 0
        for tf in ordre:
            if tf not in tfs:
                break
            if tfs[tf]["tendance"] == sens:
                compte += 1
            else:
                break
        alignement[sens] = compte
    for tf in ordre:
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
                                "txt": "M15+M30+H1 haussiers + croisement EMA M5 haussier récent → ACHAT confirmé."}
        elif m15["tendance"] == m30["tendance"] == h1["tendance"] == "baissier":
            if m5["ema_signal"] == "baissier" and (m5["ema_depuis"] or 99) <= 3:
                confirmation = {"sens": "VENTE", "couleur": "rouge",
                                "txt": "M15+M30+H1 baissiers + croisement EMA M5 baissier récent → VENTE confirmée."}

    return {
        "ach_gen": ach_gen, "ven_gen": ven_gen,
        "score_pct": score_pct, "tendance": tendance, "couleur": couleur,
        "align_sens": align_sens, "align_n": align_n,
        "align_detail": alignement["detail"],
        "confirmation": confirmation,
    }


# ───────────── SESSION ─────────────

def session_actuelle(c):
    h = datetime.now(BENIN).hour
    for code, s in c["sessions"].items():
        d, f = s["debut"], s["fin"]
        if d < f:
            if d <= h < f:
                return {"heure": h, "code": code, "qualite": s["qualite"], "label": s["label"]}
        else:  # passe minuit
            if h >= d or h < f:
                return {"heure": h, "code": code, "qualite": s["qualite"], "label": s["label"]}
    return {"heure": h, "code": "inconnu", "qualite": "faible", "label": "Inconnue"}


# ───────────── MOTEUR DE CONSEIL CONTEXTUEL ─────────────

def generer_conseil(pair, d, c):
    """Analyse complète et génère un conseil humain clair."""
    cond = d["cond"]
    tfs = d["tfs"]
    fr = d["fraicheur"]
    niv = d["niv"]
    sltp = d["sltp"]
    bt = d["bt"]

    if fr["etat"] == "ferme":
        return {"titre": "🌙 Marché fermé",
                "texte": f"{pair} est fermé. Le bot ne donne pas de conseil tant que le marché est fermé.",
                "couleur": "gris", "action": "ATTENDRE", "urgence": "aucune",
                "raison": "Marché fermé"}

    if not tfs or "M5" not in tfs or not cond:
        return {"titre": "⏸️ Données insuffisantes",
                "texte": f"Pas assez de données sur {pair} pour un conseil fiable. Attends le prochain cycle.",
                "couleur": "gris", "action": "ATTENDRE", "urgence": "aucune",
                "raison": "Données manquantes"}

    sig = cond["signal"]
    timing = cond.get("timing")
    session = session_actuelle(c)
    filtres_actifs = c.get("filtres_actifs", {})

    if sig not in ("ACHAT", "VENTE"):
        manque = [k for k, v in cond["details"].items() if not v]
        premier = manque[0] if manque else "alignement des timeframes"
        return {"titre": "⏸️ Attendre",
                "texte": f"Pas de signal clair sur {pair}. Il manque : {premier.lower()}. Ne force pas l'entrée.",
                "couleur": "jaune", "action": "ATTENDRE", "urgence": "aucune",
                "raison": f"Pas de signal"}

    sens = sig
    achat = sens == "ACHAT"
    prix = tfs["M5"]["prix"]

    session_ok = (not filtres_actifs.get("session", True)) or session["qualite"] in ("forte", "excellente")
    timing_frais = timing and timing["niveau"] == "FRAIS"
    timing_moyen = timing and timing["niveau"] == "EN COURS"
    timing_tardif = timing and timing["niveau"] == "TARDIF"

    h4 = tfs.get("H4")
    h1 = tfs.get("H1")
    contre = "baissier" if achat else "haussier"
    h4_contre = h4 and h4["tendance"] == contre
    h1_contre = h1 and h1["tendance"] == contre
    fond_contre = filtres_actifs.get("tendance_superieure", True) and h4_contre and h1_contre

    agitation = tfs["H1"]["pct_agitation"] if "H1" in tfs else 0
    trop_agite = filtres_actifs.get("volatilite", True) and agitation > c["seuil_agitation_max"]

    cote = sltp["achat" if achat else "vente"] if sltp else None
    ratio_ok = cote and cote["ratio"] >= c["ratio_min"]
    ratio_faible = cote and cote["ratio"] < c["ratio_min"]

    mm_h1 = mouvement_moyen([{"high": h1["prix"], "low": h1["prix"], "close": h1["prix"]}], 1) if h1 else 0
    res_proche = False
    sup_proche = False
    if niv:
        if achat and niv["res_brut"] is not None and niv["res_brut"] - prix < mm_h1:
            res_proche = True
        if not achat and niv["sup_brut"] is not None and prix - niv["sup_brut"] < mm_h1:
            sup_proche = True

    grosse_bougie = tfs["M5"]["grande"]
    bt_mauvais = bt and bt["n"] >= 10 and bt["pct"] is not None and bt["pct"] < 45

    # BLOQUÉS
    if fond_contre:
        return {"titre": "🚫 N'ENTRE PAS",
                "texte": f"Signal {sens} sur {pair}, MAIS H4 et H1 sont {contre}s. C'est un piège : tu irais contre la tendance de fond. Passe ce trade.",
                "couleur": "rouge", "action": "NE PAS ENTRER", "urgence": "haute",
                "raison": f"H4 et H1 contre le signal"}
    if filtres_actifs.get("timing", True) and timing_tardif:
        return {"titre": "❌ TROP TARD",
                "texte": f"Signal {sens} sur {pair}, mais le croisement EMA M5 a {tfs['M5']['ema_depuis']} bougies. Le mouvement est déjà fait. Attends le prochain croisement.",
                "couleur": "rouge", "action": "NE PAS ENTRER", "urgence": "haute",
                "raison": f"Timing tardif"}
    if trop_agite:
        return {"titre": "🚫 MARCHÉ TROP AGITÉ",
                "texte": f"Signal {sens} sur {pair}, mais agitation H1 = {agitation}%. Le bruit va toucher ton stop. Attends que ça se calme.",
                "couleur": "rouge", "action": "NE PAS ENTRER", "urgence": "haute",
                "raison": f"Agitation {agitation}%"}

    # FEU VERT PARFAIT
    if session_ok and timing_frais and not h4_contre and not h1_contre and ratio_ok and not res_proche and not sup_proche and not grosse_bougie and not bt_mauvais:
        qualite = "excellent" if session["qualite"] == "excellente" else "très bon"
        return {"titre": f"⭐ SETUP {qualite.upper()} — ENTRE",
                "texte": f"Tout est aligné sur {pair} : session {session['label']}, timing FRAIS ({tfs['M5']['ema_depuis']} bougies), H4/H1 d'accord, ratio {cote['ratio']}, pas de niveau proche. Entre à la clôture M5 avec LOT PLEIN. Stop : {cote['stop']}, objectif : {cote['objectif']}.",
                "couleur": "vert", "action": "ENTRER LOT PLEIN", "urgence": "haute",
                "raison": "Tous les filtres au vert"}

    # BON mais timing EN COURS
    if session_ok and timing_moyen and not h4_contre and not h1_contre and ratio_ok and not res_proche and not sup_proche and not grosse_bougie:
        return {"titre": "✅ SIGNAL VALIDE — DEMI-LOT",
                "texte": f"Signal {sens} sur {pair}, session {session['label']}, H4/H1 d'accord, ratio {cote['ratio']}. MAIS croisement EMA M5 il y a {tfs['M5']['ema_depuis']} bougies : mouvement déjà lancé. Entre avec DEMI-LOT. Stop : {cote['stop']}, objectif : {cote['objectif']}.",
                "couleur": "vert", "action": "ENTRER DEMI-LOT", "urgence": "moyenne",
                "raison": "Timing en cours"}

    # Signal OK mais mauvaise session
    if not session_ok and timing_frais and not h4_contre and not h1_contre:
        return {"titre": "⚠️ SIGNAL OK, MAUVAISE SESSION",
                "texte": f"Signal {sens} sur {pair}, timing frais, tendance OK. MAIS session {session['label']} : peu de volume, signaux souvent faux. Attends Londres (9h) ou NY (15h).",
                "couleur": "jaune", "action": "ATTENDRE LA SESSION", "urgence": "moyenne",
                "raison": f"Session {session['label']}"}

    # Ratio faible
    if ratio_faible and timing_frais and session_ok:
        return {"titre": "⚠️ RATIO FAIBLE",
                "texte": f"Signal {sens} sur {pair}, bon timing, bonne session. MAIS ratio {cote['ratio']} (sous {c['ratio_min']}). Tu risques plus que ce que tu peux gagner. Passe ou attends un meilleur point.",
                "couleur": "jaune", "action": "PASSER OU ATTENDRE", "urgence": "moyenne",
                "raison": f"Ratio {cote['ratio']}"}

    # Niveau proche
    if (res_proche or sup_proche) and timing_frais and session_ok:
        niveau_txt = "résistance" if res_proche else "support"
        niveau_val = niv["res"] if res_proche else niv["sup"]
        return {"titre": "⚠️ NIVEAU PROCHE",
                "texte": f"Signal {sens} sur {pair}, bon timing, bonne session. MAIS un {niveau_txt} est proche ({niveau_val}). Ton objectif risque d'être bloqué. Réduis ton objectif ou attends la cassure.",
                "couleur": "jaune", "action": "RÉDUIRE OBJECTIF OU ATTENDRE", "urgence": "moyenne",
                "raison": f"{niveau_txt} proche"}

    # Grosse bougie
    if grosse_bougie:
        return {"titre": "⚠️ GROSSE BOUGIE",
                "texte": f"Signal {sens} sur {pair}, mais la dernière bougie M5 est ×{tfs['M5']['taille_x']} la moyenne. Le mouvement a déjà été fait d'un coup. Attends la prochaine bougie.",
                "couleur": "jaune", "action": "ATTENDRE LA PROCHAINE BOUGIE", "urgence": "moyenne",
                "raison": "Grosse bougie"}

    # Backtest mauvais
    if bt_mauvais:
        return {"titre": "⚠️ BACKTEST FAIBLE",
                "texte": f"Signal {sens} sur {pair}. MAIS sur le passé testé, ce signal ne va dans le bon sens que {bt['pct']}% du temps sur cette paire. Sois très prudent ou passe.",
                "couleur": "jaune", "action": "PRUDENCE", "urgence": "moyenne",
                "raison": f"Backtest {bt['pct']}%"}

    # Défaut : signal valide mais un point à surveiller
    return {"titre": "⏸️ SIGNAL VALIDE SOUS CONDITIONS",
            "texte": f"Signal {sens} sur {pair}. Vérifie le guide avant d'entrer, et respecte le lot conseillé.",
            "couleur": "jaune", "action": "VÉRIFIER LE GUIDE", "urgence": "basse",
            "raison": "Cas mixte"}


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
        cc = 1 if b["close"] > b["open"] else -1 if b["close"] < b["open"] else 0
        if n_serie == 0:
            if cc == 0:
                break
            sens, n_serie = cc, 1
        elif cc == sens:
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


# ───────────── NIVEAUX ─────────────

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


# ───────────── SL/TP ─────────────

def cote_sltp(pair, prix, stop, obj, sens, ratio_min):
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
        "ratio": round(ratio, 2), "ok": ratio >= ratio_min,
    }


def calc_sltp(pair, prix, m15, ratio_min):
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

    return {"achat": cote_sltp(pair, prix, stop_a, obj_a, "achat", ratio_min),
            "vente": cote_sltp(pair, prix, stop_v, obj_v, "vente", ratio_min)}


# ───────────── HEURES ─────────────

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


# ───────────── BACKTEST ─────────────

def tendance_simple(cl):
    x = np.arange(len(cl))
    p = np.polyfit(x, np.array(cl), 1)[0]
    return 1 if p > 0 else -1 if p < 0 else 0


def backtest(series, c, pair="", horizon=12, test=1000, max_trade=48):
    if any(len(series.get(tf, [])) < 30 for tf in ("H1", "M30", "M15", "M5")):
        return None
    m5 = series["M5"]
    ends = {tf: [b["t"] + g for b in series[tf]] for tf, g in (("H1", 3600), ("M30", 1800), ("M15", 900))}
    debut = max(40, len(m5) - test)
    n = {"ACHAT": 0, "VENTE": 0}
    w = {"ACHAT": 0, "VENTE": 0}
    sim = {"tp": 0, "sl": 0, "to": 0, "r": []}
    prev = None
    periode = c["periode_tendance"]
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
            tend = tendance_simple([b["close"] for b in sl[-periode:]])
            d = sl[-periode:]
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
            e5, e25 = calc_ema_serie(cl, c["ema_court"]), calc_ema_serie(cl, c["ema_long"])
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
                cote = calc_sltp(pair, entree, sl15, c["ratio_min"])["achat" if sig == "ACHAT" else "vente"]
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
        sim_verdict, sim_couleur = "Trop peu de trades simulés pour conclure.", ""
    elif sim_r > 0.15:
        sim_verdict, sim_couleur = "Avec stop et objectif, le signal gagne en moyenne.", "vert"
    elif sim_r < -0.1:
        sim_verdict, sim_couleur = "Avec stop et objectif, le signal perd en moyenne.", "rouge"
    else:
        sim_verdict, sim_couleur = "Résultat proche de zéro.", ""

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

def conditions_paire(tfs, c, ferme=False, generale=None):
    manque = [k for k in ["H1", "M30", "M15", "M5"] if k not in tfs]
    if manque:
        return {"signal": "?", "couleur": "jaune", "bonus": False,
                "conseil": "Données manquantes : " + ", ".join(manque), "details": {}, "alertes": [],
                "generale": generale, "timing": None}

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
    timing = calc_timing_entree(m5, c)
    base = None
    for sens, nom, couleur in (("haussier", "ACHAT", "vert"), ("baissier", "VENTE", "rouge")):
        d, bonus, fl = res[sens]
        if all(d.values()):
            conseil = f"✅ Signal {nom} : M15, M30 et H1 alignés {sens}s, acheteurs dominent, EMA M5 croisée."
            if bonus:
                conseil = f"⭐ Signal {nom} renforcé : H4 aussi {sens}."
            if timing:
                if timing["niveau"] == "FRAIS":
                    conseil += f" 🔥 Croisement M5 frais ({m5['ema_depuis']} bg) : lot plein."
                elif timing["niveau"] == "EN COURS":
                    conseil += f" ⏳ Croisement M5 en cours ({m5['ema_depuis']} bg) : demi-lot."
                else:
                    conseil += f" ❌ Croisement M5 tardif ({m5['ema_depuis']} bg) : attends."
            details = {**d, f"H4 {fl}": bonus}
            base = {"signal": nom, "couleur": couleur, "bonus": bonus, "conseil": conseil, "details": details, "timing": timing}
            break

    if base is None:
        meilleur = max(("haussier", "baissier"), key=lambda s: sum(res[s][0].values()))
        d, bonus, fl = res[meilleur]
        nom = "ACHAT" if meilleur == "haussier" else "VENTE"
        manque_c = [k for k, v in d.items() if not v]
        ok = len(d) - len(manque_c)
        conseil = f"⏸️ Attendre : {ok}/{len(d)} conditions {nom}. Manque : {', '.join(manque_c)}."
        base = {"signal": "ATTENDRE", "couleur": "jaune", "bonus": False, "conseil": conseil,
                "details": {**d, f"H4 {fl}": bonus}, "timing": timing}

    base["alertes"] = []
    base["generale"] = generale
    if ferme:
        return {**base, "signal": "FERMÉ", "couleur": "gris", "bonus": False,
                "conseil": "🌙 Marché fermé. N'entre pas."}
    return base


def alertes_signal(cond, tfs, niv, sltp, bt, mm_h1, c):
    sig = cond["signal"]
    if sig not in ("ACHAT", "VENTE"):
        return []
    al = []
    achat = sig == "ACHAT"
    m5, m15 = tfs["M5"], tfs["M15"]
    prix = m5["prix"]

    if achat and niv and niv["res_brut"] is not None and niv["res_brut"] - prix < mm_h1:
        al.append({"txt": f"⚠️ Résistance proche ({niv['res']}). Rebond possible.", "cls": "jaune"})
    if not achat and niv and niv["sup_brut"] is not None and prix - niv["sup_brut"] < mm_h1:
        al.append({"txt": f"⚠️ Support proche ({niv['sup']}). Rebond possible.", "cls": "jaune"})

    if m5["grande"]:
        al.append({"txt": f"⚠️ Grosse bougie M5 (×{m5['taille_x']}). Attends la prochaine.", "cls": "jaune"})

    cote = sltp["achat" if achat else "vente"] if sltp else None
    if cote is None:
        al.append({"txt": "⚠️ Pas d'objectif clair. Attends.", "cls": "jaune"})
    elif cote["ok"]:
        al.append({"txt": f"✅ Ratio {cote['ratio']} (≥ {c['ratio_min']}).", "cls": "vert"})
    else:
        al.append({"txt": f"⚠️ Ratio {cote['ratio']} < {c['ratio_min']}. Passe.", "cls": "jaune"})

    sens = "haussiere" if achat else "baissiere"
    if m15["cassure"]["sens"] == sens:
        al.append({"txt": "💥 Cassure M15 dans le sens du signal.", "cls": "vert"})

    pour = "motif_haussier" if achat else "motif_baissier"
    contre = "motif_baissier" if achat else "motif_haussier"
    if m5[pour] or m15[pour]:
        al.append({"txt": "🕯️ Bougie de rejet/englobante dans le sens.", "cls": "vert"})
    if m5[contre]:
        al.append({"txt": "⚠️ Bougie M5 contraire. Prudence.", "cls": "jaune"})

    if bt and bt["n"] >= 10 and bt["pct"] is not None and bt["pct"] < 45:
        al.append({"txt": f"⚠️ Backtest faible ({bt['pct']}%).", "cls": "jaune"})
    if bt and bt["sim_n"] >= 8 and bt["sim_r"] is not None and bt["sim_r"] < 0:
        al.append({"txt": f"⚠️ Backtest perd {abs(bt['sim_r'])} R en moyenne.", "cls": "jaune"})
    return al


# ───────────── GUIDE ─────────────

def construire_guide(pair, cond, tfs, sltp, c):
    sig = cond["signal"]
    if sig not in ("ACHAT", "VENTE") or not sltp:
        return None
    achat = sig == "ACHAT"
    cote = sltp["achat" if achat else "vente"]
    prix = tfs["M5"]["prix"]
    raisons = [a["txt"] for a in cond.get("alertes", []) if a["cls"] == "jaune"]
    nxt = (int(time.time() // 300) + 1) * 300
    timing = cond.get("timing")

    feu = "vert" if not raisons else "orange"
    lot_niveau = "PLEIN"
    lot_txt = "Lot plein autorisé : tu entres au début du mouvement."

    if timing:
        if timing["niveau"] == "EN COURS":
            feu = "orange"
            lot_niveau = "DEMI"
            lot_txt = "Demi-lot conseillé : le mouvement est déjà lancé."
        elif timing["niveau"] == "TARDIF":
            feu = "orange"
            lot_niveau = "ATTENDRE"
            lot_txt = "N'ENTRE PAS : trop tardif. Attends le prochain croisement."

    g = {
        "sens": sig, "feu": feu, "raisons": raisons,
        "heure": datetime.fromtimestamp(nxt, BENIN).strftime("%H:%M"),
        "prix": fp(pair, prix), "contrat": CONTRAT.get(pair),
        "stop": None, "timing": timing,
        "lot_niveau": lot_niveau, "lot_txt": lot_txt,
        "risque_max": c["risque_max_pct"],
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


# ───────────── ANALYSE ─────────────

def analyser_paire(pair, source, c, historique=None, autres_cond=None):
    series, resultats = {}, {}
    for tf, (interval, range_, gran) in TIMEFRAMES.items():
        if source == "yahoo":
            ysym = {"EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "XAUUSD": "GC=F"}.get(pair)
            if not ysym:
                continue
            brut = recuperer_bougies_yahoo(ysym, interval, range_)
            if tf == "H4":
                brut = regrouper(brut, gran)
        else:
            dsym = {"V75": "R_75"}.get(pair)
            if not dsym:
                continue
            brut = recuperer_bougies_deriv(dsym, gran, DERIV_COUNT[tf])
        bougies = bougies_fermees(brut, gran)
        series[tf] = bougies
        r = analyser(tf, bougies, c)
        if r:
            resultats[tf] = r

    fr = fraicheur(series.get("M5"))
    stats = calc_stats(pair, series.get("H1", []), resultats)
    generale = calc_tendance_generale(resultats, c)
    cond = conditions_paire(resultats, c, fr["etat"] == "ferme", generale)
    prix = fp(pair, resultats["M5"]["prix"]) if "M5" in resultats else "—"

    niv = sltp = bt = heures = guide = None
    h1, m15 = series.get("H1", []), series.get("M15", [])
    if "M5" in resultats and len(h1) >= 30 and len(m15) >= 30:
        p = resultats["M5"]["prix"]
        niv = calc_niveaux(pair, p, m15, h1)
        sltp = calc_sltp(pair, p, m15, c["ratio_min"])
        heures = calc_heures(h1)
        bt = backtest(series, c, pair)
        mm_h1 = mouvement_moyen(h1, 24)
        cond["alertes"] = alertes_signal(cond, resultats, niv, sltp, bt, mm_h1, c)
        guide = construire_guide(pair, cond, resultats, sltp, c)

    conseil = generer_conseil(pair, {
        "cond": cond, "tfs": resultats, "generale": generale,
        "fraicheur": fr, "niv": niv, "sltp": sltp, "bt": bt
    }, c)

    return {"tfs": resultats, "stats": stats, "fraicheur": fr, "cond": cond, "prix": prix,
            "niv": niv, "sltp": sltp, "bt": bt, "heures": heures, "guide": guide,
            "generale": generale, "conseil": conseil}


def analyser_securise(pair, source, c, historique=None, autres_cond=None):
    try:
        return analyser_paire(pair, source, c, historique, autres_cond)
    except Exception as e:
        print(f"Erreur {pair}: {e}")
        return {"tfs": {}, "stats": None,
                "fraicheur": {"texte": "Erreur de chargement", "etat": "ferme"},
                "cond": conditions_paire({}, c), "prix": "—",
                "niv": None, "sltp": None, "bt": None, "heures": None, "guide": None,
                "generale": None,
                "conseil": {"titre": "❌ Erreur", "texte": f"Impossible de charger {pair}.",
                            "couleur": "rouge", "action": "ATTENDRE", "urgence": "haute",
                            "raison": "Erreur"}}


def analyser_tout():
    c = cfg()
    paires_cfg = c.get("paires", {})
    taches = [(p, paires_cfg[p]["source"]) for p in paires_cfg if paires_cfg[p].get("actif", True)]

    res = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        for (pair, src), data in zip(taches, ex.map(lambda t: analyser_securise(t[0], t[1], c), taches)):
            res[pair] = data

    # Mise à jour du journal (persistant en session)
    journal = session.get("journal", [])
    maintenant = time.time()
    for pair, d in res.items():
        cons = d.get("conseil")
        if not cons:
            continue
        # Signature : titre + action + couleur. On ajoute seulement si différent du dernier pour cette paire.
        signature = f"{pair}|{cons['titre']}|{cons['action']}|{cons['couleur']}"
        dernier = None
        for entree in reversed(journal):
            if entree["pair"] == pair:
                dernier = entree
                break
        if dernier and dernier.get("signature") == signature:
            continue  # Rien de nouveau pour cette paire
        journal.append({
            "t": maintenant,
            "heure": datetime.now(BENIN).strftime("%H:%M:%S"),
            "pair": pair,
            "titre": cons["titre"],
            "texte": cons["texte"],
            "action": cons["action"],
            "couleur": cons["couleur"],
            "raison": cons["raison"],
            "signature": signature,
            "prix": d.get("prix", "—"),
        })
    journal = journal[-c["journal_max"]:]
    session["journal"] = journal

    return res, journal


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
    c = cfg()
    paires, journal = analyser_tout()
    session_act = session_actuelle(c)
    return render_template("dashboard.html", paires=paires, now=datetime.now(BENIN).strftime("%H:%M"),
                           journal=journal, session_act=session_act, config=c)


@app.route("/config", methods=["GET", "POST"])
def config_page():
    if not logged():
        return redirect(url_for("index"))
    c = cfg()
    message = None
    if request.method == "POST":
        try:
            # Reconstruit la config depuis le formulaire
            nouvelle = copy.deepcopy(c)

            def lire_float(nom, defaut):
                try:
                    return float(request.form.get(nom, defaut))
                except:
                    return defaut
            def lire_int(nom, defaut):
                try:
                    return int(float(request.form.get(nom, defaut)))
                except:
                    return defaut

            nouvelle["ema_court"] = lire_int("ema_court", c["ema_court"])
            nouvelle["ema_long"] = lire_int("ema_long", c["ema_long"])
            nouvelle["periode_tendance"] = lire_int("periode_tendance", c["periode_tendance"])
            nouvelle["timing_frais_max"] = lire_int("timing_frais_max", c["timing_frais_max"])
            nouvelle["timing_moyen_max"] = lire_int("timing_moyen_max", c["timing_moyen_max"])
            nouvelle["seuil_agitation_max"] = lire_int("seuil_agitation_max", c["seuil_agitation_max"])
            nouvelle["duree_signal_min"] = lire_int("duree_signal_min", c["duree_signal_min"])
            nouvelle["ratio_min"] = lire_float("ratio_min", c["ratio_min"])
            nouvelle["risque_max_pct"] = lire_float("risque_max_pct", c["risque_max_pct"])
            nouvelle["journal_max"] = lire_int("journal_max", c["journal_max"])

            for tf in ["M5", "M15", "M30", "H1", "H4"]:
                if f"poids_{tf}" in request.form:
                    nouvelle["poids_tf"][tf] = lire_int(f"poids_{tf}", c["poids_tf"].get(tf, 1))

            for code in ["asie", "londres", "overlap", "newyork", "nuit"]:
                if f"sess_{code}_debut" in request.form:
                    nouvelle["sessions"][code]["debut"] = lire_int(f"sess_{code}_debut", 0)
                    nouvelle["sessions"][code]["fin"] = lire_int(f"sess_{code}_fin", 0)
                if f"sess_{code}_qualite" in request.form:
                    nouvelle["sessions"][code]["qualite"] = request.form.get(f"sess_{code}_qualite")
                if f"sess_{code}_label" in request.form:
                    nouvelle["sessions"][code]["label"] = request.form.get(f"sess_{code}_label", code)

            for pair in nouvelle["paires"]:
                nouvelle["paires"][pair]["actif"] = (f"pair_{pair}_actif" in request.form)

            for f in ["session", "volatilite", "tendance_superieure", "timing", "duplication", "correlation"]:
                nouvelle["filtres_actifs"][f] = (f"filtre_{f}" in request.form)

            if sauver_config(nouvelle):
                message = ("ok", "✅ Configuration enregistrée et prise en compte immédiatement.")
            else:
                message = ("err", "❌ Erreur lors de la sauvegarde.")
        except Exception as e:
            message = ("err", f"❌ Erreur : {e}")

    c = cfg()
    return render_template("config.html", config=c, message=message, now=datetime.now(BENIN).strftime("%H:%M"))


@app.route("/config/reset", methods=["POST"])
def config_reset():
    if not logged():
        return redirect(url_for("index"))
    sauver_config(copy.deepcopy(CONFIG_DEFAUT))
    return redirect(url_for("config_page"))


@app.route("/journal/clear", methods=["POST"])
def journal_clear():
    if not logged():
        return redirect(url_for("index"))
    session["journal"] = []
    return redirect(url_for("dashboard"))


@app.route("/ping")
def ping():
    return "ok", 200


@app.route("/cron")
def cron():
    analyser_tout()
    return "ok", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
