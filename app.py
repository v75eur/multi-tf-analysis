import os, json, hashlib, secrets, time
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
# tf: (interval Yahoo, période Yahoo, durée d'une bougie en secondes)
TIMEFRAMES = {
    "H4": ("1h", "60d", 14400),
    "H1": ("1h", "60d", 3600),
    "M30": ("30m", "30d", 1800),
    "M15": ("15m", "15d", 900),
    "M5": ("5m", "5d", 300),
}
PERIODE = 10
EMA_COURT = 5
EMA_LONG = 25
DERIV_WS = "wss://api.derivws.com/trading/v1/options/ws/public"
CACHE_SEC = {"1h": 300, "30m": 180, "15m": 120, "5m": 60}
_cache_yahoo = {}


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
    return {
        "tf": tf, "tendance": tendance, "pct_tendance": pct_tendance,
        "pct_acheteurs": pa, "pct_vendeurs": pv,
        "pct_agitation": ag,
        "divergence": divergence,
        "ema5": ema5, "ema25": ema25, "ema_signal": ema_sig, "ema_depuis": ema_depuis,
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


# ───────────── SIGNAL ─────────────

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
            conseil = f"✅ Signal {nom} fiable. Entre à la clôture M5."
            if bonus:
                conseil = f"⭐ Signal {nom} FORT (H4 confirme). Entre à la clôture M5."
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

    if ferme:
        return {**base, "signal": "FERMÉ", "couleur": "gris", "bonus": False,
                "conseil": "🌙 Marché fermé : ces chiffres viennent de la dernière séance. N'entre pas. "
                           f"(Dernier verdict : {base['signal']})"}
    return base


# ───────────── ANALYSE D'UNE PAIRE ─────────────

def analyser_paire(pair, source):
    series, resultats = {}, {}
    for tf, (interval, range_, gran) in TIMEFRAMES.items():
        if source == "yahoo":
            brut = recuperer_bougies_yahoo(YAHOO_PAIRS[pair], interval, range_)
            if tf == "H4":
                brut = regrouper(brut, gran)
        else:
            brut = recuperer_bougies_deriv(DERIV_PAIRS[pair], gran)
        bougies = bougies_fermees(brut, gran)
        series[tf] = bougies
        r = analyser(tf, bougies)
        if r:
            resultats[tf] = r

    fr = fraicheur(series.get("M5"))
    stats = calc_stats(pair, series.get("H1", []), resultats)
    cond = conditions_paire(resultats, fr["etat"] == "ferme")
    prix = fp(pair, resultats["M5"]["prix"]) if "M5" in resultats else "—"
    return {"tfs": resultats, "stats": stats, "fraicheur": fr, "cond": cond, "prix": prix}


def analyser_securise(pair, source):
    try:
        return analyser_paire(pair, source)
    except Exception as e:
        print(f"Erreur {pair}: {e}")
        return {"tfs": {}, "stats": None,
                "fraicheur": {"texte": "Erreur de chargement", "etat": "ferme"},
                "cond": conditions_paire({}), "prix": "—"}


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
