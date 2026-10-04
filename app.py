import os, json, hashlib, secrets, time
from datetime import datetime
from flask import Flask, request, redirect, url_for, session, render_template
import requests
import websocket
import numpy as np

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", secrets.token_hex(32))
ADMIN_HASH = os.getenv("ADMIN_HASH", "d0695d2f4b6487fb81c7047ba01d06d5065aa1a9f18f89633389e1e9b5d85fd6")

YAHOO_PAIRS = {"EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "XAUUSD": "GC=F"}
DERIV_PAIRS = {"V75": "R_75"}
TIMEFRAMES = {
    "H4": ("1h", "60d", 14400),
    "H1": ("1h", "60d", 3600),
    "M30": ("30m", "30d", 1800),
    "M15": ("15m", "15d", 900),
    "M5": ("5m", "5d", 300)
}
PERIODE = 10
EMA_COURT = 5
EMA_LONG = 25
DERIV_WS = "wss://api.derivws.com/trading/v1/options/ws/public"
CACHE_YAHOO_SEC = 15 * 60
_cache_yahoo = {}

def logged(): return session.get("admin") is True

def recuperer_bougies_deriv(symbol, granularity, count=100):
    try:
        ws = websocket.create_connection(DERIV_WS, timeout=15)
        ws.send(json.dumps({"ticks_history": symbol, "count": count, "end": "latest", "style": "candles", "granularity": granularity}))
        r = json.loads(ws.recv())
        ws.close()
        if "candles" in r:
            return [{"t": c["epoch"], "open": float(c["open"]), "high": float(c["high"]), "low": float(c["low"]), "close": float(c["close"])} for c in r["candles"]]
        return []
    except Exception as e:
        print(f"Deriv {symbol}: {e}")
        return []

def recuperer_bougies_yahoo(symbol, interval, range_):
    now = time.time()
    key = f"{symbol}_{interval}_{range_}"
    if key in _cache_yahoo:
        ts, data = _cache_yahoo[key]
        if now - ts < CACHE_YAHOO_SEC:
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
                bougies.append({"t": ts_list[i], "open": q["open"][i], "high": q["high"][i], "low": q["low"][i], "close": q["close"][i]})
        _cache_yahoo[key] = (now, bougies)
        return bougies
    except Exception as e:
        print(f"Yahoo {symbol}: {e}")
        return []

def calc_ema(clotures, periode):
    if len(clotures) < periode: return None
    k = 2 / (periode + 1)
    ema = np.mean(clotures[:periode])
    for c in clotures[periode:]:
        ema = c * k + ema * (1 - k)
    return round(ema, 5)

def calc_ema_signal(clotures):
    ema5 = calc_ema(clotures, EMA_COURT)
    ema25 = calc_ema(clotures, EMA_LONG)
    if ema5 is None or ema25 is None: return None, None, None
    if ema5 > ema25: sig = "haussier"
    elif ema5 < ema25: sig = "baissier"
    else: sig = "neutre"
    return ema5, ema25, sig

def calc_tendance(clotures):
    x = np.arange(1, len(clotures)+1)
    pente = np.polyfit(x, np.array(clotures), 1)[0]
    pct = min(100, abs(pente)/np.mean(clotures)*10000)
    if pente > 0: return "haussier", round(pct, 1)
    if pente < 0: return "baissier", round(pct, 1)
    return "neutre", 0

def calc_acheteurs(bougies, periode=10):
    d = bougies[-periode:]
    v = sum(1 for b in d if b["close"] > b["open"])
    r = sum(1 for b in d if b["close"] < b["open"])
    t = len(d)
    return v, r, round((v/t)*100, 1), round((r/t)*100, 1)

def calc_agitation(bougies, periode=10):
    d = bougies[-periode:]
    c = 0
    for i in range(1, len(d)):
        if d[i]["low"] <= d[i-1]["high"] and d[i]["high"] >= d[i-1]["low"]:
            c += 1
    return c, round((c/len(d))*100, 1)

def trouver_creux(bougies, fenetre=50, largeur=3):
    creux = []
    n = min(len(bougies)-largeur, fenetre)
    for i in range(largeur, n):
        if all(bougies[j]["low"] > bougies[i]["low"] for j in range(i-largeur, i+largeur+1) if j != i):
            creux.append(i)
    return creux

def trouver_sommets(bougies, fenetre=50, largeur=3):
    sommets = []
    n = min(len(bougies)-largeur, fenetre)
    for i in range(largeur, n):
        if all(bougies[j]["high"] < bougies[i]["high"] for j in range(i-largeur, i+largeur+1) if j != i):
            sommets.append(i)
    return sommets

def calc_divergence(bougies, creux, sommets):
    div_h = False
    if len(creux) >= 2:
        c1, c2 = creux[-2], creux[-1]
        if bougies[c2]["low"] < bougies[c1]["low"]:
            r1 = calc_acheteurs(bougies[:c1+1])[2]
            r2 = calc_acheteurs(bougies[:c2+1])[2]
            if r2 > r1: div_h = True
    div_b = False
    if len(sommets) >= 2:
        s1, s2 = sommets[-2], sommets[-1]
        if bougies[s2]["high"] > bougies[s1]["high"]:
            r1 = calc_acheteurs(bougies[:s1+1])[2]
            r2 = calc_acheteurs(bougies[:s2+1])[2]
            if r2 < r1: div_b = True
    if div_h: return "haussiere"
    if div_b: return "baissiere"
    return "aucune"

def analyser(tf, bougies):
    if len(bougies) < 30: return None
    clotures = [b["close"] for b in bougies]
    clotures_10 = clotures[-PERIODE:]
    tendance, pct_tendance = calc_tendance(clotures_10)
    v, r, pa, pv = calc_acheteurs(bougies)
    c, ag = calc_agitation(bougies)
    creux = trouver_creux(bougies)
    sommets = trouver_sommets(bougies)
    divergence = calc_divergence(bougies, creux, sommets)
    ema5, ema25, ema_sig = calc_ema_signal(clotures)
    return {
        "tf": tf, "tendance": tendance, "pct_tendance": pct_tendance,
        "pct_acheteurs": pa, "pct_vendeurs": pv,
        "pct_agitation": ag,
        "divergence": divergence,
        "ema5": ema5, "ema25": ema25, "ema_signal": ema_sig,
        "prix": clotures[-1]
    }

def analyser_paire(pair, source):
    resultats = {}
    for tf, (interval, range_, gran) in TIMEFRAMES.items():
        if source == "yahoo":
            bougies = recuperer_bougies_yahoo(YAHOO_PAIRS[pair], interval, range_)
        else:
            bougies = recuperer_bougies_deriv(DERIV_PAIRS[pair], gran, 100)
        r = analyser(tf, bougies)
        if r: resultats[tf] = r
    return resultats

def conditions_paire(tfs):
    if not all(k in tfs for k in ["H1", "M30", "M15", "M5"]):
        return {"signal": "?", "couleur": "jaune", "conseil": "Données manquantes", "details": {}}
    h1, m30, m15, m5 = tfs["H1"], tfs["M30"], tfs["M15"], tfs["M5"]
    h4 = tfs.get("H4")
    details = {}
    cond1 = all(t["tendance"] == "haussier" for t in [h1, m30, m15])
    cond2 = all(t["pct_acheteurs"] > t["pct_vendeurs"] for t in [h1, m30, m15])
    cond3 = m5["ema_signal"] == "haussier"
    details["H1 ↑"] = h1["tendance"] == "haussier"
    details["M30 ↑"] = m30["tendance"] == "haussier"
    details["M15 ↑"] = m15["tendance"] == "haussier"
    details["Ach>Ven H1"] = h1["pct_acheteurs"] > h1["pct_vendeurs"]
    details["Ach>Ven M30"] = m30["pct_acheteurs"] > m30["pct_vendeurs"]
    details["Ach>Ven M15"] = m15["pct_acheteurs"] > m15["pct_vendeurs"]
    details["M5 EMA↑"] = cond3
    bonus = h4 and h4["tendance"] == "haussier" if h4 else False
    details["H4 ↑"] = bonus

    if cond1 and cond2 and cond3:
        conseil = "✅ Signal ACHAT fiable. Entre à la clôture M5."
        if bonus: conseil = "⭐ Signal ACHAT FORT (H4 confirme). Entre à la clôture M5."
        return {"signal": "ACHAT", "couleur": "vert", "bonus": bonus, "conseil": conseil, "details": details}

    cond1b = all(t["tendance"] == "baissier" for t in [h1, m30, m15])
    cond2b = all(t["pct_vendeurs"] > t["pct_acheteurs"] for t in [h1, m30, m15])
    cond3b = m5["ema_signal"] == "baissier"
    details2 = {
        "H1 ↓": h1["tendance"] == "baissier",
        "M30 ↓": m30["tendance"] == "baissier",
        "M15 ↓": m15["tendance"] == "baissier",
        "Ven>Ach H1": h1["pct_vendeurs"] > h1["pct_acheteurs"],
        "Ven>Ach M30": m30["pct_vendeurs"] > m30["pct_acheteurs"],
        "Ven>Ach M15": m15["pct_vendeurs"] > m15["pct_acheteurs"],
        "M5 EMA↓": cond3b,
    }
    bonus2 = h4 and h4["tendance"] == "baissier" if h4 else False
    details2["H4 ↓"] = bonus2

    if cond1b and cond2b and cond3b:
        conseil = "✅ Signal VENTE fiable. Entre à la clôture M5."
        if bonus2: conseil = "⭐ Signal VENTE FORT (H4 confirme). Entre à la clôture M5."
        return {"signal": "VENTE", "couleur": "rouge", "bonus": bonus2, "conseil": conseil, "details": details2}

    # Compte les conditions remplies
    ok = sum(1 for v in details.values() if v)
    total = len(details)
    conseil = f"⏸️ Attendre. {ok}/{total} conditions remplies."
    return {"signal": "ATTENDRE", "couleur": "jaune", "bonus": False, "conseil": conseil, "details": details}

def analyser_tout():
    res, conds = {}, {}
    for pair in YAHOO_PAIRS:
        res[pair] = analyser_paire(pair, "yahoo")
        conds[pair] = conditions_paire(res[pair])
    for pair in DERIV_PAIRS:
        res[pair] = analyser_paire(pair, "deriv")
        conds[pair] = conditions_paire(res[pair])
    return res, conds

@app.route("/")
def index():
    if logged(): return redirect(url_for("dashboard"))
    return render_template("login.html", error=None)

@app.route("/login", methods=["POST"])
def login():
    if hashlib.sha256(request.form.get("password","").encode()).hexdigest() == ADMIN_HASH:
        session["admin"] = True
        return redirect(url_for("dashboard"))
    return render_template("login.html", error="Mot de passe incorrect")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))

@app.route("/dashboard")
def dashboard():
    if not logged(): return redirect(url_for("index"))
    res, conds = analyser_tout()
    return render_template("dashboard.html", resultats=res, conditions=conds, now=datetime.now().strftime("%H:%M"))

@app.route("/ping")
def ping(): return "ok", 200

@app.route("/cron")
def cron():
    analyser_tout()
    return "ok", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
