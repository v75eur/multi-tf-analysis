import os, json, hashlib, secrets
from datetime import datetime
from flask import Flask, request, redirect, url_for, session, render_template
import requests
import websocket
import numpy as np

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", secrets.token_hex(32))
ADMIN_HASH = os.getenv("ADMIN_HASH", "d0695d2f4b6487fb81c7047ba01d06d5065aa1a9f18f89633389e1e9b5d85fd6")

# Yahoo (REST) pour EURUSD, GBPUSD, XAUUSD
YAHOO_PAIRS = {"EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "XAUUSD": "GC=F"}
# Deriv (WebSocket) pour V75
DERIV_PAIRS = {"V75": "R_75"}
TIMEFRAMES = {"H1": ("1h", "60d", 3600), "M30": ("30m", "30d", 1800), "M15": ("15m", "15d", 900), "M5": ("5m", "5d", 300)}
PERIODE = 10
LARGEUR_PIVOT = 3
FENETRE_CREUX = 50
DERIV_WS = "wss://api.derivws.com/trading/v1/options/ws/public"

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
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval={interval}&range={range_}"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        data = r.json()["chart"]["result"][0]
        ts = data["timestamp"]
        q = data["indicators"]["quote"][0]
        bougies = []
        for i in range(len(ts)):
            if q["open"][i] and q["high"][i] and q["low"][i] and q["close"][i]:
                bougies.append({"t": ts[i], "open": q["open"][i], "high": q["high"][i], "low": q["low"][i], "close": q["close"][i]})
        return bougies
    except Exception as e:
        print(f"Yahoo {symbol}: {e}")
        return []

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
    if len(bougies) < PERIODE: return None
    clotures = [b["close"] for b in bougies[-PERIODE:]]
    tendance, pct_tendance = calc_tendance(clotures)
    v, r, pa, pv = calc_acheteurs(bougies)
    c, ag = calc_agitation(bougies)
    creux = trouver_creux(bougies)
    sommets = trouver_sommets(bougies)
    divergence = calc_divergence(bougies, creux, sommets)
    return {"tf": tf, "tendance": tendance, "pct_tendance": pct_tendance,
            "vertes": v, "rouges": r, "pct_acheteurs": pa, "pct_vendeurs": pv,
            "chevauchements": c, "pct_agitation": ag, "divergence": divergence,
            "prix": clotures[-1]}

def analyser_paire(pair, source):
    resultats = []
    for tf, (interval, range_, gran) in TIMEFRAMES.items():
        if source == "yahoo":
            bougies = recuperer_bougies_yahoo(YAHOO_PAIRS[pair], interval, range_)
        else:
            bougies = recuperer_bougies_deriv(DERIV_PAIRS[pair], gran, 100)
        r = analyser(tf, bougies)
        if r: resultats.append(r)
    return resultats

def stats_paire(resultats):
    if not resultats: return None
    n = len(resultats)
    pct_h = sum(1 for r in resultats if r["tendance"] == "haussier") / n * 100
    pct_b = sum(1 for r in resultats if r["tendance"] == "baissier") / n * 100
    moy_ach = round(sum(r["pct_acheteurs"] for r in resultats) / n, 1)
    moy_ven = round(sum(r["pct_vendeurs"] for r in resultats) / n, 1)
    moy_agit = round(sum(r["pct_agitation"] for r in resultats) / n, 1)
    moy_tend = round(sum(r["pct_tendance"] for r in resultats) / n, 1)
    div_h = sum(1 for r in resultats if r["divergence"] == "haussiere")
    div_b = sum(1 for r in resultats if r["divergence"] == "baissiere")
    if pct_h > 60: conclusion, couleur = "HAUSSIER", "vert"
    elif pct_b > 60: conclusion, couleur = "BAISSIER", "rouge"
    else: conclusion, couleur = "NEUTRE", "jaune"
    if moy_agit < 40: etat = "Propre"
    elif moy_agit < 70: etat = "Normal"
    else: etat = "Agite"
    if pct_h > 60 and moy_agit < 70: decision = "ACHAT possible"
    elif pct_b > 60 and moy_agit < 70: decision = "VENTE possible"
    else: decision = "ATTENDRE"
    return {"pct_haussier": round(pct_h, 0), "pct_baissier": round(pct_b, 0),
            "moy_acheteurs": moy_ach, "moy_vendeurs": moy_ven,
            "moy_agitation": moy_agit, "moy_tendance": moy_tend,
            "div_haussiere": div_h, "div_baissiere": div_b,
            "conclusion": conclusion, "couleur": couleur,
            "etat": etat, "decision": decision, "n_tf": n}

def analyser_tout():
    res, stats = {}, {}
    for pair in YAHOO_PAIRS:
        res[pair] = analyser_paire(pair, "yahoo")
        stats[pair] = stats_paire(res[pair])
    for pair in DERIV_PAIRS:
        res[pair] = analyser_paire(pair, "deriv")
        stats[pair] = stats_paire(res[pair])
    return res, stats

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
    res, stats = analyser_tout()
    return render_template("dashboard.html", resultats=res, stats=stats, now=datetime.now().strftime("%H:%M"))

@app.route("/ping")
def ping(): return "ok", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
