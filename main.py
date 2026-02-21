"""
STOCKPILE — Backend API v3
==========================
- Petróleo y gas: inventarios reales EIA (semanales)
- Soja, trigo, maíz: stocks reales USDA NASS (trimestrales)
- Metales: precios futuros yfinance (proxy)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import yfinance as yf
import httpx
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="STOCKPILE API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

EIA_API_KEY  = "cj21fTFCc9IZm1YvF4WP2cvOuSH9eIeQDkvDfwyK"
USDA_API_KEY = "1D9FB410-B3F4-3935-AD81-A5C8584B2BA0"
EIA_BASE     = "https://api.eia.gov/v2"
USDA_BASE    = "https://quickstats.nass.usda.gov/api/api_GET"

# ── EIA CONFIG ────────────────────────────────────────────────────────────────
EIA_CONFIG = {
    "crude": {
        "name": "Petróleo Crudo",
        "category": "ENERGÍA",
        "cat": "energia",
        "unit": "M barriles",
        "accent": "#ff6b35",
        "source": "EIA",
        "context": "Cushing, Oklahoma — hub de referencia WTI",
        "eia_route": "petroleum/stoc/wstk",
        "facets": {"product": "EPC0", "duoarea": "YCUOK"},
        "scale": 1000,   # MBBL → millones de barriles
    },
    "gas": {
        "name": "Gas Natural",
        "category": "ENERGÍA",
        "cat": "energia",
        "unit": "Bcf",
        "accent": "#ff8c5a",
        "source": "EIA",
        "context": "Almacenamiento subterráneo Lower 48 EEUU",
        "eia_route": "natural-gas/stor/wkly",
        "facets": {"process": "SWO", "duoarea": "R48"},
        "scale": 1,
    },
}

# ── USDA CONFIG ───────────────────────────────────────────────────────────────
USDA_CONFIG = {
    "soy": {
        "name": "Soja",
        "category": "GRANOS",
        "cat": "granos",
        "unit": "M bushels",
        "accent": "#4ade80",
        "source": "USDA NASS",
        "context": "Stocks totales EEUU — trimestral",
        "commodity_desc": "SOYBEANS",
        "short_desc": "SOYBEANS - STOCKS, MEASURED IN BU",
        "scale": 1_000_000,  # bushels → millones de bushels
    },
    "wheat": {
        "name": "Trigo",
        "category": "GRANOS",
        "cat": "granos",
        "unit": "M bushels",
        "accent": "#fbbf24",
        "source": "USDA NASS",
        "context": "Stocks totales EEUU — trimestral",
        "commodity_desc": "WHEAT",
        "short_desc": "WHEAT - STOCKS, MEASURED IN BU",
        "scale": 1_000_000,
    },
    "corn": {
        "name": "Maíz",
        "category": "GRANOS",
        "cat": "granos",
        "unit": "M bushels",
        "accent": "#fde047",
        "source": "USDA NASS",
        "context": "Stocks totales EEUU — trimestral",
        "commodity_desc": "CORN",
        "short_desc": "CORN, GRAIN - STOCKS, MEASURED IN BU",
        "scale": 1_000_000,
    },
}

# ── PRICE CONFIG (proxy para metales) ─────────────────────────────────────────
PRICE_CONFIG = {
    "copper":   {"name": "Cobre",    "ticker": "HG=F",  "category": "METALES", "cat": "metales", "unit": "USD/lb",    "scale": 1,   "accent": "#e8a87c", "source": "COMEX", "context": "Precio futuro COMEX (proxy inventario)"},
    "gold":     {"name": "Oro",      "ticker": "GC=F",  "category": "METALES", "cat": "metales", "unit": "USD/oz",    "scale": 1,   "accent": "#f5c842", "source": "COMEX", "context": "Precio futuro COMEX (proxy inventario)"},
    "silver":   {"name": "Plata",    "ticker": "SI=F",  "category": "METALES", "cat": "metales", "unit": "USD/oz",    "scale": 1,   "accent": "#c0c0c0", "source": "COMEX", "context": "Precio futuro COMEX (proxy inventario)"},
    "aluminum": {"name": "Aluminio", "ticker": "ALI=F", "category": "METALES", "cat": "metales", "unit": "USD/lb",   "scale": 1,   "accent": "#a8b8d0", "source": "COMEX", "context": "Precio futuro COMEX (proxy inventario)"},
    "lithium":  {"name": "Litio",    "ticker": "LIT",   "category": "METALES", "cat": "metales", "unit": "USD (ETF)","scale": 1,   "accent": "#c084fc", "source": "NYSE",  "context": "Global X Lithium ETF (proxy inventario)"},
}


class CommodityData(BaseModel):
    id: str
    name: str
    category: str
    cat: str
    unit: str
    accent: str
    source: str
    context: str
    current: float
    avg5y: float
    min5y: float
    max5y: float
    weeklyChange: float
    percentile: int
    signal: str
    history: List[float]
    dataType: str
    lastUpdated: str


# ── HELPERS ───────────────────────────────────────────────────────────────────

def compute_metrics(values: List[float], config: dict, data_type: str, commodity_id: str) -> CommodityData:
    current       = values[-1]
    avg5y         = sum(values) / len(values)
    min5y         = min(values)
    max5y         = max(values)
    weekly_change = round(((current - values[-2]) / values[-2]) * 100, 2) if len(values) >= 2 else 0.0
    percentile    = int(((current - min5y) / (max5y - min5y)) * 100) if max5y != min5y else 50
    ratio         = current / avg5y
    signal        = "low" if ratio < 0.85 else "high" if ratio > 1.15 else "normal"
    history       = [round(v, 3) for v in values[-10:]]

    return CommodityData(
        id=commodity_id,
        name=config["name"],
        category=config["category"],
        cat=config["cat"],
        unit=config["unit"],
        accent=config["accent"],
        source=config["source"],
        context=config["context"],
        current=round(current, 3),
        avg5y=round(avg5y, 3),
        min5y=round(min5y, 3),
        max5y=round(max5y, 3),
        weeklyChange=weekly_change,
        percentile=percentile,
        signal=signal,
        history=history,
        dataType=data_type,
        lastUpdated=datetime.now().isoformat(),
    )


# ── EIA FETCH ─────────────────────────────────────────────────────────────────

def fetch_eia(commodity_id: str) -> CommodityData:
    config = EIA_CONFIG[commodity_id]
    logger.info(f"Fetching EIA for {commodity_id}")

    end_date   = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365 * 5)).strftime("%Y-%m-%d")

    params = {
        "api_key":            EIA_API_KEY,
        "frequency":          "weekly",
        "data[]":             "value",
        "start":              start_date,
        "end":                end_date,
        "sort[0][column]":    "period",
        "sort[0][direction]": "asc",
        "length":             300,
    }
    for k, v in config["facets"].items():
        params[f"facets[{k}][]"] = v

    with httpx.Client(timeout=30, verify=False) as client:
        resp = client.get(f"{EIA_BASE}/{config['eia_route']}/data/", params=params)
        resp.raise_for_status()
        data = resp.json()

    series = data.get("response", {}).get("data", [])
    if not series:
        raise ValueError(f"EIA returned no data for {commodity_id}")

    values = []
    for row in series:
        val = row.get("value")
        if val is not None:
            try:
                values.append(float(val) / config["scale"])
            except (ValueError, TypeError):
                pass

    if len(values) < 5:
        raise ValueError(f"Not enough EIA data: {len(values)} points")

    return compute_metrics(values, config, "inventory", commodity_id)


# ── USDA FETCH ────────────────────────────────────────────────────────────────

def fetch_usda(commodity_id: str) -> CommodityData:
    config = USDA_CONFIG[commodity_id]
    logger.info(f"Fetching USDA for {commodity_id}")

    current_year = datetime.now().year
    start_year   = current_year - 5

    params = {
        "key":               USDA_API_KEY,
        "commodity_desc":    config["commodity_desc"],
        "statisticcat_desc": "STOCKS",
        "agg_level_desc":    "NATIONAL",
        "format":            "JSON",
        "year__GE":          str(start_year),
    }

    with httpx.Client(timeout=30, verify=False) as client:
        resp = client.get(USDA_BASE, params=params)
        resp.raise_for_status()
        data = resp.json()

    items = data.get("data", [])

    # Filtrar solo stocks totales (no on-farm, off-farm por separado)
    target = config["short_desc"]
    filtered = [i for i in items if i.get("short_desc") == target]

    if not filtered:
        raise ValueError(f"USDA returned no data for {commodity_id}")

    # Ordenar por año y período
    period_order = {"FIRST OF MAR": 1, "FIRST OF JUN": 2, "FIRST OF SEP": 3, "FIRST OF DEC": 4}
    filtered.sort(key=lambda x: (int(x.get("year", 0)), period_order.get(x.get("reference_period_desc", ""), 0)))

    values = []
    for item in filtered:
        raw = item.get("Value", "").replace(",", "")
        try:
            values.append(float(raw) / config["scale"])
        except (ValueError, TypeError):
            pass

    if len(values) < 4:
        raise ValueError(f"Not enough USDA data: {len(values)} points")

    return compute_metrics(values, config, "inventory", commodity_id)


# ── YFINANCE FETCH ────────────────────────────────────────────────────────────

def get_close(df, ticker):
    if df.empty:
        raise ValueError(f"Empty DataFrame for {ticker}")
    if hasattr(df.columns, 'levels'):
        if ('Close', ticker) in df.columns:
            return df[('Close', ticker)].dropna()
        close_cols = [c for c in df.columns if c[0] == 'Close']
        if close_cols:
            return df[close_cols[0]].dropna()
    if 'Close' in df.columns:
        return df['Close'].dropna()
    raise ValueError(f"No Close column for {ticker}")


def fetch_price(commodity_id: str) -> CommodityData:
    config = PRICE_CONFIG[commodity_id]
    ticker = config["ticker"]
    scale  = config["scale"]
    logger.info(f"Fetching price for {commodity_id} ({ticker})")

    end       = datetime.now()
    start_5y  = end - timedelta(days=365 * 5)
    start_10w = end - timedelta(weeks=11)

    data_5y  = yf.download(ticker, start=start_5y,  end=end, progress=False, auto_adjust=True)
    data_10w = yf.download(ticker, start=start_10w, end=end, progress=False, interval="1wk", auto_adjust=True)

    if data_5y.empty:
        raise ValueError(f"No yfinance data for {ticker}")

    close_5y  = get_close(data_5y, ticker) / scale
    close_10w = get_close(data_10w, ticker) / scale if not data_10w.empty else close_5y.tail(10)

    values = [float(v) for v in close_5y.tolist()]
    result = compute_metrics(values, config, "price", commodity_id)

    # Override history with weekly data for better sparkline
    result.history = [round(float(v), 2) for v in close_10w.tail(10).tolist()]
    return result


# ── ENDPOINTS ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "3.0.0", "timestamp": datetime.now().isoformat()}


@app.get("/commodities", response_model=List[CommodityData])
def get_all_commodities():
    results = []

    for cid in EIA_CONFIG:
        try:
            results.append(fetch_eia(cid))
        except Exception as e:
            logger.error(f"EIA failed for {cid}: {e}")

    for cid in USDA_CONFIG:
        try:
            results.append(fetch_usda(cid))
        except Exception as e:
            logger.error(f"USDA failed for {cid}: {e}")

    for cid in PRICE_CONFIG:
        try:
            results.append(fetch_price(cid))
        except Exception as e:
            logger.error(f"Price failed for {cid}: {e}")

    if not results:
        raise HTTPException(status_code=502, detail="No se pudo obtener datos")

    return results


@app.get("/commodity/{commodity_id}", response_model=CommodityData)
def get_commodity(commodity_id: str):
    if commodity_id in EIA_CONFIG:
        return fetch_eia(commodity_id)
    if commodity_id in USDA_CONFIG:
        return fetch_usda(commodity_id)
    if commodity_id in PRICE_CONFIG:
        return fetch_price(commodity_id)
    raise HTTPException(status_code=404, detail=f"'{commodity_id}' no encontrado")


# ── CLIMA ─────────────────────────────────────────────────────────────────────

GRAIN_ZONES = {
    "soy":   [
        {"name": "Corn Belt (Iowa)", "lat": 42.0, "lon": -93.6, "role": "Principal zona productora EEUU"},
        {"name": "Pampa Húmeda (Bs As)", "lat": -34.6, "lon": -58.4, "role": "Principal zona productora Argentina"},
    ],
    "corn":  [
        {"name": "Corn Belt (Iowa)", "lat": 42.0, "lon": -93.6, "role": "Principal zona productora EEUU"},
        {"name": "Pampa Húmeda (Bs As)", "lat": -34.6, "lon": -58.4, "role": "Principal zona productora Argentina"},
    ],
    "wheat": [
        {"name": "Great Plains (Kansas)", "lat": 38.5, "lon": -98.0, "role": "Principal zona productora EEUU"},
        {"name": "Mar Negro (Ucrania)", "lat": 49.0, "lon": 32.0, "role": "Principal exportador mundial"},
    ],
}

CROP_CRITICAL_MONTHS = {
    "soy":   {"norte": [6, 7, 8], "sur": [12, 1, 2]},   # Jun-Ago EEUU, Dic-Feb ARG
    "corn":  {"norte": [6, 7, 8], "sur": [12, 1, 2]},
    "wheat": {"norte": [4, 5, 6], "sur": [9, 10, 11]},  # Abr-Jun EEUU, Sep-Nov ARG
}


class ZoneClimate(BaseModel):
    name: str
    lat: float
    lon: float
    role: str
    temp_c: float
    precip_mm: float          # últimos 7 días
    precip_forecast_mm: float # próximos 7 días
    temp_forecast_c: float
    stress_signal: str        # "ok" | "drought" | "flood" | "cold" | "heat"
    stress_label: str
    in_critical_period: bool


class GrainClimate(BaseModel):
    commodity_id: str
    zones: List[ZoneClimate]
    overall_signal: str       # "favorable" | "neutral" | "stress"
    summary: str
    lastUpdated: str


def classify_stress(temp_c: float, precip_7d: float, precip_forecast: float, commodity: str, is_critical: bool) -> tuple:
    """Clasifica el nivel de estrés climático para un cultivo."""
    # Umbrales de estrés
    drought_threshold  = 5   # mm en 7 días = sequía
    flood_threshold    = 80  # mm en 7 días = exceso hídrico
    heat_threshold     = 35  # °C = estrés por calor
    cold_threshold     = 2   # °C = riesgo de helada

    if temp_c >= heat_threshold:
        return "heat", f"Estrés calórico ({temp_c:.0f}°C)"
    if temp_c <= cold_threshold:
        return "cold", f"Riesgo de helada ({temp_c:.0f}°C)"
    if precip_7d <= drought_threshold and precip_forecast <= drought_threshold:
        return "drought", f"Sequía ({precip_7d:.0f}mm últimos 7d)"
    if precip_7d >= flood_threshold:
        return "flood", f"Exceso hídrico ({precip_7d:.0f}mm últimos 7d)"
    return "ok", f"Condiciones normales ({temp_c:.0f}°C, {precip_7d:.0f}mm)"


def fetch_zone_climate(zone: dict, commodity: str) -> ZoneClimate:
    lat, lon = zone["lat"], zone["lon"]

    # Open-Meteo: sin API key, completamente gratuito
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&daily=temperature_2m_max,precipitation_sum"
        f"&past_days=7&forecast_days=7"
        f"&timezone=auto"
    )

    with httpx.Client(timeout=15, verify=False) as client:
        resp = client.get(url)
        resp.raise_for_status()
        data = resp.json()

    daily = data.get("daily", {})
    temps  = daily.get("temperature_2m_max", [])
    precip = daily.get("precipitation_sum", [])

    # Dividir entre pasado (7 días) y futuro (7 días)
    past_temps   = [t for t in temps[:7]  if t is not None]
    future_temps = [t for t in temps[7:]  if t is not None]
    past_precip  = [p for p in precip[:7] if p is not None]
    future_precip= [p for p in precip[7:] if p is not None]

    temp_c             = sum(past_temps)   / len(past_temps)   if past_temps   else 20.0
    temp_forecast_c    = sum(future_temps) / len(future_temps) if future_temps else 20.0
    precip_mm          = sum(past_precip)
    precip_forecast_mm = sum(future_precip)

    # ¿Estamos en período crítico?
    current_month = datetime.now().month
    is_north = lat > 0
    key = "norte" if is_north else "sur"
    critical_months = CROP_CRITICAL_MONTHS.get(commodity, {}).get(key, [])
    in_critical = current_month in critical_months

    stress_signal, stress_label = classify_stress(temp_c, precip_mm, precip_forecast_mm, commodity, in_critical)

    return ZoneClimate(
        name=zone["name"],
        lat=lat,
        lon=lon,
        role=zone["role"],
        temp_c=round(temp_c, 1),
        precip_mm=round(precip_mm, 1),
        precip_forecast_mm=round(precip_forecast_mm, 1),
        temp_forecast_c=round(temp_forecast_c, 1),
        stress_signal=stress_signal,
        stress_label=stress_label,
        in_critical_period=in_critical,
    )


def fetch_grain_climate(commodity_id: str) -> GrainClimate:
    zones_config = GRAIN_ZONES.get(commodity_id, [])
    zones = []
    for z in zones_config:
        try:
            zones.append(fetch_zone_climate(z, commodity_id))
        except Exception as e:
            logger.error(f"Climate failed for {z['name']}: {e}")

    if not zones:
        raise ValueError(f"No climate data for {commodity_id}")

    # Señal general
    signals = [z.stress_signal for z in zones]
    if any(s in ("drought", "heat", "cold", "flood") for s in signals):
        critical_zones = [z for z in zones if z.stress_signal != "ok"]
        if any(z.in_critical_period for z in critical_zones):
            overall = "stress"
            summary = f"Condiciones de estrés en período crítico: {', '.join(z.name for z in critical_zones)}"
        else:
            overall = "neutral"
            summary = f"Estrés fuera de período crítico: {', '.join(z.name for z in critical_zones)}"
    else:
        overall = "favorable"
        summary = "Condiciones favorables en todas las zonas monitoreadas"

    return GrainClimate(
        commodity_id=commodity_id,
        zones=zones,
        overall_signal=overall,
        summary=summary,
        lastUpdated=datetime.now().isoformat(),
    )


@app.get("/climate/{commodity_id}", response_model=GrainClimate)
def get_climate(commodity_id: str):
    if commodity_id not in GRAIN_ZONES:
        raise HTTPException(status_code=404, detail=f"No hay zonas climáticas para '{commodity_id}'")
    return fetch_grain_climate(commodity_id)


@app.get("/climate", response_model=List[GrainClimate])
def get_all_climate():
    results = []
    for cid in GRAIN_ZONES:
        try:
            results.append(fetch_grain_climate(cid))
        except Exception as e:
            logger.error(f"Climate failed for {cid}: {e}")
    return results


# ── NOTICIAS ──────────────────────────────────────────────────────────────────

NEWS_API_KEY = "bf037cc81d33433c8e32a7791b541b4d"
NEWS_BASE    = "https://newsapi.org/v2/everything"

COMMODITY_KEYWORDS = {
    "crude":    "crude oil WTI petroleum inventory",
    "gas":      "natural gas storage inventory",
    "copper":   "copper LME warehouse stocks",
    "gold":     "gold prices market",
    "silver":   "silver prices market",
    "aluminum": "aluminum aluminium LME stocks",
    "lithium":  "lithium battery market supply",
    "soy":      "soybean soy USDA stocks harvest",
    "wheat":    "wheat USDA stocks harvest supply",
    "corn":     "corn maize USDA stocks harvest",
}


class NewsArticle(BaseModel):
    title: str
    source: str
    url: str
    publishedAt: str
    description: str


class CommodityNews(BaseModel):
    commodity_id: str
    articles: List[NewsArticle]
    lastUpdated: str


def fetch_news(commodity_id: str) -> CommodityNews:
    keywords = COMMODITY_KEYWORDS.get(commodity_id, commodity_id)
    logger.info(f"Fetching news for {commodity_id}")

    params = {
        "apiKey":   NEWS_API_KEY,
        "q":        keywords,
        "language": "en",
        "sortBy":   "publishedAt",
        "pageSize": 5,
    }

    with httpx.Client(timeout=15, verify=False) as client:
        resp = client.get(NEWS_BASE, params=params)
        resp.raise_for_status()
        data = resp.json()

    articles = []
    for a in data.get("articles", [])[:5]:
        if a.get("title") and a.get("url"):
            articles.append(NewsArticle(
                title=a.get("title", "")[:120],
                source=a.get("source", {}).get("name", ""),
                url=a.get("url", ""),
                publishedAt=a.get("publishedAt", ""),
                description=(a.get("description") or "")[:200],
            ))

    return CommodityNews(
        commodity_id=commodity_id,
        articles=articles,
        lastUpdated=datetime.now().isoformat(),
    )


@app.get("/news/{commodity_id}", response_model=CommodityNews)
def get_news(commodity_id: str):
    all_ids = list(EIA_CONFIG.keys()) + list(USDA_CONFIG.keys()) + list(PRICE_CONFIG.keys())
    if commodity_id not in all_ids:
        raise HTTPException(status_code=404, detail=f"'{commodity_id}' no encontrado")
    return fetch_news(commodity_id)
