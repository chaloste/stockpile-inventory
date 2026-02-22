"""
STOCKPILE — Backend API v3
==========================
- Petróleo y gas: inventarios reales EIA (semanales)
- Soja, trigo, maíz: stocks reales USDA NASS (trimestrales)
- Metales: precios futuros yfinance (proxy)
"""

import os
import time
import json
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

EIA_API_KEY  = os.environ.get("EIA_API_KEY", "")
USDA_API_KEY = os.environ.get("USDA_API_KEY", "")
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
    # Análisis estacional (D)
    seasonal_avg: float = 0.0          # promedio histórico para esta época del año
    seasonal_diff_pct: float = 0.0     # % diferencia vs mismo período año anterior
    seasonal_label: str = ""           # ej: "23% por debajo del promedio estacional"
    yoy_change_pct: float = 0.0        # variación year-over-year
    # Correlación precio/inventario (B)
    price_correlation: str = ""        # descripción textual de correlación histórica
    similar_periods: List[str] = []    # períodos históricos similares


# ── HELPERS ───────────────────────────────────────────────────────────────────

def compute_seasonal(values: List[float], dates: List[datetime], current: float) -> dict:
    """Calcula análisis estacional: compara valor actual con mismo período histórico."""
    if len(values) < 52 or not dates:
        return {"seasonal_avg": 0.0, "seasonal_diff_pct": 0.0, "seasonal_label": "", "yoy_change_pct": 0.0}

    now = datetime.now()
    current_week = now.isocalendar()[1]
    current_month = now.month

    # Valores del mismo período en años anteriores (±2 semanas)
    seasonal_values = []
    for i, d in enumerate(dates):
        week_diff = abs(d.isocalendar()[1] - current_week)
        if week_diff <= 2 and d.year < now.year:
            seasonal_values.append(values[i])

    if not seasonal_values:
        # Fallback: mismo mes
        seasonal_values = [v for i, v in enumerate(values) if dates[i].month == current_month and dates[i].year < now.year]

    if not seasonal_values:
        return {"seasonal_avg": 0.0, "seasonal_diff_pct": 0.0, "seasonal_label": "", "yoy_change_pct": 0.0}

    seasonal_avg = sum(seasonal_values) / len(seasonal_values)
    seasonal_diff = ((current - seasonal_avg) / seasonal_avg) * 100 if seasonal_avg != 0 else 0.0

    # Year-over-year: comparar con hace ~52 semanas
    yoy_change = 0.0
    if len(values) >= 52:
        prev_year_val = values[-52]
        yoy_change = ((current - prev_year_val) / prev_year_val) * 100 if prev_year_val != 0 else 0.0

    # Label descriptivo
    direction = "por encima" if seasonal_diff > 0 else "por debajo"
    abs_diff = abs(round(seasonal_diff, 1))
    if abs_diff < 3:
        label = "En línea con el promedio estacional"
    else:
        label = f"{abs_diff}% {direction} del promedio estacional"

    return {
        "seasonal_avg": round(seasonal_avg, 3),
        "seasonal_diff_pct": round(seasonal_diff, 1),
        "seasonal_label": label,
        "yoy_change_pct": round(yoy_change, 1),
    }


def compute_price_correlation(commodity_id: str, percentile: int, signal: str) -> dict:
    """Genera descripción de correlación histórica precio/inventario."""
    correlations = {
        "crude": {
            "low":    "Históricamente, niveles bajos en Cushing preceden subas de precio del WTI de 8-15% en 4-8 semanas.",
            "high":   "Niveles altos de inventario en Cushing suelen presionar el precio del WTI a la baja en 6-10 semanas.",
            "normal": "Inventarios en rango normal. Sin señal clara de presión sobre el precio del WTI.",
        },
        "gas": {
            "low":    "Stocks bajos de gas natural históricamente disparan el precio del Henry Hub, especialmente en invierno.",
            "high":   "Exceso de inventario de gas suele deprimir el Henry Hub. El mercado descuenta la sobreoferta.",
            "normal": "Stocks en rango. El precio del gas responderá principalmente a condiciones climáticas.",
        },
        "soy": {
            "low":    "Stocks bajos de soja USDA históricamente correlacionan con rallies del CBOT de 10-20% en el trimestre siguiente.",
            "high":   "Abundancia de soja presiona los precios del CBOT. Los exportadores tienen menos urgencia de comprar.",
            "normal": "Stocks equilibrados. El precio responderá a producción de Brasil/Argentina y demanda china.",
        },
        "wheat": {
            "low":    "Reservas bajas de trigo históricamente generan volatilidad alcista, especialmente ante shocks climáticos.",
            "high":   "Exceso de trigo limita el upside del precio. Los compradores esperan mejores condiciones.",
            "normal": "Stocks normales de trigo. El precio seguirá factores geopolíticos (Mar Negro) y climáticos.",
        },
        "corn": {
            "low":    "Stocks bajos de maíz USDA históricamente preceden subas del CBOT, con mayor impacto en verano boreal.",
            "high":   "Abundancia de maíz pesa sobre el CBOT. La demanda de etanol puede absorber parte del exceso.",
            "normal": "Inventarios equilibrados. El precio del maíz seguirá la demanda de feed y etanol.",
        },
        "copper": {
            "low":    "Precios bajos de cobre históricamente preceden recuperaciones cuando el ciclo industrial se reactiva.",
            "high":   "Precios altos del cobre reflejan tensión en suministro. Suelen atraer producción nueva en 12-18 meses.",
            "normal": "Precio del cobre en rango. Seguirá el ciclo económico global y la demanda de China.",
        },
        "gold": {
            "low":    "Precio bajo del oro históricamente atrae demanda de bancos centrales y fondos como reserva de valor.",
            "high":   "Oro en máximos refleja búsqueda de refugio. Suele corregir cuando la aversión al riesgo baja.",
            "normal": "Oro en rango. Responderá a tasas reales de EEUU y fortaleza del dólar.",
        },
        "silver": {
            "low":    "Plata barata históricamente ofrece valor relativo frente al oro. El ratio oro/plata es clave.",
            "high":   "Plata en máximos: combinación de demanda industrial (solar, EV) y demanda de inversión.",
            "normal": "Plata en rango. Seguirá al oro y a la demanda industrial de semiconductores y energía solar.",
        },
        "aluminum": {
            "low":    "Precio bajo del aluminio históricamente precede subas cuando la producción china se restringe.",
            "high":   "Aluminio caro presiona a industrias del packaging y automotriz. Suele incentivar sustitución.",
            "normal": "Aluminio en rango. Seguirá costos energéticos (principal input) y demanda de construcción.",
        },
        "lithium": {
            "low":    "ETF de litio bajo históricamente precede recuperaciones ligadas a ciclos de adopción de EVs.",
            "high":   "Litio caro históricamente atrae inversión en nuevas minas, que tarda 3-5 años en llegar al mercado.",
            "normal": "Litio en rango. Seguirá los planes de producción de Tesla, BYD y los grandes fabricantes de baterías.",
        },
    }

    commodity_corr = correlations.get(commodity_id, {})
    price_correlation = commodity_corr.get(signal, "")

    # Períodos similares basados en percentil
    similar_periods = []
    if percentile <= 20:
        similar_periods = ["Mar 2022 (pre-rally)", "Dic 2020 (post-COVID recovery)"]
    elif percentile >= 80:
        similar_periods = ["Nov 2023 (precio presionado)", "Jun 2019 (máximos recientes)"]
    else:
        similar_periods = ["Promedio 2021-2023"]

    return {
        "price_correlation": price_correlation,
        "similar_periods": similar_periods,
    }


def compute_metrics(values: List[float], config: dict, data_type: str, commodity_id: str, dates: List[datetime] = None) -> CommodityData:
    current       = values[-1]
    avg5y         = sum(values) / len(values)
    min5y         = min(values)
    max5y         = max(values)
    weekly_change = round(((current - values[-2]) / values[-2]) * 100, 2) if len(values) >= 2 else 0.0
    percentile    = int(((current - min5y) / (max5y - min5y)) * 100) if max5y != min5y else 50
    ratio         = current / avg5y
    signal        = "low" if ratio < 0.85 else "high" if ratio > 1.15 else "normal"
    history       = [round(v, 3) for v in values[-10:]]

    # Análisis estacional
    seasonal = compute_seasonal(values, dates or [], current)

    # Correlación precio/inventario
    corr = compute_price_correlation(commodity_id, percentile, signal)

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
        seasonal_avg=seasonal["seasonal_avg"],
        seasonal_diff_pct=seasonal["seasonal_diff_pct"],
        seasonal_label=seasonal["seasonal_label"],
        yoy_change_pct=seasonal["yoy_change_pct"],
        price_correlation=corr["price_correlation"],
        similar_periods=corr["similar_periods"],
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
    dates  = []
    for row in series:
        val = row.get("value")
        if val is not None:
            try:
                values.append(float(val) / config["scale"])
                dates.append(datetime.strptime(row.get("period", "2020-01-01"), "%Y-%m-%d"))
            except (ValueError, TypeError):
                pass

    if len(values) < 5:
        raise ValueError(f"Not enough EIA data: {len(values)} points")

    result = compute_metrics(values, config, "inventory", commodity_id, dates)
    return cache_set(f"eia_{commodity_id}", result)


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
    cache_size = len(_general_cache)
    return {
        "status": "ok",
        "version": "4.0.0",
        "timestamp": datetime.now().isoformat(),
        "cache_entries": cache_size,
    }

@app.post("/cache/clear")
def clear_cache():
    _general_cache.clear()
    _general_cache_time.clear()
    _climate_cache.clear()
    _climate_cache_time.clear()
    return {"status": "ok", "message": "Cache limpiado"}


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

# Cache de clima — se renueva cada 6 horas para no saturar Open-Meteo
_climate_cache = {}
_climate_cache_time = {}
CLIMATE_CACHE_TTL = 6 * 3600  # 6 horas en segundos

# Cache general — EIA, USDA, precios y noticias
_general_cache = {}
_general_cache_time = {}
GENERAL_CACHE_TTL = 30 * 60  # 30 minutos

def cache_get(key: str):
    if key in _general_cache:
        age = time.time() - _general_cache_time.get(key, 0)
        if age < GENERAL_CACHE_TTL:
            logger.info(f"Cache hit: {key}")
            return _general_cache[key]
    return None

def cache_set(key: str, value):
    _general_cache[key] = value
    _general_cache_time[key] = time.time()
    return value

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
    cache_key = f"{lat},{lon}"

    # Verificar cache
    if cache_key in _climate_cache:
        age = time.time() - _climate_cache_time.get(cache_key, 0)
        if age < CLIMATE_CACHE_TTL:
            logger.info(f"Climate cache hit for {zone['name']}")
            return _climate_cache[cache_key]

    # Delay para no saturar Open-Meteo
    time.sleep(1)

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

    result = ZoneClimate(
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
    _climate_cache[cache_key] = result
    _climate_cache_time[cache_key] = time.time()
    return result


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

NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "")
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
    cached = cache_get(f"news_{commodity_id}")
    if cached: return cached
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

    result = CommodityNews(
        commodity_id=commodity_id,
        articles=articles,
        lastUpdated=datetime.now().isoformat(),
    )
    return cache_set(f"news_{commodity_id}", result)


@app.get("/news/{commodity_id}", response_model=CommodityNews)
def get_news(commodity_id: str):
    all_ids = list(EIA_CONFIG.keys()) + list(USDA_CONFIG.keys()) + list(PRICE_CONFIG.keys())
    if commodity_id not in all_ids:
        raise HTTPException(status_code=404, detail=f"'{commodity_id}' no encontrado")
    return fetch_news(commodity_id)


# ── ALERTAS ───────────────────────────────────────────────────────────────────



SENDGRID_API_KEY  = os.environ.get("SENDGRID_API_KEY", "")
SENDGRID_FROM     = os.environ.get("SENDGRID_FROM", "")
SENDGRID_FROM_NAME = "Stockpile Inventory"
SUBSCRIBERS_FILE  = "subscribers.json"
SIGNALS_FILE      = "last_signals.json"


class SubscribeRequest(BaseModel):
    email: str
    commodities: List[str]  # lista de IDs, ej: ["crude", "soy", "gold"]


class UnsubscribeRequest(BaseModel):
    email: str


def load_json(path: str, default):
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return default


def save_json(path: str, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def send_alert_email(to_email: str, alerts: list):
    """Envía email con las alertas de señal via SendGrid."""
    rows = ""
    for a in alerts:
        color = "#c0392b" if a["new_signal"] == "high" else "#27ae60" if a["new_signal"] == "low" else "#d4a017"
        label = "ALTO" if a["new_signal"] == "high" else "BAJO" if a["new_signal"] == "low" else "NORMAL"
        prev  = "ALTO" if a["old_signal"] == "high" else "BAJO" if a["old_signal"] == "low" else "NORMAL"
        rows += f"""
        <tr>
          <td style="padding:12px 16px;border-bottom:1px solid #eee;font-family:monospace;font-weight:bold">{a['name']}</td>
          <td style="padding:12px 16px;border-bottom:1px solid #eee;font-family:monospace;color:#888">{prev}</td>
          <td style="padding:12px 16px;border-bottom:1px solid #eee">
            <span style="background:{color};color:white;padding:3px 10px;font-family:monospace;font-size:12px;font-weight:bold">{label}</span>
          </td>
          <td style="padding:12px 16px;border-bottom:1px solid #eee;font-family:monospace;font-size:13px">{a['value']} {a['unit']}</td>
        </tr>"""

    html = f"""
    <div style="font-family:sans-serif;max-width:600px;margin:0 auto;background:#f2ede6">
      <div style="background:#1a1410;padding:24px 32px">
        <div style="font-family:monospace;font-size:11px;color:#8a7d6e;letter-spacing:3px">GLOBAL COMMODITY STOCK MONITOR</div>
        <div style="font-size:28px;font-weight:900;color:white;margin-top:4px">STOCKPILE <span style="color:#c0392b">INVENTORY</span></div>
      </div>
      <div style="padding:32px">
        <h2 style="font-size:18px;font-weight:800;margin-bottom:8px">⚠️ Cambio de señal detectado</h2>
        <p style="color:#666;margin-bottom:24px">Los siguientes commodities cambiaron su señal de inventario:</p>
        <table style="width:100%;border-collapse:collapse;background:white">
          <thead>
            <tr style="background:#1a1410;color:white">
              <th style="padding:10px 16px;text-align:left;font-family:monospace;font-size:11px">COMMODITY</th>
              <th style="padding:10px 16px;text-align:left;font-family:monospace;font-size:11px">ANTERIOR</th>
              <th style="padding:10px 16px;text-align:left;font-family:monospace;font-size:11px">NUEVA SEÑAL</th>
              <th style="padding:10px 16px;text-align:left;font-family:monospace;font-size:11px">VALOR ACTUAL</th>
            </tr>
          </thead>
          <tbody>{rows}</tbody>
        </table>
        <div style="margin-top:24px">
          <a href="https://playful-torrone-1a3563.netlify.app" 
             style="background:#c0392b;color:white;padding:12px 24px;text-decoration:none;font-weight:bold;font-family:monospace;letter-spacing:1px">
            VER DASHBOARD →
          </a>
        </div>
        <p style="color:#999;font-size:11px;margin-top:32px;font-family:monospace">
          Recibís este email porque estás suscripto a alertas de Stockpile Inventory.<br>
          Para desuscribirte respondé este email con "UNSUB".
        </p>
      </div>
    </div>"""

    payload = {
        "personalizations": [{"to": [{"email": to_email}]}],
        "from": {"email": SENDGRID_FROM, "name": SENDGRID_FROM_NAME},
        "subject": f"⚠️ Stockpile Alert — {len(alerts)} señal{'es' if len(alerts) > 1 else ''} cambiada{'s' if len(alerts) > 1 else ''}",
        "content": [{"type": "text/html", "value": html}],
    }

    with httpx.Client(timeout=15) as client:
        resp = client.post(
            "https://api.sendgrid.com/v3/mail/send",
            json=payload,
            headers={"Authorization": f"Bearer {SENDGRID_API_KEY}"},
        )
        resp.raise_for_status()
        logger.info(f"Alert email sent to {to_email}")


@app.post("/subscribe")
def subscribe(req: SubscribeRequest):
    """Suscribir email a alertas de commodities."""
    subscribers = load_json(SUBSCRIBERS_FILE, {})
    subscribers[req.email] = {
        "email": req.email,
        "commodities": req.commodities,
        "created": datetime.now().isoformat(),
    }
    save_json(SUBSCRIBERS_FILE, subscribers)
    logger.info(f"New subscriber: {req.email} → {req.commodities}")
    return {"status": "ok", "message": f"Suscripto a {len(req.commodities)} commodities"}


@app.delete("/unsubscribe")
def unsubscribe(req: UnsubscribeRequest):
    subscribers = load_json(SUBSCRIBERS_FILE, {})
    if req.email in subscribers:
        del subscribers[req.email]
        save_json(SUBSCRIBERS_FILE, subscribers)
    return {"status": "ok", "message": "Desuscripto correctamente"}


@app.get("/subscribers/count")
def subscriber_count():
    subscribers = load_json(SUBSCRIBERS_FILE, {})
    return {"count": len(subscribers)}


@app.post("/alerts/check")
def check_and_send_alerts():
    """Compara señales actuales con las anteriores y manda emails si cambiaron."""
    last_signals = load_json(SIGNALS_FILE, {})
    
    # Obtener señales actuales
    current = {}
    all_ids = list(EIA_CONFIG.keys()) + list(USDA_CONFIG.keys()) + list(PRICE_CONFIG.keys())
    
    for cid in all_ids:
        try:
            if cid in EIA_CONFIG:
                data = fetch_eia(cid)
            elif cid in USDA_CONFIG:
                data = fetch_usda(cid)
            else:
                data = fetch_price(cid)
            current[cid] = {
                "signal": data.signal,
                "name": data.name,
                "value": data.current,
                "unit": data.unit,
            }
        except Exception as e:
            logger.error(f"Alert check failed for {cid}: {e}")

    # Guardar señales actuales
    save_json(SIGNALS_FILE, current)

    # Detectar cambios
    changes = {}
    for cid, curr in current.items():
        prev = last_signals.get(cid, {})
        if prev and prev.get("signal") != curr["signal"]:
            changes[cid] = {
                "name": curr["name"],
                "old_signal": prev["signal"],
                "new_signal": curr["signal"],
                "value": round(curr["value"], 2),
                "unit": curr["unit"],
            }

    if not changes:
        return {"status": "ok", "alerts_sent": 0, "message": "Sin cambios de señal"}

    # Notificar suscriptores
    subscribers = load_json(SUBSCRIBERS_FILE, {})
    emails_sent = 0

    for email, sub in subscribers.items():
        relevant = [v for k, v in changes.items() if k in sub.get("commodities", [])]
        if relevant:
            try:
                send_alert_email(email, relevant)
                emails_sent += 1
            except Exception as e:
                logger.error(f"Failed to send alert to {email}: {e}")

    return {
        "status": "ok",
        "alerts_sent": emails_sent,
        "signal_changes": len(changes),
        "changed": list(changes.keys()),
    }
