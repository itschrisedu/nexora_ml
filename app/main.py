"""
Nexora ML — Microservicio de Predicción de Demanda.

FastAPI application que expone endpoints REST para:
  - POST /prediccion        → Generar predicción de demanda
  - POST /reentrenamiento   → Forzar re-entrenamiento del modelo
  - GET  /modelo/{tenant}   → Estado del modelo de un tenant
  - GET  /health            → Health check
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .engine import entrenar_modelo, obtener_estado_modelo, predecir_demanda
from .schemas import (
    HealthResponse,
    ModelStatusResponse,
    PrediccionRequest,
    PrediccionResponse,
    ReentrenamientoRequest,
)

# ─── Logging ─────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
)
logger = logging.getLogger("nexora-ml")

# ─── FastAPI App ─────────────────────────────────────────────

app = FastAPI(
    title="Nexora ML — Predicción de Demanda",
    description=(
        "Microservicio de Machine Learning para predicción de demanda "
        "de calzado. Utiliza GradientBoostingRegressor con features "
        "temporales y de volumen."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Endpoints ───────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["Sistema"])
async def health_check():
    """Health check del microservicio."""
    return HealthResponse()


@app.post("/prediccion", response_model=PrediccionResponse, tags=["Predicción"])
async def generar_prediccion(body: PrediccionRequest):
    """
    Genera predicción de demanda para un tenant.

    Requiere un historial de ventas (mínimo 10 registros).
    El modelo se entrena automáticamente si no existe uno previo.
    """
    try:
        resultado = predecir_demanda(
            tenant_id=body.tenant_id,
            ventas=body.ventas,
            horizonte_dias=body.horizonte_dias,
            temporada=body.temporada,
        )
        logger.info(
            "Predicción generada para tenant=%s — %d productos analizados",
            body.tenant_id,
            resultado.total_productos_analizados,
        )
        return resultado
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("Error en predicción: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error interno: {e}")


@app.post(
    "/reentrenamiento",
    response_model=ModelStatusResponse,
    tags=["Entrenamiento"],
)
async def forzar_reentrenamiento(body: ReentrenamientoRequest):
    """
    Fuerza el re-entrenamiento del modelo ML para un tenant.
    Útil cuando se acumulan nuevas ventas significativas.
    """
    try:
        model_data = entrenar_modelo(
            tenant_id=body.tenant_id,
            ventas=body.ventas,
        )
        logger.info(
            "Re-entrenamiento completado para tenant=%s  R²=%.4f",
            body.tenant_id,
            model_data["r2_score"],
        )
        estado = obtener_estado_modelo(body.tenant_id)
        return ModelStatusResponse(**estado)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("Error en re-entrenamiento: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error interno: {e}")


@app.get(
    "/modelo/{tenant_id}",
    response_model=ModelStatusResponse,
    tags=["Entrenamiento"],
)
async def estado_modelo(tenant_id: str):
    """Consulta el estado del modelo ML para un tenant específico."""
    estado = obtener_estado_modelo(tenant_id)
    return ModelStatusResponse(**estado)
