"""Schemas Pydantic para request/response del microservicio ML."""

from __future__ import annotations

from datetime import date
from pydantic import BaseModel, Field


# ─── Request Schemas ─────────────────────────────────────────

class VentaHistorica(BaseModel):
    """Registro de venta individual para alimentar el modelo."""

    fecha: date
    modelo: str
    serie: str
    talla: int
    cantidad: int
    precio_unitario: float
    canal: str  # MANUAL | WHATSAPP | CATALOGO


class PrediccionRequest(BaseModel):
    """Payload para solicitar predicción de demanda."""

    tenant_id: str
    ventas: list[VentaHistorica] = Field(
        ..., description="Historial de ventas (mínimo 30 registros recomendado)"
    )
    horizonte_dias: int = Field(
        default=30,
        ge=7,
        le=180,
        description="Días a futuro para la predicción",
    )
    temporada: str = Field(
        default="REGULAR",
        description="Escenario estacional: REGULAR | CLASES_SIERRA | CLASES_COSTA | NAVIDAD_FIN_ANIO | DIA_MADRE_PADRE | FERIA_CEVALLOS",
    )


class ReentrenamientoRequest(BaseModel):
    """Payload para forzar re-entrenamiento del modelo."""

    tenant_id: str
    ventas: list[VentaHistorica]


# ─── Response Schemas ────────────────────────────────────────

class PrediccionItem(BaseModel):
    """Predicción individual por modelo/serie/talla."""

    modelo: str
    serie: str
    talla: int
    demanda_estimada: int = Field(
        ..., description="Unidades estimadas en el horizonte (ajustadas por temporada)"
    )
    demanda_base: int = Field(
        default=0, description="Demanda histórica regular no estacional"
    )
    factor_estacional: float = Field(
        default=1.0, description="Multiplicador estacional aplicado"
    )
    impacto_estacional_pct: float = Field(
        default=0.0, description="Incremento o decremento porcentual estimado"
    )
    confianza: float = Field(
        ..., ge=0, le=1, description="Nivel de confianza de la predicción"
    )
    tendencia: str = Field(
        ..., description="ALZA | ESTABLE | BAJA"
    )
    sugerencia_reorden: int = Field(
        ..., description="Cantidad sugerida a re-ordenar a talleres"
    )


class PrediccionResponse(BaseModel):
    """Respuesta completa de predicción."""

    tenant_id: str
    horizonte_dias: int
    total_productos_analizados: int
    temporada_activa: str = "REGULAR"
    temporada_nombre: str = "Temporada Regular"
    temporada_descripcion: str = "Proyección de rotación estándar basada en historial"
    multiplicador_global: float = 1.0
    predicciones: list[PrediccionItem]
    modelo_score: float = Field(
        ..., description="R² score del modelo entrenado"
    )
    alerta_stock_bajo: list[str] = Field(
        default_factory=list,
        description="Productos con predicción de demanda alta o reorden prioritario",
    )


class ModelStatusResponse(BaseModel):
    """Estado del modelo ML para un tenant."""

    tenant_id: str
    modelo_entrenado: bool
    ultimo_entrenamiento: str | None = None
    registros_entrenamiento: int = 0
    score_r2: float | None = None
    features_utilizados: list[str] = []


class HealthResponse(BaseModel):
    """Respuesta de health check."""

    status: str = "ok"
    service: str = "nexora-ml"
    version: str = "1.0.0"
