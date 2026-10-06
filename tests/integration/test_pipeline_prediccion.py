"""Integration tests for Nexora ML prediction and inference pipeline."""

from __future__ import annotations

from datetime import date, timedelta
from app.engine import entrenar_modelo, obtener_estado_modelo, predecir_demanda
from app.schemas import PrediccionRequest, ReentrenamientoRequest, VentaHistorica


def _generar_ventas_historicas(num_dias: int = 45) -> list[VentaHistorica]:
    """Genera dataset sintético de ventas de calzado para pruebas de integración."""
    base_date = date(2026, 1, 1)
    ventas = []
    modelos = ["Mocasín Clásico", "Botín Cuero Londres", "Zapato Escolar Oxford"]
    series = ["ADULTO", "JUVENIL", "INFANTIL"]
    tallas = [38, 39, 40, 41]

    for d in range(num_dias):
        current_date = base_date + timedelta(days=d)
        for mod in modelos:
            for s in series:
                for t in tallas:
                    ventas.append(
                        VentaHistorica(
                            fecha=current_date,
                            modelo=mod,
                            serie=s,
                            talla=t,
                            cantidad=2 if (d % 2 == 0) else 4,
                            precio_unitario=24.50,
                            canal="CATALOGO" if d % 3 == 0 else "MANUAL",
                        )
                    )
    return ventas


def test_flujo_integrado_prediccion_demanda_estacional():
    """Valida el ciclo completo de inferencia ajustada por escenario estacional."""
    tenant_id = "tenant-test-cevallos-01"
    ventas = _generar_ventas_historicas(num_dias=40)

    # 1. Solicitar predicción con escenario estacional 'FERIA_CEVALLOS'
    req_feria = PrediccionRequest(
        tenant_id=tenant_id,
        ventas=ventas,
        horizonte_dias=30,
        temporada="FERIA_CEVALLOS",
    )

    resp_feria = predecir_demanda(
        tenant_id=req_feria.tenant_id,
        ventas=req_feria.ventas,
        horizonte_dias=req_feria.horizonte_dias,
        temporada=req_feria.temporada,
    )

    assert resp_feria.tenant_id == tenant_id
    assert resp_feria.horizonte_dias == 30
    assert resp_feria.temporada == "FERIA_CEVALLOS"
    assert len(resp_feria.predicciones) > 0
    assert resp_feria.total_pares_estimados > 0

    # Validar que los ítems predichos tengan estructura coherente
    for item in resp_feria.predicciones:
        assert item.modelo in ["Mocasín Clásico", "Botín Cuero Londres", "Zapato Escolar Oxford"]
        assert item.demanda_estimada >= 0
        assert item.nivel_confianza in ["ALTO", "MEDIO", "BAJO"]


def test_flujo_integrado_reentrenamiento_y_estado():
    """Valida el re-entrenamiento del modelo y la persistencia de métricas."""
    tenant_id = "tenant-test-reentrenamiento-02"
    ventas = _generar_ventas_historicas(num_dias=35)

    # 1. Ejecutar re-entrenamiento
    status = entrenar_modelo(tenant_id=tenant_id, ventas=ventas)
    assert status.tenant_id == tenant_id
    assert status.entrenado is True
    assert status.total_registros_entrenamiento == len(ventas)

    # 2. Consultar estado del modelo
    estado = obtener_estado_modelo(tenant_id)
    assert estado.tenant_id == tenant_id
    assert estado.entrenado is True
    assert estado.algoritmo == "GradientBoostingRegressor"
