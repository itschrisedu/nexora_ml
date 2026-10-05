"""Tests unitarios para los esquemas Pydantic de Nexora ML."""

import unittest
from datetime import date
from pydantic import ValidationError

from app.schemas import (
    HealthResponse,
    ModelStatusResponse,
    PrediccionItem,
    PrediccionRequest,
    PrediccionResponse,
    ReentrenamientoRequest,
    VentaHistorica,
)


class TestSchemasUnit(unittest.TestCase):
    """Suite de pruebas para validación y serialización de schemas."""

    def test_venta_historica_valida(self):
        venta = VentaHistorica(
            fecha=date(2026, 3, 1),
            modelo="Mocasín Clásico",
            serie="Caballero Formal",
            talla=40,
            cantidad=12,
            precio_unitario=35.50,
            canal="MANUAL",
        )
        self.assertEqual(venta.modelo, "Mocasín Clásico")
        self.assertEqual(venta.talla, 40)
        self.assertEqual(venta.cantidad, 12)
        self.assertEqual(venta.precio_unitario, 35.50)
        self.assertEqual(venta.canal, "MANUAL")

    def test_prediccion_request_horizonte_valido(self):
        ventas = [
            VentaHistorica(
                fecha=date(2026, 2, i + 1),
                modelo="Oxford Cuero",
                serie="Caballero",
                talla=39,
                cantidad=5,
                precio_unitario=40.0,
                canal="WHATSAPP",
            )
            for i in range(10)
        ]
        req = PrediccionRequest(
            tenant_id="tenant-cevallos-1",
            ventas=ventas,
            horizonte_dias=30,
            temporada="CLASES_SIERRA",
        )
        self.assertEqual(req.tenant_id, "tenant-cevallos-1")
        self.assertEqual(len(req.ventas), 10)
        self.assertEqual(req.horizonte_dias, 30)
        self.assertEqual(req.temporada, "CLASES_SIERRA")

    def test_prediccion_request_horizonte_invalido(self):
        ventas = [
            VentaHistorica(
                fecha=date(2026, 1, 1),
                modelo="Botín Dama",
                serie="Mujer",
                talla=36,
                cantidad=2,
                precio_unitario=28.0,
                canal="CATALOGO",
            )
        ]
        # horizonte_dias < 7 debe fallar
        with self.assertRaises(ValidationError):
            PrediccionRequest(
                tenant_id="t-1",
                ventas=ventas,
                horizonte_dias=3,
            )

        # horizonte_dias > 180 debe fallar
        with self.assertRaises(ValidationError):
            PrediccionRequest(
                tenant_id="t-1",
                ventas=ventas,
                horizonte_dias=365,
            )

    def test_prediccion_item_confianza_bounds(self):
        item = PrediccionItem(
            modelo="Casual Urbano",
            serie="Unisex",
            talla=38,
            demanda_estimada=24,
            demanda_base=20,
            factor_estacional=1.2,
            impacto_estacional_pct=20.0,
            confianza=0.85,
            tendencia="ALZA",
            sugerencia_reorden=24,
        )
        self.assertEqual(item.confianza, 0.85)
        self.assertEqual(item.tendencia, "ALZA")

        # Confianza > 1.0 debe fallar
        with self.assertRaises(ValidationError):
            PrediccionItem(
                modelo="Casual",
                serie="U",
                talla=38,
                demanda_estimada=10,
                confianza=1.5,
                tendencia="ALZA",
                sugerencia_reorden=10,
            )

    def test_prediccion_response_estructura(self):
        resp = PrediccionResponse(
            tenant_id="tenant-123",
            horizonte_dias=30,
            total_productos_analizados=1,
            temporada_activa="FERIA_CEVALLOS",
            temporada_nombre="Feria de Calzado Cevallos",
            multiplicador_global=1.35,
            predicciones=[],
            modelo_score=0.88,
            alerta_stock_bajo=["Mocasín T40"],
        )
        self.assertEqual(resp.tenant_id, "tenant-123")
        self.assertEqual(resp.modelo_score, 0.88)
        self.assertIn("Mocasín T40", resp.alerta_stock_bajo)

    def test_health_response_defaults(self):
        health = HealthResponse()
        self.assertEqual(health.status, "ok")
        self.assertEqual(health.service, "nexora-ml")
        self.assertEqual(health.version, "1.0.0")

    def test_model_status_response(self):
        status = ModelStatusResponse(
            tenant_id="t-abc",
            modelo_entrenado=True,
            ultimo_entrenamiento="2026-10-05T12:00:00",
            registros_entrenamiento=120,
            score_r2=0.91,
            features_utilizados=["dia_semana", "rolling_7d", "talla"],
        )
        self.assertTrue(status.modelo_entrenado)
        self.assertEqual(status.registros_entrenamiento, 120)
        self.assertEqual(status.score_r2, 0.91)


if __name__ == "__main__":
    unittest.main()
