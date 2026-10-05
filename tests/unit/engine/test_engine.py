"""Tests unitarios para el motor de Machine Learning (app.engine)."""

import unittest
from datetime import date, timedelta
import pandas as pd

from app.engine import (
    TEMPORADAS_CONFIG,
    _build_features,
    _agregar_rolling,
    entrenar_modelo,
    obtener_estado_modelo,
    predecir_demanda,
)
from app.schemas import VentaHistorica


class TestEngineUnit(unittest.TestCase):
    """Pruebas unitarias para feature engineering, entrenamiento y predicción."""

    def setUp(self):
        """Generar un dataset sintético de ventas de calzado para tests."""
        self.tenant_id = "test-tenant-cevallos-modular"
        base_date = date(2026, 1, 1)
        self.ventas_mock = []

        modelos = ["Mocasín Escolar", "Botín Cuero", "Oxford Casual"]
        series = ["Escolar", "Dama", "Caballero"]
        tallas = [36, 38, 40]

        # Generar 40 registros de venta
        for i in range(40):
            self.ventas_mock.append(
                VentaHistorica(
                    fecha=base_date + timedelta(days=i),
                    modelo=modelos[i % len(modelos)],
                    serie=series[i % len(series)],
                    talla=tallas[i % len(tallas)],
                    cantidad=(i % 5) + 1,
                    precio_unitario=35.0 + (i % 3) * 5.0,
                    canal="MANUAL" if i % 2 == 0 else "WHATSAPP",
                )
            )

    def test_build_features(self):
        """Verifica que las columnas de features temporales se generen adecuadamente."""
        df_raw = pd.DataFrame([v.model_dump() for v in self.ventas_mock])
        df_feat = _build_features(df_raw)

        expected_cols = [
            "dia_semana",
            "dia_mes",
            "mes",
            "quincena",
            "semana_anio",
            "mes_sin",
            "mes_cos",
        ]
        for col in expected_cols:
            self.assertIn(col, df_feat.columns)
            self.assertEqual(len(df_feat[col]), len(df_raw))

    def test_agregar_rolling(self):
        """Verifica que se calculen medias móviles y coeficiente de tendencia."""
        df_raw = pd.DataFrame([v.model_dump() for v in self.ventas_mock])
        df_roll = _agregar_rolling(df_raw)

        self.assertIn("rolling_7d", df_roll.columns)
        self.assertIn("rolling_14d", df_roll.columns)
        self.assertIn("tendencia_coef", df_roll.columns)
        self.assertFalse(df_roll["rolling_7d"].isnull().any())

    def test_entrenar_modelo_exitoso(self):
        """Verifica el pipeline de entrenamiento de GradientBoosting."""
        result = entrenar_modelo(self.tenant_id, self.ventas_mock)

        self.assertIn("model", result)
        self.assertIn("r2_score", result)
        self.assertIn("n_registros", result)
        self.assertEqual(result["n_registros"], len(self.ventas_mock))
        self.assertGreaterEqual(result["r2_score"], 0.0)

    def test_entrenar_modelo_insuficientes_registros(self):
        """Debe lanzar ValueError si hay menos de 10 ventas registradas."""
        ventas_cortas = self.ventas_mock[:5]
        with self.assertRaises(ValueError):
            entrenar_modelo(self.tenant_id, ventas_cortas)

    def test_predecir_demanda_regular(self):
        """Verifica la generación de predicciones en temporada REGULAR."""
        pred = predecir_demanda(
            tenant_id=self.tenant_id,
            ventas=self.ventas_mock,
            horizonte_dias=30,
            temporada="REGULAR",
        )

        self.assertEqual(pred.tenant_id, self.tenant_id)
        self.assertEqual(pred.horizonte_dias, 30)
        self.assertEqual(pred.temporada_activa, "REGULAR")
        self.assertGreater(pred.total_productos_analizados, 0)
        self.assertGreater(len(pred.predicciones), 0)

        for item in pred.predicciones:
            self.assertGreaterEqual(item.demanda_estimada, 0)
            self.assertGreaterEqual(item.sugerencia_reorden, 0)
            self.assertIn(item.tendencia, ["ALZA", "ESTABLE", "BAJA"])

    def test_predecir_demanda_estacional_clases_sierra(self):
        """Verifica el factor estacional en productos escolares durante CLASES_SIERRA."""
        pred = predecir_demanda(
            tenant_id=self.tenant_id,
            ventas=self.ventas_mock,
            horizonte_dias=30,
            temporada="CLASES_SIERRA",
        )

        self.assertEqual(pred.temporada_activa, "CLASES_SIERRA")
        self.assertEqual(pred.temporada_nombre, TEMPORADAS_CONFIG["CLASES_SIERRA"]["nombre"])

        # Los productos escolares deben recibir multiplicador >= default
        escolar_item = next(
            (p for p in pred.predicciones if "escolar" in p.modelo.lower()),
            None,
        )
        if escolar_item:
            self.assertGreaterEqual(escolar_item.factor_estacional, 1.35)

    def test_obtener_estado_modelo(self):
        """Verifica consulta de metadata del modelo entrenado."""
        entrenar_modelo(self.tenant_id, self.ventas_mock)
        status = obtener_estado_modelo(self.tenant_id)

        self.assertEqual(status["tenant_id"], self.tenant_id)
        self.assertTrue(status["modelo_entrenado"])
        self.assertEqual(status["registros_entrenamiento"], len(self.ventas_mock))


if __name__ == "__main__":
    unittest.main()
