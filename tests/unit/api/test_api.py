"""Tests unitarios para la API REST de Nexora ML (FastAPI)."""

import unittest
from datetime import date, timedelta
from fastapi.testclient import TestClient

from app.main import app


class TestApiUnit(unittest.TestCase):
    """Pruebas para endpoints HTTP de FastAPI."""

    def setUp(self):
        self.client = TestClient(app)
        self.tenant_id = "tenant-api-test-modular"

        base_date = date(2026, 1, 1)
        self.ventas_mock = [
            {
                "fecha": (base_date + timedelta(days=i)).isoformat(),
                "modelo": "Calzado Ejecutivo",
                "serie": "Caballero",
                "talla": 39 + (i % 3),
                "cantidad": 3 + (i % 4),
                "precio_unitario": 42.0,
                "canal": "MANUAL",
            }
            for i in range(25)
        ]

    def test_health_endpoint(self):
        """GET /health debe retornar status ok y version."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["service"], "nexora-ml")

    def test_prediccion_endpoint_valido(self):
        """POST /prediccion debe procesar historial y retornar predicciones."""
        payload = {
            "tenant_id": self.tenant_id,
            "ventas": self.ventas_mock,
            "horizonte_dias": 30,
            "temporada": "REGULAR",
        }
        response = self.client.post("/prediccion", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["tenant_id"], self.tenant_id)
        self.assertGreater(len(data["predicciones"]), 0)

    def test_prediccion_endpoint_sin_suficientes_ventas(self):
        """POST /prediccion debe retornar 422 si hay menos de 10 registros en tenant sin modelo."""
        payload = {
            "tenant_id": "tenant-nuevo-sin-modelo-999",
            "ventas": self.ventas_mock[:4],
            "horizonte_dias": 30,
            "temporada": "REGULAR",
        }
        response = self.client.post("/prediccion", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_modelo_status_endpoint(self):
        """GET /modelo/{tenant_id} debe retornar estado del modelo."""
        # Primero entrenar con predicción
        payload = {
            "tenant_id": self.tenant_id,
            "ventas": self.ventas_mock,
            "horizonte_dias": 30,
            "temporada": "REGULAR",
        }
        self.client.post("/prediccion", json=payload)

        # Consultar estado
        response = self.client.get(f"/modelo/{self.tenant_id}")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["tenant_id"], self.tenant_id)
        self.assertTrue(data["modelo_entrenado"])


if __name__ == "__main__":
    unittest.main()
