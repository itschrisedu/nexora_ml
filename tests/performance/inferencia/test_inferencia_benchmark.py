"""Benchmark de rendimiento y latencia para el motor de Machine Learning de Nexora."""

from __future__ import annotations

import time
from datetime import date, timedelta
from app.engine import predecir_demanda
from app.schemas import VentaHistorica


def _generar_dataset_benchmark(num_registros: int = 100) -> list[VentaHistorica]:
    """Genera dataset sintético para benchmark."""
    base_date = date(2026, 1, 1)
    ventas = []
    for i in range(num_registros):
        ventas.append(
            VentaHistorica(
                fecha=base_date + timedelta(days=i % 30),
                modelo="Botín Londres" if i % 2 == 0 else "Mocasín Clásico",
                serie="ADULTO",
                talla=38 + (i % 5),
                cantidad=3,
                precio_unitario=22.0,
                canal="MANUAL",
            )
        )
    return ventas


def test_latencia_inferencia_menor_a_150ms():
    """Demuestra empíricamente que la inferencia de demanda se ejecuta en menos de 150 ms."""
    tenant_id = "tenant-benchmark-01"
    ventas = _generar_dataset_benchmark(120)

    # 1. Warm-up
    predecir_demanda(tenant_id=tenant_id, ventas=ventas, horizonte_dias=30, temporada="REGULAR")

    # 2. Medir tiempo de inferencia
    num_iteraciones = 10
    tiempos = []

    for _ in range(num_iteraciones):
        t_inicio = time.perf_counter()
        resp = predecir_demanda(
            tenant_id=tenant_id,
            ventas=ventas,
            horizonte_dias=30,
            temporada="FERIA_CEVALLOS",
        )
        t_fin = time.perf_counter()
        tiempos.append((t_fin - t_inicio) * 1000)  # Convertir a ms

    latencia_promedio_ms = sum(tiempos) / len(tiempos)

    assert resp is not None
    assert len(resp.predicciones) > 0
    # Latencia promedio esperada menor a 150 ms
    assert latencia_promedio_ms < 150.0, f"Latencia promedio ({latencia_promedio_ms:.2f}ms) superó los 150ms"
