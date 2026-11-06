import unittest

import tests._langchain_stubs  # noqa: F401
from core import tools as nvidia_tools


def call_tool(name: str, **kwargs) -> str:
    selected_tool = getattr(nvidia_tools, name)
    if hasattr(selected_tool, "invoke"):
        return selected_tool.invoke(kwargs)
    return selected_tool(**kwargs)


class NvidiaToolsTests(unittest.TestCase):
    def test_inventory_query_reads_recorded_rtx_4090_units(self):
        response = call_tool(
            "analizar_ventas",
            agrupar_por="producto",
            producto="RTX 4090",
        )

        self.assertIn("1,250.00 unidades", response)

    def test_monthly_query_does_not_infer_from_launch_date(self):
        response = call_tool(
            "analizar_ventas",
            producto="RTX 4090",
            mes="enero 2024",
        )

        self.assertIn("no registra ventas por mes", response)
        self.assertIn("2024-01-15", response)

    def test_stock_metric_groups_by_category(self):
        response = call_tool(
            "analizar_ventas",
            agrupar_por="categoria",
            metrica="stock",
        )

        self.assertIn("stock disponible", response.lower())
        self.assertIn("GPU Consumidor", response)

    def test_power_estimate_requires_known_profiles(self):
        known = call_tool("calcular_watts_fuente", gpu="RTX 4090", cpu="i9-14900K")
        unknown = call_tool("calcular_watts_fuente", gpu="Unknown GPU", cpu="Unknown CPU")

        self.assertIn("fuente de al menos 1050 W", known)
        self.assertIn("No tengo un perfil", unknown)

    def test_compatibility_checks_socket_and_memory(self):
        compatible = call_tool(
            "verificar_compatibilidad",
            socket_cpu="AM5",
            socket_motherboard="am5",
            tipo_ram_motherboard="DDR5",
            tipo_ram_memoria="ddr5",
        )
        incompatible = call_tool(
            "verificar_compatibilidad",
            socket_cpu="AM4",
            socket_motherboard="AM5",
            tipo_ram_motherboard="DDR5",
            tipo_ram_memoria="DDR4",
        )

        self.assertIn("Compatible según", compatible)
        self.assertIn("No compatible", incompatible)

    def test_budget_build_uses_static_gpu_price_and_discloses_estimate(self):
        response = call_tool("recomendar_build_presupuesto", presupuesto=2500)

        self.assertIn("RTX 4080 SUPER", response)
        self.assertIn("CPU:", response)
        self.assertIn("Motherboard:", response)
        self.assertIn("estimado", response)
        self.assertIn("estimaciones locales", response)


if __name__ == "__main__":
    unittest.main()