from pydantic import BaseModel, Field


# ════════════════════════════════════════════════════════════
#  MODELOS NVIDIA ESPECÍFICOS
# ════════════════════════════════════════════════════════════

class DriverUpdate(BaseModel):
    """Modelo para actualización de drivers NVIDIA."""
    version: str = Field(description="Versión del driver (ej. 552.22)")
    fecha_lanzamiento: str = Field(description="Fecha de lanzamiento")
    componentes_soportados: list[str] = Field(description="Componentes compatibles")
    mejoras_principal: str = Field(description="Mejoras clave en esta versión")
    problemas_conocidos: list[str] = Field(default_factory=list, description="Problemas conocidos")

class ProductoNVIDIA(BaseModel):
    """Modelo para un producto NVIDIA."""
    nombre: str = Field(description="Nombre del producto (ej. RTX 4090)")
    categoria: str = Field(description="Categoría: GPU, DRIVER, SOFTWARE, HARDWARE")
    serie: str = Field(description="Serie (ej. 40-series, 30-series)")
    memoria: str = Field(description="Memoria video (ej. 24GB GDDR6X)")
    rendimiento: str = Field(description="Nivel de rendimiento: consumidor, profesional, IA")
    precio_referencia: float = Field(description="Precio sugerido en USD")

class ReporteDesempenho(BaseModel):
    """Reporte de rendimiento."""
    configuracion: str = Field(description="Configuración de hardware testeada")
    benchmark: str = Field(description="Nombre del benchmark utilizado")
    fps_promedio: float = Field(description="FPS promedio obtenido")
    temperatura: float = Field(description="Temperatura máxima observada")
    consumo_watts: float = Field(description="Consumo de potencia")
    nivel_desempeño: str = Field(description="1080p, 1440p, 4K o IA")


class Alerta(BaseModel):
    """Una alerta o riesgo detectado en un reporte."""
    area: str = Field(description="Área afectada")
    descripcion: str = Field(description="Descripción concreta de la alerta")
    accion_recomendada: str = Field(description="Acción recomendada")


class ReporteEjecutivo(BaseModel):
    """Esquema legado conservado para consumidores existentes."""
    resumen_ejecutivo: str = Field(description="Resumen ejecutivo")
    alertas: list[Alerta] = Field(default_factory=list)
    oportunidades: list[str] = Field(default_factory=list)
    recomendaciones: list[str] = Field(default_factory=list)


class ComponentePC(BaseModel):
    """Componente de una build o del inventario disponible."""
    nombre: str
    categoria: str
    marca: str = ""
    modelo: str = ""
    precio: float | None = None
    stock: int | None = None
    especificaciones: dict[str, str] = Field(default_factory=dict)


class BuildPC(BaseModel):
    componentes: list[ComponentePC] = Field(default_factory=list)
    costo_total: float
    watts_recomendados: int
    nivel_desempeno: str = Field(description="1080p, 1440p, 4K o IA")


class AlertaHardware(BaseModel):
    area: str
    descripcion: str
    accion_recomendada: str


class ReporteInventarioHardware(BaseModel):
    resumen_general: str
    alertas: list[AlertaHardware] = Field(default_factory=list)
    oportunidades: list[str] = Field(default_factory=list)
    recomendaciones: list[str] = Field(default_factory=list)
