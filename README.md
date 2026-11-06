# InvidiAgent

Proyecto de asistente técnico llamado **RetroTron**, especializado en GPUs NVIDIA, drivers, software, compatibilidad de hardware y análisis del inventario local. Usa Groq u Ollama como proveedor LLM; NVIDIA describe el dominio del agente, no el proveedor del modelo.

## Capacidades

- Atención general y consultas de catálogo mediante un subgrafo especializado.
- Recuperación de información técnica desde archivos TXT estructurados, con PDF homónimo como respaldo.
- Análisis Pandas del inventario en `data/documentos/04_ventas_e_inventario.csv`.
- Cálculo orientativo de fuente, verificación de socket/RAM y recomendación de build por presupuesto.
- Clasificación en `atencion_cliente`, `documentos`, `analisis` o `reporte`.

## Datos y límites

El CSV incluye `producto`, `categoria`, `marca`, `arquitectura`, `version`, `fecha_lanzamiento`, `precio_usd`, `cantidad_vendida` y `stock_disponible`. Las cantidades son agregadas: no hay fechas de venta ni historial mensual. Por tanto, `fecha_lanzamiento` no permite calcular ventas por mes. Los precios son datos estáticos del archivo y no representan cotizaciones actuales.

Las estimaciones para CPU y fuente y el costo conjunto de componentes no GPU son orientativos. Verifica las especificaciones, compatibilidad, precios y disponibilidad con los fabricantes o vendedores antes de comprar.

## Configuración

Requiere Python 3.11 o superior.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Configura en `.env` al menos `LLM_PROVIDER`, las credenciales/modelo de Groq u Ollama y la identidad del agente. `DOCUMENTOS_DIR` por defecto es `data/documentos`; las rutas relativas se resuelven desde la raíz del proyecto. TXT tiene prioridad sobre PDF del mismo nombre base. Consulta [.env.example](.env.example) para todas las opciones.

## Ejecución

```powershell
uvicorn main:app --reload --port 8000
```

- Interfaz: `http://localhost:8000/`
- API: `http://localhost:8000/api`
- Chat: `POST http://localhost:8000/chat`
- Estado: `http://localhost:8000/health`
- Documentación: `http://localhost:8000/docs`

## Verificación

```powershell
python -m unittest discover -s tests
python -m compileall core
```

El servicio requiere las dependencias de [requirements.txt](requirements.txt) y la configuración del proveedor LLM antes de probar consultas que invocan el modelo.