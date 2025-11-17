---
title: "Prompting, Plantillas y Salida Estructurada en LLM"
date: 2025-01-01
---

# Prompting, Plantillas y Salida Estructurada en LLM

## Contexto

En esta actividad se usara LangChain para integrar modelos de lenguaje de OpenAI, controlando su comportamiento mediante parámetros, prompts reutilizables y salidas estructuradas. Se busca entender cómo estas herramientas permiten diseñar flujos de generación más controlados, trazables y robustos dentro de aplicaciones basadas en LLMs.

## Objetivos

- Instanciar y utilizar un modelo de chat de OpenAI mediante LangChain para realizar invocaciones básicas.
- Ajustar y razonar sobre los parámetros de decodificación: temperature, max_tokens y top_p.
- Diseñar prompts reutilizables con `ChatPromptTemplate` y el operador `|` para encadenar componentes.
- Obtener salidas estructuradas (JSON/Pydantic) usando `with_structured_output(...)` de forma fiable.
- Medir tokens y latencia empleando LangSmith.

## Actividades

- Parte 0: Setup y Hello LLM
- Parte 1: Parámetros clave (temperature, max_tokens, top_p)
- Parte 2: De texto suelto a plantillas con ChatPromptTemplate + LCEL
- Parte 3: Salida estructurada (JSON) sin post-processing frágil
- Parte 4: Métricas mínimas — tokens y latencia (LangSmith / callbacks)
- Parte 5: Mini-tareas guiadas (aún sin RAG)
- Parte 6: Zero-shot vs Few-shot — “Playground” guiado
- Parte 7: Resúmenes — single-doc y multi-doc (map-reduce)
- Parte 8: Extracción de información — entidades y campos clave
- Parte 9: RAG básico con textos locales (sin base de datos externa)

## Desarrollo

### Invocación básica del modelo

En primer lugar, se estableció una conexión directa con el modelo utilizando la clase `ChatOpenAI`, definiendo los parámetros fundamentales (`model`, `temperature`, `max_tokens`).

El primer experimento fue una consulta simple

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-5", temperature=0.4)

resp = llm.invoke("Definí 'Transformer' en una sola oración.")
print(resp.content)
```

```python
Un Transformer es una arquitectura de redes neuronales basada en mecanismos de atención (especialmente autoatención) que procesa secuencias en paralelo sin recurrencias ni convoluciones, capturando dependencias de largo alcance y sirviendo de base para modelos avanzados de lenguaje y otras tareas secuenciales.
```

La respuesta del modelo fue coherente, técnica y contextualizada, incluyendo referencias a los mecanismos de auto-atención y a su uso en procesamiento del lenguaje natural (NLP).

Esto permitió confirmar la correcta inicialización del entorno, la comunicación con la API de OpenAI y la capacidad de obtener respuestas ajustadas al dominio solicitado.

### Parámetros de decodificación y su impacto en la generación

Se exploraron los parámetros de decodificación que controlan la variabilidad y extensión de las respuestas:

* `temperature`: controla el grado de aleatoriedad.

  * Con `temperature=0.0`, las respuestas fueron deterministas, precisas y repetibles.
  * Con `temperature=0.5`, se observó un balance entre creatividad y coherencia.
  * Con `temperature=0.9`, aumentó notablemente la diversidad de formulaciones, aunque con ligera pérdida de foco temático.

En conjunto, estos experimentos mostraron cómo los parámetros de decodificación permiten adaptar el modelo según la tarea: baja temperatura para tareas analíticas o clasificatorias y alta temperatura para tareas creativas o generativas.

```python
prompts = [
    "Escribí un tuit (<=20 palabras) celebrando un paper de IA.",
    "Dame 3 bullets concisos sobre ventajas de los Transformers."
]

for t in [0.0, 0.5, 0.9]:
    llm_t = ChatOpenAI(model="gpt-5-mini", temperature=t)
    outs = [llm_t.invoke(p).content for p in prompts]
    print(f"\n--- Temperature={t} ---")
    for i, o in enumerate(outs, 1):
        print(f"[{i}] {o}")
```

```python
--- Temperature=0.0 ---
[1] ¡Enhorabuena! Este paper de IA marca un avance clave: rigor, creatividad y gran impacto. Felicidades al equipo. #IA
[2] - Paralelización: la self-attention permite entrenar en paralelo, siendo mucho más rápida que RNNs.
- Contexto global: captura dependencias a largo plazo eficazmente mediante atención sobre toda la secuencia.
- Escalabilidad y transferencia: escalan bien con datos/modelo y facilitan preentrenamiento y fine‑tuning para muchas tareas.

--- Temperature=0.5 ---
[1] ¡Gran paper de IA! Innovación, rigor y ética que impulsa la ciencia hacia adelante.
[2] - Paralelizable: la auto‑atención permite procesar todos los tokens a la vez, acelerando el entrenamiento frente a RNN/LSTM.
- Captura dependencias a largo plazo: conecta directamente cualquier par de posiciones, mejorando contexto y coherencia.
- Escalabilidad y transferencia: al aumentar parámetros y preentrenar en grandes corpus, obtiene rendimiento superior en muchas tareas mediante transfer learning.

--- Temperature=0.9 ---
[1] ¡Felicidades equipo! Este paper de IA impulsa la innovación y abre nuevas fronteras en aprendizaje automático. Bravo.
[2] - La autoatención permite capturar dependencias a larga distancia y paralelizar el entrenamiento, aprovechando mejor GPU/TPU.  
- Escalan muy bien: el preentrenamiento masivo y el fine-tuning permiten transferir conocimiento a muchas tareas.  
- Alto rendimiento y versatilidad: dominan SOTA en NLP y se aplican con éxito a visión, audio y modelos multimodales.
```

### Construcción de prompts reutilizables con `ChatPromptTemplate`

A continuación, se implementó el uso de plantillas de prompts con `ChatPromptTemplate.from_template()`, separando los roles de sistema, usuario e instrucciones dinámicas.

Esto permitió generar prompts parametrizados, por ejemplo:

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "Sos un asistente conciso, exacto y profesional."),
    ("human",  "Explicá {tema} en <= 3 oraciones, con un ejemplo real.")
])

chain = prompt | llm  # LCEL: prompt → LLM
print(chain.invoke({"tema": "atención multi-cabeza"}).content)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Sos un asistente conciso, exacto y profesional."),
    ("human",  "Explicá {tema} en <= 3 oraciones, con un ejemplo real."),
    ("human", "Tema: atención multi-cabeza, Respuesta: La atención multi‑cabeza ejecuta varias atenciones en paralelo con proyecciones distintas (Q/K/V) para capturar relaciones complementarias; sus salidas se concatenan y se mezclan, permitiendo que el modelo enfoque diferentes aspectos del contexto a la vez. Ejemplo: al traducir “El banco no concedió el crédito porque era muy arriesgado”, una cabeza puede alinear “banco→concedió”, otra atender la negación “no→concedió” y otra vincular “crédito→arriesgado” para desambiguar y mantener coherencia.")
])

chain = prompt | llm  # LCEL: prompt → LLM
print(chain.invoke({"tema": "LLM"}).content)
```

```python
La atención multi‑cabeza ejecuta varias atenciones en paralelo sobre diferentes proyecciones de Q, K y V; cada cabeza captura relaciones distintas (léxicas, sintácticas, de posición) y sus salidas se concatenan y re‑proyectan. 
Ejemplo real: en traducción EN→ES de “The cat that you saw ate fish”, una cabeza puede alinear “cat” con “ate” para mantener la concordancia sujeto‑verbo (“el gato … comió”), otra reorganiza la cláusula relativa “that you saw” → “que viste”, y otra mapea “fish” → “pescado”.

Un LLM (Large Language Model) es un modelo de aprendizaje profundo entrenado con enormes cantidades de texto para predecir la siguiente palabra, lo que le permite comprender y generar lenguaje natural, seguir instrucciones y realizar tareas como resumen, traducción o preguntas y respuestas. Ejemplo real: un equipo de soporte conecta su base de artículos al LLM y este, al recibir un ticket, redacta una respuesta coherente en el tono de la marca y genera un resumen del caso para el CRM.
```

Luego, mediante el operador `|` de LangChain Expression Language (LCEL), se encadenaron el *prompt template* y el modelo (`prompt | model`), consiguiendo una composición modular y fácilmente reutilizable.

Este enfoque fue probado para distintos temas (como *atención multi-cabeza*, *fine-tuning* o *embeddings*), obteniendo respuestas uniformes en tono y estructura. La reutilización de plantillas permitió garantizar consistencia entre llamadas y facilita la integración de este esquema en aplicaciones productivas.

### Salida estructurada mediante `with_structured_output(...)`

Para garantizar la consistencia del formato de salida, se utilizó el método `with_structured_output()` junto con modelos Pydantic que definen la estructura esperada de la respuesta.

```python
from typing import List
from pydantic import BaseModel

class Resumen(BaseModel):
    title: str
    bullets: List[str]

llm_json = llm.with_structured_output(Resumen)  # garantiza JSON válido que cumple el esquema

pedido = "Resumí en 3 bullets los riesgos de la 'prompt injection' en LLM apps."
res = llm_json.invoke(pedido)
res
```

Al invocar el modelo con esta configuración, las respuestas fueron devueltas en JSON validado, respetando el tipo de datos definido.

```python
Resumen(title='Riesgos de prompt injection en LLM apps', bullets=['Exfiltración de datos y secretos: el atacante induce al modelo a revelar prompts del sistema, PII, credenciales o a extraer información sensible vía conectores y bases internas.', 'Abuso de herramientas y acciones no autorizadas: la inyección subierte al agente para ejecutar llamadas a APIs, leer/editar/borrar datos, realizar transacciones o gastar recursos, incluso fuera de la intención del usuario.', 'Pérdida de integridad y cumplimiento: se anulan guardrails y políticas (jailbreak), se genera contenido malicioso o engañoso y se manipulan decisiones, con riesgos legales, reputacionales y de seguridad.'])
```

Se realizaron varios casos de prueba:

* Resúmenes temáticos, con lista de ideas principales.
* Traducciones, con campos `idioma_origen`, `idioma_destino` y `texto_traducido`.
* Extracción de entidades, donde cada ítem incluía `entidad` y `tipo` (por ejemplo, PERSONA, ORGANIZACIÓN, LUGAR).

```python
# Esqueleto sugerido para 1) y 2)
from pydantic import BaseModel

class Traduccion(BaseModel):
    text: str
    lang: str

traductor = llm.with_structured_output(Traduccion)
salida = traductor.invoke("Traducí al portugués: 'Excelente trabajo del equipo'.")
print(salida)

# Q&A con contexto (sin RAG)
from langchain_core.prompts import ChatPromptTemplate
QA_prompt = ChatPromptTemplate.from_messages([
    ("system", "Respondé SOLO usando el contexto. Si no alcanza, decí 'No suficiente contexto'."),
    ("human",  "Contexto:\n{contexto}\n\nPregunta: {pregunta}\nRespuesta breve:")
])
salida = (QA_prompt | llm).invoke({
    "contexto": "OpenAI y LangChain permiten structured output con JSON...",
    "pregunta": "¿Qué ventaja tiene structured output?"
})
print(salida)
```

```python
text='Excelente trabalho da equipe' lang='pt'
content='No suficiente contexto' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 332, 'prompt_tokens': 54, 'total_tokens': 386, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 320, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_provider': 'openai', 'model_name': 'gpt-5-2025-08-07', 'system_fingerprint': None, 'id': 'chatcmpl-Cb3waed34dssbbYuqdia4QiGOB1xr', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None} id='lc_run--85400561-a02a-4a0f-bd12-b98ea9fd3177-0' usage_metadata={'input_tokens': 54, 'output_tokens': 332, 'total_tokens': 386, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 320}}
```

```python
from typing import List, Optional
from pydantic import BaseModel
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-5-mini", temperature=0)

class Entidad(BaseModel):
    tipo: str   # p.ej., 'ORG', 'PER', 'LOC'
    valor: str

class ExtractInfo(BaseModel):
    titulo: Optional[str]
    fecha: Optional[str]
    entidades: List[Entidad]

extractor = llm.with_structured_output(ExtractInfo)
texto = "OpenAI anunció una colaboración con la Universidad Catolica del Uruguay en Montevideo el 05/11/2025."
extractor.invoke(f"Extraé titulo, fecha y entidades (ORG/PER/LOC) del siguiente texto:\n\n{texto}")
```

```python
ExtractInfo(titulo='OpenAI anunció colaboración con la Universidad Catolica del Uruguay', fecha='05/11/2025', entidades=[Entidad(tipo='ORG', valor='OpenAI'), Entidad(tipo='ORG', valor='Universidad Catolica del Uruguay'), Entidad(tipo='LOC', valor='Montevideo')])
```

Este enfoque permitió asegurar salidas estructuradas y verificables, adecuadas para integraciones en sistemas que requieran validación sintáctica o almacenamiento directo en bases de datos.

### Clasificación y *few-shot prompting*

Posteriormente, se abordó el problema de la clasificación de sentimientos.
Primero, se probó un enfoque zero-shot, simplemente indicando al modelo que clasificara una frase como *positiva*, *negativa* o *neutral*.

Luego, se aplicó un few-shot prompting, incorporando ejemplos previos en el contexto del prompt. Esto permitió comprobar cómo el modelo aprovecha ejemplos previos para inferir patrones de clasificación un poquito más estables.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-5-mini", temperature=0)

# Zero-shot
zs_prompt = ChatPromptTemplate.from_messages([
    ("system", "Sos un asistente conciso y exacto."),
    ("human",  "Clasificá el sentimiento de este texto como POS, NEG o NEU:\n\n{texto}")
])

# Few-shot (1–2 ejemplos)
fs_prompt = ChatPromptTemplate.from_messages([
    ("system", "Sos un asistente conciso y exacto. Clasificá cada texto en POS, NEG o NEU según el sentimiento general."),

    # POSITIVOS
    ("human", "Ejemplo:\nTexto: 'El producto superó mis expectativas'\nEtiqueta: POS"),
    ("human", "Ejemplo:\nTexto: 'Excelente servicio, todo llegó a tiempo'\nEtiqueta: POS"),
    ("human", "Ejemplo:\nTexto: 'Muy buena atención y calidad'\nEtiqueta: POS"),
    ("human", "Ejemplo:\nTexto: 'Está bastante bien, considerando el precio'\nEtiqueta: POS"),
    ("human", "Ejemplo:\nTexto: 'La interfaz es linda, aunque un poco lenta'\nEtiqueta: POS"),

    # NEGATIVOS
    ("human", "Ejemplo:\nTexto: 'La entrega fue tarde y vino roto'\nEtiqueta: NEG"),
    ("human", "Ejemplo:\nTexto: 'Increíble cómo logran empeorar con cada actualización'\nEtiqueta: NEG"),
    ("human", "Ejemplo:\nTexto: 'Gracias por arruinar mi día'\nEtiqueta: NEG"),
    ("human", "Ejemplo:\nTexto: 'Excelente... si te gusta esperar tres semanas por nada'\nEtiqueta: NEG"),
    ("human", "Ejemplo:\nTexto: 'Si eso era atención al cliente, prefiero no tenerla'\nEtiqueta: NEG"),

    # NEUTROS
    ("human", "Ejemplo:\nTexto: 'Cumplió su función básica'\nEtiqueta: NEU"),
    ("human", "Ejemplo:\nTexto: 'Está bien, nada extraordinario'\nEtiqueta: NEU"),
    ("human", "Ejemplo:\nTexto: 'No tengo una opinión fuerte al respecto'\nEtiqueta: NEU"),
    ("human", "Ejemplo:\nTexto: 'La experiencia fue diferente, sin ser buena ni mala'\nEtiqueta: NEU"),
    ("human", "Ejemplo:\nTexto: 'No me encantó, pero tampoco fue un desastre'\nEtiqueta: NEU"),

    # Consulta
    ("human", "Texto: {texto}\nEtiqueta:")
])

textos = [
    # Claros
    "Me encantó la experiencia, repetiría.",
    "No cumple lo prometido; decepcionante.",
    "Está bien, nada extraordinario.",
    # Ambiguos o irónicos
    "Bueno, al menos no explotó esta vez.",
    "Pensé que iba a ser peor, así que supongo que bien.",
    "Increíble cómo logran empeorar con cada actualización.",
    "Si eso era 'atención al cliente', prefiero no tenerla.",
    "No me encantó, pero tampoco fue un desastre.",
    "Excelente... si te gusta esperar tres semanas por nada.",
    "El sabor no estaba mal, aunque el olor daba miedo.",
    # Neutros / difíciles de contexto
    "El servicio fue lo que esperaba.",
    "Cumplió su función básica.",
    "No tengo una opinión fuerte al respecto.",
    "La experiencia fue diferente, sin ser buena ni mala.",
    "Fue interesante, aunque no volvería.",
    # Positivos con duda o condición
    "Está bastante bien, considerando el precio.",
    "La interfaz es linda, aunque un poco lenta.",
    "Buena idea, ejecución regular.",
    # Negativos disfrazados
    "Gracias por arruinar mi día.",
    "Definitivamente inolvidable… en el peor sentido.",
    "Qué detalle, enviarme algo que no pedí.",
    "El mejor error que he cometido en mucho tiempo.",
]

print("== Zero-shot ==")
for t in textos:
    print(t)
    print((zs_prompt | llm).invoke({"texto": t}).content)
    print("")

print("\n== Few-shot ==")
for t in textos:
    print(t)
    print((fs_prompt | llm).invoke({"texto": t}).content)
    print("")
```

```python
| #  | Texto                                                   | Zero-shot | Few-shot | Diferencia   |
| -- | ------------------------------------------------------- | --------- | -------- | ------------ |
| 1  | Me encantó la experiencia, repetiría.                   | POS       | POS      | ✅ Igual      |
| 2  | No cumple lo prometido; decepcionante.                  | NEG       | NEG      | ✅ Igual      |
| 3  | Está bien, nada extraordinario.                         | NEU       | NEU      | ✅ Igual      |
| 4  | Bueno, al menos no explotó esta vez.                    | POS       | NEG      | ⚠️ Diferente |
| 5  | Pensé que iba a ser peor, así que supongo que bien.     | POS       | POS      | ✅ Igual      |
| 6  | Increíble cómo logran empeorar con cada actualización.  | NEG       | NEG      | ✅ Igual      |
| 7  | Si eso era 'atención al cliente', prefiero no tenerla.  | NEG       | NEG      | ✅ Igual      |
| 8  | No me encantó, pero tampoco fue un desastre.            | NEU       | NEU      | ✅ Igual      |
| 9  | Excelente... si te gusta esperar tres semanas por nada. | NEG       | NEG      | ✅ Igual      |
| 10 | El sabor no estaba mal, aunque el olor daba miedo.      | NEU       | NEG      | ⚠️ Diferente |
| 11 | El servicio fue lo que esperaba.                        | NEU       | NEU      | ✅ Igual      |
| 12 | Cumplió su función básica.                              | NEU       | NEU      | ✅ Igual      |
| 13 | No tengo una opinión fuerte al respecto.                | NEU       | NEU      | ✅ Igual      |
| 14 | La experiencia fue diferente, sin ser buena ni mala.    | NEU       | NEU      | ✅ Igual      |
| 15 | Fue interesante, aunque no volvería.                    | NEG       | NEU      | ⚠️ Diferente |
| 16 | Está bastante bien, considerando el precio.             | POS       | POS      | ✅ Igual      |
| 17 | La interfaz es linda, aunque un poco lenta.             | POS       | POS      | ✅ Igual      |
| 18 | Buena idea, ejecución regular.                          | NEU       | NEU      | ✅ Igual      |
| 19 | Gracias por arruinar mi día.                            | NEG       | NEG      | ✅ Igual      |
| 20 | Definitivamente inolvidable… en el peor sentido.        | NEG       | NEG      | ✅ Igual      |
| 21 | Qué detalle, enviarme algo que no pedí.                 | NEG       | NEG      | ✅ Igual      |
| 22 | El mejor error que he cometido en mucho tiempo.         | POS       | POS      | ✅ Igual      |
```

### Procesamiento de textos largos y resumen jerárquico

Para abordar documentos extensos, se utilizó el componente `RecursiveCharacterTextSplitter`, que divide el texto en fragmentos de tamaño controlado (por ejemplo, 1000 caracteres con 100 de solapamiento).
Cada fragmento fue resumido individualmente y, posteriormente, se aplicó un segundo resumen sobre los resultados parciales, generando una síntesis jerárquica.

Este enfoque permitió manejar textos de gran longitud sin exceder los límites de tokens del modelo.
El resultado final fue un resumen global coherente y compacto, que conservaba la estructura conceptual original del texto técnico sobre GPT-5 y sus mecanismos de atención, entrenamiento y evaluación.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


long_text = """Texto largo ..."""

# Split en chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
chunks = splitter.split_text(long_text)

# Cadena para resumir un chunk
chunk_summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "Resumí el siguiente fragmento en 2–3 bullets, claros y factuales."),
    ("human", "{input}")
])

llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
chunk_summary = chunk_summary_prompt | llm

bullets = [chunk_summary.invoke({"input": c}).content for c in chunks]

# Reduce (combinar resultados)
reduce_prompt = ChatPromptTemplate.from_messages([
    ("system", "Consolidá bullets redundantes y producí un resumen único y breve."),
    ("human", "Bullets:\n{bullets}\n\nResumen final (<=120 tokens):")
])

final = (reduce_prompt | llm).invoke({"bullets": "\n".join(bullets)}).content
print(final)
```

```python
GPT‑5 es el modelo insignia: mejor desempeño en tareas agentivas, codificación, razonamiento y capacidad para seguir directrices, pensado para desarrolladores y llamadas a herramientas. La guía y el "prompt optimizer" ofrecen buenas prácticas (calibrar "agentic eagerness", limitar reasoning_effort, migrar a Responses API para flujos con herramientas) y recomiendan iterar. Incluye fundamentos de prompt engineering y consejos de parámetros: temperatura baja para tareas factuales, alta para creativo; ajustar solo temperatura o top_p. Ejemplos probados con text-davinci-003 (temp=0.7, top_p=1).
```

### Integración de recuperación y generación (RAG)

Finalmente, se construyó un flujo RAG (Retrieval-Augmented Generation) mínimo, utilizando FAISS como vector store y OpenAIEmbeddings para generar los embeddings semánticos.

El pipeline consistió en:

1. Indexar documentos fuente en el vector store.
2. Recuperar fragmentos relevantes según una consulta del usuario.
3. Combinar los textos recuperados con un prompt que restringía la respuesta al contenido de los documentos.

Este enfoque demostró cómo la recuperación mejora el *grounding* de las respuestas, reduciendo alucinaciones y asegurando que el modelo fundamente sus afirmaciones en información verificada.

```python
!pip install langchain-classic

from langchain_classic.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

# Documentos
docs_raw = [
    "LangChain soporta structured output con Pydantic.",
    "RAG combina recuperación + generación para mejor grounding.",
    "OpenAIEmbeddings facilita embeddings para indexar textos.",
    "De nuevo, si lo que dice la psicología positiva fuese cierto, las malas condiciones de vida, el pobre funcionamiento de ciertas instituciones."
]
docs = [Document(page_content=t) for t in docs_raw]

# Split y vector store
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(docs)

emb = OpenAIEmbeddings()
vs = FAISS.from_documents(chunks, embedding=emb)
retriever = vs.as_retriever(search_kwargs={"k": 4})

# LLM
llm = ChatOpenAI(model="gpt-5-mini", temperature=0)

# Prompt mejorado
prompt = ChatPromptTemplate.from_messages([
    ("system", "Usá el siguiente contexto para responder la pregunta. Si el contexto no alcanza, decí 'No suficiente contexto'."),
    ("human", "Contexto:\n{context}\n\nPregunta: {input}")
])

# Cadena de combinación (concatena texto de los docs)
combine_docs_chain = create_stuff_documents_chain(llm, prompt)

# RAG final
rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

# Invocación
response = rag_chain.invoke({"input": "¿Qué ventaja clave aporta RAG?"})
print(response["answer"])
```

```python
La ventaja clave es que RAG combina recuperación y generación para lograr un mejor grounding: las respuestas se apoyan en documentos recuperados, lo que mejora la precisión y relevancia (reduce alucinaciones) y permite incorporar conocimiento amplio o actualizado sin cargarlo todo en el modelo.
```

## Reflexión

La actividad permitió comprender cómo LangChain facilita la integración de modelos como GPT-5, ofreciendo herramientas para controlar la generación de texto, estructurar la salida y reutilizar prompts de manera eficiente.

* Control del comportamiento del modelo: Ajustar parámetros como `temperature` que influye directamente en la creatividad y coherencia de las respuestas, permitiendo adaptar la generación según la tarea específica.
* Reutilización y consistencia mediante plantillas: `ChatPromptTemplate` y la LangChain Expression Language permiten construir prompts parametrizados, modulares y fáciles de mantener, garantizando uniformidad en las respuestas y facilitando la escalabilidad de flujos de trabajo.
* Salida estructurada confiable: Usar `with_structured_output(...)` y modelos Pydantic asegura que las respuestas del modelo cumplan un esquema definido, evitando ambigüedades y errores en post-procesamiento. Esto es clave para aplicaciones que integran LLMs en sistemas de información o bases de datos.

### Aprendizajes

* Comprender la relación entre parámetros de decodificación y calidad de la salida es fundamental para controlar la creatividad y coherencia del modelo.
* La modularidad de los prompts y la separación de roles (sistema/usuario/instrucciones) mejora la reutilización y facilita la adaptación a distintos escenarios.
* La validación de salidas mediante Pydantic reduce errores en la interpretación de la respuesta y asegura consistencia.
* Few-shot prompting aporta mayor precisión en tareas de clasificación y extracción de información.
* Dividir y resumir textos largos permite manejar límites de tokens sin perder coherencia, y sienta las bases para flujos más complejos como RAG.

## Referencias

https://colab.research.google.com/drive/19QGlehQSbDH3bdQBA-KoLajT9S2SaM63?usp=sharing