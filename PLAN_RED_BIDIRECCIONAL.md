# 🧠 PLAN: RED NEURONAL BIDIRECCIONAL - TP2 AA2

## 🎯 OBJETIVO PRINCIPAL
Implementar una **RNN bidireccional** que prediga simultáneamente:
1. **Puntuación inicial** (¿ o sin puntuación)
2. **Puntuación final** (coma, punto, ? o sin puntuación) 
3. **Capitalización** (0=minus, 1=title, 2=mixed, 3=upper)

---

## 📋 REQUISITOS ESTRICTOS DE LA CONSIGNA

### ✅ **Arquitectura Obligatoria**
- [x] **RNN bidireccional** (forward + backward)
- [x] Puede usar **LSTM** o **GRU** como celdas
- [x] **Multi-tarea**: 3 salidas simultáneas
- [x] Entrada: embeddings de `bert-base-multilingual-cased` (768 dim)

### ✅ **Datos de Entrada**
- [x] Tokens procesados con tokenizador BERT multilingual
- [x] Embeddings estáticos de BERT (no contextuales)
- [x] Predicciones a **nivel de token** (no palabra)

### ✅ **Tareas y Labels**
1. **Puntuación inicial**: 
   - `0` = sin puntuación
   - `1` = ¿ (signo de pregunta apertura)

2. **Puntuación final**:
   - `0` = coma (,)
   - `1` = punto (.)
   - `2` = signo pregunta cierre (?)
   - `3` = sin puntuación

3. **Capitalización**:
   - `0` = todo minúsculas ("hola")
   - `1` = primera mayúscula ("Hola") 
   - `2` = mixta ("McDonald's", "iPhone")
   - `3` = todo mayúsculas ("UBA", "NASA")

### ✅ **Evaluación**
- **Métrica**: F1 macro para cada tarea
- **Nivel**: Evaluación a nivel palabra (no token)
- **Clases**:
  - Punt. inicial: 2 clases (¿, sin)
  - Punt. final: 4 clases (coma, punto, ?, sin)
  - Capitalización: 4 clases (0, 1, 2, 3)

---

## 🏗️ ARQUITECTURA DETALLADA

### **1. Capa de Entrada**
```python
# Input shape: (batch_size, sequence_length, 768)
# 768 = dimensión embeddings BERT
input_layer = Input(shape=(None, 768))
```

### **2. Capa Bidireccional**
```python
# Bidirectional LSTM/GRU
# return_sequences=True para predicción por token
bidirectional_layer = Bidirectional(
    LSTM(units=128, return_sequences=True),
    merge_mode='concat'  # concat forward + backward
)
# Output shape: (batch_size, sequence_length, 256)
```

### **3. Capas de Salida Multi-tarea**
```python
# 3 cabezas independientes para cada tarea
punct_initial_head = Dense(2, activation='softmax', name='punct_initial')
punct_final_head = Dense(4, activation='softmax', name='punct_final') 
capitalization_head = Dense(4, activation='softmax', name='capitalization')
```

### **4. Función de Pérdida Multi-tarea**
```python
losses = {
    'punct_initial': 'sparse_categorical_crossentropy',
    'punct_final': 'sparse_categorical_crossentropy',
    'capitalization': 'sparse_categorical_crossentropy'
}

# Pesos balanceados según distribución de clases
loss_weights = {
    'punct_initial': 1.0,
    'punct_final': 1.0, 
    'capitalization': 2.0  # Más peso por desbalance
}
```

---

## 📊 MANEJO DE SECUENCIAS

### **Problema: Secuencias de Longitud Variable**
- Los textos tienen diferentes longitudes
- BERT tokeniza en sub-tokens (##)
- Necesitamos padding/truncation

### **Solución: Estrategia de Secuencias**
1. **Chunking**: Dividir secuencias muy largas
2. **Padding**: Rellenar secuencias cortas
3. **Masking**: Ignorar tokens de padding en pérdida

```python
# Parámetros de secuencia
MAX_SEQUENCE_LENGTH = 512  # Límite BERT
PADDING_VALUE = 0
```

---

## 🎛️ HIPERPARÁMETROS INICIALES

### **Arquitectura**
- **Unidades LSTM**: 128 (bidireccional → 256 total)
- **Dropout**: 0.3 (prevenir overfitting)
- **Recurrent dropout**: 0.2

### **Entrenamiento**
- **Batch size**: 32 (ajustar según memoria)
- **Learning rate**: 0.001 (Adam optimizer)
- **Epochs**: 50 (con early stopping)
- **Validation split**: Usar val set existente

### **Callbacks**
- **EarlyStopping**: patience=5, monitor='val_loss'
- **ModelCheckpoint**: guardar mejor modelo
- **ReduceLROnPlateau**: reducir LR si no mejora

---

## 🔄 PIPELINE DE ENTRENAMIENTO

### **Paso 1: Carga de Datos**
```python
# Cargar embeddings y labels
X_train = np.load('bert_features/X_train.npy')
y_punct_initial_train = np.load('bert_features/y_punt_inicial_train.npy')
y_punct_final_train = np.load('bert_features/y_punt_final_train.npy')
y_capitalization_train = np.load('bert_features/y_capitalizacion_train.npy')
```

### **Paso 2: Preparación de Secuencias**
```python
# Agrupar tokens por instancia_id
# Crear secuencias de longitud variable
# Aplicar padding/truncation
```

### **Paso 3: Construcción del Modelo**
```python
# Definir arquitectura bidireccional
# Compilar con optimizador y métricas
# Mostrar resumen del modelo
```

### **Paso 4: Entrenamiento**
```python
# Fit con validation data
# Callbacks para monitoreo
# Guardar métricas de entrenamiento
```

### **Paso 5: Evaluación**
```python
# Predicciones en test set
# Cálculo de F1 macro por tarea
# Matriz de confusión
# Análisis de errores
```

---

## ⚠️ REGLAS Y CONSIDERACIONES CRÍTICAS

### **🚨 REGLA 1: Manejo de Secuencias**
- **NUNCA** mezclar tokens de diferentes instancias
- **SIEMPRE** mantener correspondencia token ↔ label
- **CUIDADO** con el padding: usar masking en la pérdida

### **🚨 REGLA 2: Multi-tarea**
- **3 salidas simultáneas** (no modelos separados)
- **Pesos de pérdida** balanceados por desbalance de clases
- **Métricas independientes** para cada tarea

### **🚨 REGLA 3: Evaluación Correcta**
- **F1 macro** (no weighted, no micro)
- **Por tarea separada** (3 métricas F1)
- **Nivel palabra** en evaluación final (aunque entrenemos por token)

### **🚨 REGLA 4: Reproducibilidad**
- **Seeds fijos** para numpy, tensorflow, random
- **Guardar configuración** completa del modelo
- **Logs detallados** de entrenamiento

### **🚨 REGLA 5: Manejo de Memoria**
- **Batch size adaptativo** según GPU disponible
- **Gradient checkpointing** si es necesario
- **Liberación de memoria** entre pasos

---

## 📈 MÉTRICAS Y MONITOREO

### **Durante Entrenamiento**
- Loss total y por tarea
- Accuracy por tarea
- Validation metrics
- Learning rate actual

### **Evaluación Final**
- **F1 macro** por tarea (MÉTRICA PRINCIPAL)
- Precision y Recall por clase
- Matriz de confusión
- Análisis de casos difíciles

---

## 🚀 PLAN DE IMPLEMENTACIÓN

### **Fase 1: Preparación de Datos** ✅
- [x] Cargar embeddings BERT
- [x] Verificar shapes y tipos
- [ ] Agrupar por secuencias (instancia_id)
- [ ] Implementar padding/truncation

### **Fase 2: Arquitectura**
- [ ] Definir modelo bidireccional
- [ ] Implementar multi-tarea
- [ ] Configurar pérdidas y métricas

### **Fase 3: Entrenamiento**
- [ ] Pipeline de entrenamiento
- [ ] Callbacks y monitoreo
- [ ] Validación en val set

### **Fase 4: Evaluación**
- [ ] Predicciones en test
- [ ] Cálculo F1 macro
- [ ] Análisis de resultados

### **Fase 5: Optimización**
- [ ] Tuning de hiperparámetros
- [ ] Análisis de errores
- [ ] Mejoras de arquitectura

---

## 🎯 CRITERIOS DE ÉXITO

### **Mínimo Aceptable**
- Modelo entrena sin errores
- F1 > 0.5 en las 3 tareas
- Predicciones coherentes

### **Objetivo Deseable**
- F1 > 0.7 en puntuación
- F1 > 0.8 en capitalización
- Convergencia estable

### **Excelencia**
- F1 > 0.85 en todas las tareas
- Análisis detallado de errores
- Comparación con baselines

---

## 📁 ESTRUCTURA DE ARCHIVOS

```
neural_networks.py          # Implementación principal
models/                     # Modelos guardados
├── bidirectional_model.h5
├── model_config.json
└── training_history.pkl
results/                    # Resultados y métricas
├── training_curves.png
├── confusion_matrices.png
├── f1_scores.json
└── error_analysis.txt
```

---

## 🔧 DEBUGGING Y TROUBLESHOOTING

### **Problemas Comunes**
1. **OOM (Out of Memory)**: Reducir batch_size
2. **Nan Loss**: Verificar learning rate, gradient clipping
3. **No convergencia**: Ajustar arquitectura, datos
4. **Overfitting**: Más dropout, regularización

### **Verificaciones Críticas**
- [ ] Shapes de entrada y salida
- [ ] Correspondencia token-label
- [ ] Distribución de clases balanceada
- [ ] Métricas calculadas correctamente

---

## 📝 NOTAS FINALES

- **Seguir consigna AL PIE DE LA LETRA**
- **Documentar todas las decisiones**
- **Código limpio y comentado**
- **Resultados reproducibles**

¡VAMOS A CREAR UNA RED BIDIRECCIONAL ÉPICA! 🚀🧠 