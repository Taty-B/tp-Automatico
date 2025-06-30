# %%
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer



# %%
"""
CELDA 1: CARGA Y FUSIÓN DE DATOS
"""

def load_csv_data(file_path):
    """Cargar CSV con formato: id, lang, text"""
    return pd.read_csv(file_path, sep='\t', header=None, names=['id', 'lang', 'text'], encoding='utf-8')

# Cargar datasets con muestreo estratégico para mejor balance
df_sentences = load_csv_data('datasets/spanish_sentences.csv')
# Reducir spanish_sentences al 20% para mejor balance de clases
df_sentences = df_sentences.sample(frac=0.20, random_state=42)
print(f"✅ spanish_sentences.csv: {len(df_sentences):,} oraciones (20% para balance)")

df_questions = load_csv_data('datasets/xquad_preguntas_espanol.csv') 
print(f"✅ xquad_preguntas_espanol.csv: {len(df_questions):,} oraciones (100%)")

df_synthetic = load_csv_data('datasets/synthetic_sentences_3500.csv')
print(f"✅ synthetic_sentences_3500.csv: {len(df_synthetic):,} oraciones (100%)")

# Fusionar todos los datasets
df_original = pd.concat([df_sentences, df_questions, df_synthetic], ignore_index=True)



# %%
"""
CELDA 2: ANÁLISIS INICIAL
========================
Analizar las características del dataset antes de la limpieza
"""

print("🔍 ANÁLISIS INICIAL DEL DATASET")
print("=" * 50)

# Estadísticas de longitud en caracteres
char_lengths = df_original['text'].str.len()
print(f"📏 Longitud en caracteres:")
print(f"   Promedio: {char_lengths.mean():.1f}")
print(f"   Mediana: {char_lengths.median():.1f}")
print(f"   Min: {char_lengths.min()}")
print(f"   Max: {char_lengths.max()}")

# Estadísticas de longitud en palabras
word_counts = df_original['text'].str.split().str.len()
print(f"\n📏 Longitud en palabras:")
print(f"   Promedio: {word_counts.mean():.1f}")
print(f"   Mediana: {word_counts.median():.1f}")
print(f"   Min: {word_counts.min()}")
print(f"   Max: {word_counts.max()}")

# Distribución de longitudes
print(f"\n📊 Distribución de palabras por oración:")
print(f"   1-5 palabras: {((word_counts >= 1) & (word_counts <= 5)).sum():,}")
print(f"   6-10 palabras: {((word_counts >= 6) & (word_counts <= 10)).sum():,}")
print(f"   11-20 palabras: {((word_counts >= 11) & (word_counts <= 20)).sum():,}")
print(f"   21-30 palabras: {((word_counts >= 21) & (word_counts <= 30)).sum():,}")
print(f"   >30 palabras: {(word_counts > 30).sum():,}")


# %%
"""
CELDA 3: APLICAR FILTROS DE LIMPIEZA
===================================
Aplicar todos los filtros de limpieza paso a paso
"""

def clean_dataset_step_by_step(df):
    print("🧹 APLICANDO LIMPIEZA ESTRICTA AL DATASET")
    print("=" * 50)
    
    original_count = len(df)
    df_clean = df.copy()
    
    # 1. Filtrar oraciones muy cortas
    before = len(df_clean)
    df_clean = df_clean[df_clean['text'].str.len() >= 3]
    print(f"✅ Paso 1 - Filtradas oraciones < 3 caracteres: {before - len(df_clean):,} eliminadas")
    
    # 2. Filtrar oraciones con signos de exclamación
    before = len(df_clean)
    df_clean = df_clean[~df_clean['text'].str.contains(r'[¡!]', regex=True, na=False)]
    print(f"✅ Paso 2 - Filtradas oraciones con exclamaciones (¡!): {before - len(df_clean):,} eliminadas")
    
    # 3. Filtrar oraciones con caracteres raros/especiales (MENOS RESTRICTIVO para preservar mixed case)
    before = len(df_clean)
    # Permitir más caracteres para preservar marcas, acrónimos, etc.
    valid_pattern = r'^[a-zA-ZáéíóúñüÁÉÍÓÚÑÜ0-9\s¿?.,\'\-()°ªº/&%]+$'
    df_clean = df_clean[df_clean['text'].str.match(valid_pattern, na=False)]
    print(f"✅ Paso 3 - Filtradas oraciones con caracteres raros: {before - len(df_clean):,} eliminadas")
    
    
    
    # 5. Filtrar oraciones con muchos números (probablemente no útiles)
    before = len(df_clean)
    # Eliminar oraciones donde más del 30% son números
    def has_too_many_numbers(text):
        if not text:
            return True
        total_chars = len(text.replace(' ', ''))
        if total_chars == 0:
            return True
        number_chars = sum(1 for c in text if c.isdigit())
        return (number_chars / total_chars) > 0.3
    
    df_clean = df_clean[~df_clean['text'].apply(has_too_many_numbers)]
    print(f"✅ Paso 5 - Filtradas oraciones con muchos números: {before - len(df_clean):,} eliminadas")
    
    # 6. Filtrar por longitud de palabras
    before = len(df_clean)
    word_counts = df_clean['text'].str.split().str.len()
    df_clean = df_clean[(word_counts >= 5) & (word_counts <= 30)]
    print(f"✅ Paso 6 - Filtradas oraciones fuera del rango 5-30 palabras: {before - len(df_clean):,} eliminadas")
    
    # 7. Normalizar espacios en blanco múltiples
    df_clean['text'] = df_clean['text'].str.replace(r'\s+', ' ', regex=True).str.strip()
    print(f"✅ Paso 7 - Normalizados espacios en blanco")
    
    # 8. Filtrar oraciones duplicadas
    before = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=['text'], keep='first')
    print(f"✅ Paso 8 - Eliminadas oraciones duplicadas: {before - len(df_clean):,} eliminadas")
    
    # 9. Filtrar oraciones muy repetitivas (con palabras muy repetidas)
    before = len(df_clean)
    def is_too_repetitive(text):
        if not text:
            return True
        words = text.lower().split()
        if len(words) < 5:
            return False
        # Si más del 50% de las palabras se repiten, es muy repetitiva
        unique_words = len(set(words))
        return (unique_words / len(words)) < 0.5
    
    df_clean = df_clean[~df_clean['text'].apply(is_too_repetitive)]
    print(f"✅ Paso 9 - Filtradas oraciones muy repetitivas: {before - len(df_clean):,} eliminadas")
    
    print(f"\n📊 RESUMEN DE LIMPIEZA ESTRICTA:")
    print(f"   Oraciones originales: {original_count:,}")
    print(f"   Oraciones después de limpieza: {len(df_clean):,}")
    print(f"   Oraciones eliminadas: {original_count - len(df_clean):,} ({(original_count - len(df_clean))/original_count*100:.1f}%)")
    print(f"   Oraciones conservadas: {len(df_clean)/original_count*100:.1f}%")
    print(f"   🎯 Reducción significativa del dataset para mejor calidad")
    
    return df_clean

df_clean = clean_dataset_step_by_step(df_original)

# Guardar dataset limpio completo
print(f"\n💾 GUARDANDO DATASET LIMPIO COMPLETO")
df_clean.to_csv('spanish_clean_all.csv', sep='\t', index=False, header=False)
print(f"✅ Guardado: spanish_clean_all.csv ({len(df_clean):,} oraciones)")


# %%
"""
CELDA 4: ANÁLISIS DEL DATASET LIMPIO
===================================
Analizar las características del dataset después de la limpieza
"""

print("📊 ANÁLISIS DEL DATASET LIMPIO")
print("=" * 50)

# Estadísticas de longitud en palabras
word_counts_clean = df_clean['text'].str.split().str.len()
print(f"📏 Longitud en palabras (después de limpieza):")
print(f"   Promedio: {word_counts_clean.mean():.1f} palabras")
print(f"   Mediana: {word_counts_clean.median():.1f} palabras")
print(f"   Min: {word_counts_clean.min()} palabras")
print(f"   Max: {word_counts_clean.max()} palabras")

# Distribución de puntuación
punct_stats = {}
punct_stats['con_punto'] = (df_clean['text'].str.contains(r'\.', regex=True)).sum()
punct_stats['con_coma'] = (df_clean['text'].str.contains(r',', regex=True)).sum()
punct_stats['preguntas'] = (df_clean['text'].str.contains(r'[¿?]', regex=True)).sum()
punct_stats['sin_puntuacion'] = (~df_clean['text'].str.contains(r'[.!?¿¡,;:]', regex=True)).sum()

print(f"\n📝 Distribución de puntuación:")
for key, value in punct_stats.items():
    print(f"   {key.replace('_', ' ').title()}: {value:,} ({value/len(df_clean)*100:.1f}%)")

# Capitalización
cap_stats = {}
cap_stats['todo_minusculas'] = (df_clean['text'].str.islower()).sum()
cap_stats['primera_mayuscula'] = (df_clean['text'].str.match(r'^[A-ZÁÉÍÓÚÑÜ]', na=False)).sum()
cap_stats['con_acronimos'] = (df_clean['text'].str.contains(r'\b[A-ZÁÉÍÓÚÑÜ]{2,}\b', regex=True)).sum()

print(f"\n🔤 Distribución de capitalización:")
for key, value in cap_stats.items():
    print(f"   {key.replace('_', ' ').title()}: {value:,} ({value/len(df_clean)*100:.1f}%)")


# %%
"""
CELDA 5: DIVISIÓN EN TRAIN/VALIDATION/TEST
=========================================
Dividir el dataset limpio en conjuntos de entrenamiento, validación y prueba
"""

def split_dataset(df, train_size=0.8, val_size=0.1, test_size=0.1, random_state=42):
    """Dividir dataset en train/validation/test"""
    print(f"📂 DIVIDIENDO DATASET LIMPIO")
    print("=" * 50)
    
    try:
        # Primera división: train vs (val + test)
        train_df, temp_df = train_test_split(
            df, 
            test_size=(val_size + test_size), 
            random_state=random_state,
            shuffle=True
        )
        
        # Segunda división: val vs test
        val_df, test_df = train_test_split(
            temp_df,
            test_size=test_size/(val_size + test_size),
            random_state=random_state,
            shuffle=True
        )
        
        print(f"✅ Train: {len(train_df):,} oraciones ({len(train_df)/len(df)*100:.1f}%)")
        print(f"✅ Validation: {len(val_df):,} oraciones ({len(val_df)/len(df)*100:.1f}%)")
        print(f"✅ Test: {len(test_df):,} oraciones ({len(test_df)/len(df)*100:.1f}%)")
        
        return train_df, val_df, test_df

    except Exception as e:
        print(f"❌ Error dividiendo dataset: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

train_df, val_df, test_df = split_dataset(df_clean)

print(f"\n📊 VERIFICACIÓN DE DISTRIBUCIONES:")
for name, dataset in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
    questions = (dataset['text'].str.contains(r'[¿?]', regex=True)).sum()
    points = (dataset['text'].str.contains(r'\.', regex=True)).sum()
    print(f"   {name}: {questions} preguntas ({questions/len(dataset)*100:.1f}%), {points} con punto ({points/len(dataset)*100:.1f}%)")

def save_split_datasets(train_df, val_df, test_df, base_name="spanish_clean"):
    """Guardar los datasets divididos"""
    print(f"\n💾 GUARDANDO DATASETS DIVIDIDOS")
    print("=" * 50)
    
    # Guardar versiones SIN headers (formato del proyecto)
    train_df.to_csv(f'{base_name}_train.csv', sep='\t', index=False, header=False)
    val_df.to_csv(f'{base_name}_val.csv', sep='\t', index=False, header=False)
    test_df.to_csv(f'{base_name}_test.csv', sep='\t', index=False, header=False)
    
    print(f"✅ Guardado: {base_name}_train.csv ({len(train_df):,} oraciones)")
    print(f"✅ Guardado: {base_name}_val.csv ({len(val_df):,} oraciones)")
    print(f"✅ Guardado: {base_name}_test.csv ({len(test_df):,} oraciones)")
    
    return f'{base_name}_train.csv', f'{base_name}_val.csv', f'{base_name}_test.csv'

train_file, val_file, test_file = save_split_datasets(train_df, val_df, test_df)


# %%
"""
CELDA 6: SIMULACIÓN ASR
"""

def simulate_asr_output(text):
    """Simular salida de sistema ASR: texto plano, minúsculas, sin puntuación"""
    # Remover toda la puntuación
    asr_text = re.sub(r'[¿?.,;:!¡()"\'-]', '', text)
    
    # Convertir a minúsculas
    asr_text = asr_text.lower()
    
    # Normalizar espacios múltiples
    asr_text = re.sub(r'\s+', ' ', asr_text).strip()
    
    return asr_text

def add_asr_simulation(df):
    """Agregar columna con simulación ASR a cada dataset"""
    df_asr = df.copy()
    df_asr['asr_text'] = df_asr['text'].apply(simulate_asr_output)
    return df_asr



train_df_asr = add_asr_simulation(train_df)
val_df_asr = add_asr_simulation(val_df)
test_df_asr = add_asr_simulation(test_df)

print(f"✅ Train: {len(train_df_asr):,} oraciones con simulación ASR")
print(f"✅ Val: {len(val_df_asr):,} oraciones con simulación ASR")
print(f"✅ Test: {len(test_df_asr):,} oraciones con simulación ASR")

# Ejemplo de transformación ASR
print(f"\n📝 Ejemplo de simulación ASR:")
sample_text = train_df_asr.iloc[0]['text']
sample_asr = train_df_asr.iloc[0]['asr_text']
print(f"Original: {sample_text}")
print(f"ASR:      {sample_asr}")



# %%
"""
CELDA 7: TOKENIZACIÓN CON BERT
"""

def extract_labels_from_text(text):
    """Extraer etiquetas de puntuación y capitalización del texto original"""
    words = text.split()
    labels = []
    
    for i, word in enumerate(words):
        original_word = word
        
        # Extraer puntuación inicial - solo ¿ o vacío
        punt_inicial = ""
        if word.startswith("¿"):
            punt_inicial = "¿"
            word = word[1:]  # Remover el signo
        
        # Extraer puntuación final - solo .,? o vacío
        punt_final = ""
        if word.endswith("?"):
            punt_final = "?"
            word = word[:-1]
        elif word.endswith("."):
            punt_final = "."
            word = word[:-1]
        elif word.endswith(","):
            punt_final = ","
            word = word[:-1]
        # Otros signos de puntuación se ignoran para este ejercicio
        
        # Determinar capitalización 0,1,2,3 (usando palabra sin puntuación)
        if not word:  # Si la palabra quedó vacía después de quitar puntuación
            capitalizacion = 0
        elif len(word) >= 2 and word.isupper():
            capitalizacion = 3  # Todo mayúsculas (acrónimos como NASA, UBA)
        elif word.islower():
            capitalizacion = 0  # Todo minúsculas (hola)
        elif len(word) >= 2 and word[0].isupper() and word[1:].islower():
            capitalizacion = 1  # Primera mayúscula (Hola)
        elif any(c.isupper() for c in word) and any(c.islower() for c in word):
            capitalizacion = 2  # Capitalización mixta (McDonald's, iPhone)
        else:
            # Casos especiales: palabras de 1 letra
            capitalizacion = 1 if word[0].isupper() else 0
        
        labels.append({
            'original_word': original_word,
            'clean_word': word,
            'punt_inicial': punt_inicial,
            'punt_final': punt_final,
            'capitalizacion': capitalizacion
        })
    
    return labels

def align_tokens_with_labels(tokens, labels):
    """
    Alinear tokens BERT con las etiquetas extraídas
    CORREGIDO según consigna:
    - punt_inicial: solo en el PRIMER token de la palabra
    - punt_final: solo en el ÚLTIMO token de la palabra  
    - capitalizacion: clase máxima entre sub-tokens
    """
    aligned_data = []
    label_idx = 0
    current_word_tokens = []
    
    for token in tokens:
        if token.startswith("##"):
            # Es continuación de palabra anterior
            current_word_tokens.append(token)
        else:
            # Es inicio de nueva palabra
            if current_word_tokens and label_idx < len(labels):
                # Distribuir etiquetas CORRECTAMENTE según consigna
                label = labels[label_idx]
                
                for i, wt in enumerate(current_word_tokens):
                    # PUNTUACIÓN INICIAL: solo primer token
                    punt_inicial = label['punt_inicial'] if i == 0 else ''
                    
                    # PUNTUACIÓN FINAL: solo último token
                    punt_final = label['punt_final'] if i == len(current_word_tokens) - 1 else ''
                    
                    # CAPITALIZACIÓN: todos los tokens tienen la misma clase
                    # (la evaluación word-level tomará el máximo después)
                    capitalizacion = label['capitalizacion']
                    
                    aligned_data.append({
                        'token': wt,
                        'punt_inicial': punt_inicial,
                        'punt_final': punt_final,
                        'capitalizacion': capitalizacion
                    })
                
                label_idx += 1
            
            # Iniciar nueva palabra
            current_word_tokens = [token]
    
    # Procesar última palabra
    if current_word_tokens and label_idx < len(labels):
        label = labels[label_idx]
        
        for i, wt in enumerate(current_word_tokens):
            # PUNTUACIÓN INICIAL: solo primer token
            punt_inicial = label['punt_inicial'] if i == 0 else ''
            
            # PUNTUACIÓN FINAL: solo último token
            punt_final = label['punt_final'] if i == len(current_word_tokens) - 1 else ''
            
            # CAPITALIZACIÓN: todos los tokens tienen la misma clase
            capitalizacion = label['capitalizacion']
            
            aligned_data.append({
                'token': wt,
                'punt_inicial': punt_inicial,
                'punt_final': punt_final,
                'capitalizacion': capitalizacion
            })
    
    return aligned_data

def tokenize_dataset(df, dataset_name):
    """Tokenizar dataset ASR y extraer etiquetas del texto original"""
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    
    tokenized_data = []
    
    for idx, row in df.iterrows():
        original_text = row['text']  # Texto original con puntuación y mayúsculas
        asr_text = row['asr_text']   # Texto ASR simulado (entrada del modelo)
        
        # Extraer etiquetas del texto original (targets)
        labels = extract_labels_from_text(original_text)
        
        # Tokenizar texto ASR (entrada del modelo)
        tokens = tokenizer.tokenize(asr_text)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        # Alinear tokens con etiquetas
        aligned_tokens = align_tokens_with_labels(tokens, labels)
        
        # Crear entrada para cada token
        for token_idx, (token, token_id, aligned) in enumerate(zip(tokens, token_ids, aligned_tokens)):
            tokenized_data.append({
                'instancia_id': row['id'],
                'token_id': token_id,
                'token': token,
                'punt_inicial': aligned['punt_inicial'] if aligned['punt_inicial'] else '',
                'punt_final': aligned['punt_final'] if aligned['punt_final'] else '',
                'capitalizacion': aligned['capitalizacion']
            })
    
    df_tokenized = pd.DataFrame(tokenized_data)
    
    # Guardar dataset tokenizado con etiquetas
    output_file = f'tokenized_{dataset_name}.csv'
    df_tokenized.to_csv(output_file, index=False)
    print(f"✅ Tokenizado y guardado: {output_file} ({len(df_tokenized):,} tokens)")
    
    return df_tokenized

# Tokenizar cada split por separado (usando datasets con simulación ASR)
print("🔤 TOKENIZANDO DATASETS CON SIMULACIÓN ASR")
print("=" * 50)

train_tokenized = tokenize_dataset(train_df_asr, 'train')
val_tokenized = tokenize_dataset(val_df_asr, 'val') 
test_tokenized = tokenize_dataset(test_df_asr, 'test')


# %%
"""
CELDA 8: SMOKE TEST - DESTOKENIZACIÓN Y VALIDACIÓN
"""

def detokenize_and_reconstruct(df_tokenized):
    """Reconstruir oraciones aplicando las etiquetas predichas"""
    reconstructed_sentences = {}
    
    # Agrupar tokens por instancia
    for _, row in df_tokenized.iterrows():
        instancia_id = row['instancia_id']
        
        if instancia_id not in reconstructed_sentences:
            reconstructed_sentences[instancia_id] = []
        
        reconstructed_sentences[instancia_id].append({
            'token': row['token'],
            'punt_inicial': row['punt_inicial'],
            'punt_final': row['punt_final'],
            'capitalizacion': row['capitalizacion']
        })
    
    # Reconstruir cada oración
    results = []
    for instancia_id, tokens in reconstructed_sentences.items():
        reconstructed_text = reconstruct_sentence_from_tokens(tokens)
        results.append({
            'instancia_id': instancia_id,
            'reconstructed_text': reconstructed_text
        })
    
    return pd.DataFrame(results)

def reconstruct_sentence_from_tokens(tokens):
    """Reconstruir una oración a partir de tokens y sus etiquetas"""
    words = []
    word_info = []
    current_word = ""
    current_capitalizacion = 0
    current_punt_inicial = ""
    current_punt_final = ""
    
    # Primero, reconstruir palabras completas agrupando tokens
    for i, token_info in enumerate(tokens):
        token = token_info['token']
        
        if token.startswith("##"):
            # Es continuación de palabra
            current_word += token[2:]  # Remover ##
            # La puntuación final se toma del último token de la palabra
            current_punt_final = token_info['punt_final']
        else:
            # Es inicio de nueva palabra
            if current_word:  # Si hay palabra anterior, guardarla
                words.append(apply_capitalization(current_word, current_capitalizacion))
                word_info.append({
                    'punt_inicial': current_punt_inicial,
                    'punt_final': current_punt_final
                })
            
            # Iniciar nueva palabra
            current_word = token
            current_capitalizacion = token_info['capitalizacion']
            current_punt_inicial = token_info['punt_inicial']
            current_punt_final = token_info['punt_final']
        
        # Si es el último token, procesar la palabra final
        if i == len(tokens) - 1:
            words.append(apply_capitalization(current_word, current_capitalizacion))
            word_info.append({
                'punt_inicial': current_punt_inicial,
                'punt_final': current_punt_final
            })
    
    # Reconstruir oración con puntuación
    sentence = ""
    for i, (word, info) in enumerate(zip(words, word_info)):
        # Agregar puntuación inicial
        if info['punt_inicial']:
            sentence += info['punt_inicial']
        
        # Agregar palabra
        sentence += word
        
        # Agregar puntuación final
        if info['punt_final']:
            sentence += info['punt_final']
        
        # Agregar espacio (excepto al final)
        if i < len(words) - 1:
            sentence += " "
    
    return sentence

def apply_capitalization(word, capitalizacion):
    """Aplicar capitalización según el código"""
    if capitalizacion == 0:
        return word.lower()  # Todo minúsculas
    elif capitalizacion == 1:
        return word.capitalize()  # Primera mayúscula
    elif capitalizacion == 2:
        # Capitalización mixta - mantener como está o aplicar reglas específicas
        return word  # Por simplicidad, mantener como está
    elif capitalizacion == 3:
        return word.upper()  # Todo mayúsculas
    else:
        return word

def compare_with_original(df_reconstructed, df_original):
    """Comparar oraciones reconstruidas con las originales"""
    print("🔍 COMPARACIÓN: RECONSTRUIDAS vs ORIGINALES")
    print("=" * 60)
    
    matches = 0
    total = 0
    failed_examples = []
    
    # Crear diccionario de originales por ID
    original_dict = {row['id']: row['text'] for _, row in df_original.iterrows()}
    
    print("⏳ Procesando comparaciones...")
    
    for _, row in df_reconstructed.iterrows():
        instancia_id = row['instancia_id']
        reconstructed = row['reconstructed_text']
        
        if instancia_id in original_dict:
            original = original_dict[instancia_id]
            is_match = reconstructed.strip() == original.strip()
            
            if is_match:
                matches += 1
            else:
                # Guardar ejemplos que fallaron para análisis
                if len(failed_examples) < 3:
                    failed_examples.append({
                        'original': original,
                        'reconstructed': reconstructed,
                        'id': instancia_id
                    })
            
            total += 1
            
            # Mostrar progreso cada 1000 oraciones
            if total % 1000 == 0:
                current_acc = (matches / total * 100)
                print(f"   Procesadas {total:,} oraciones - Accuracy actual: {current_acc:.1f}%")
            
            # Mostrar algunos ejemplos exitosos al inicio
            if total <= 3 and is_match:
                print(f"\n📄 Ejemplo exitoso {total}:")
                print(f"   Original:     {original}")
                print(f"   Reconstruida: {reconstructed}")
                print(f"   ✅ Match: {is_match}")
    
    accuracy = (matches / total * 100) if total > 0 else 0
    
    print(f"\n📊 RESULTADOS FINALES DEL SMOKE TEST:")
    print(f"   Oraciones comparadas: {total:,}")
    print(f"   Matches exactos: {matches:,}")
    print(f"   Fallos: {total - matches:,}")
    print(f"   Accuracy: {accuracy:.2f}%")
    
    # Mostrar ejemplos de fallos para análisis
    if failed_examples:
        print(f"\n❌ EJEMPLOS DE FALLOS PARA ANÁLISIS:")
        for i, example in enumerate(failed_examples, 1):
            print(f"\n   Fallo {i} (ID: {example['id']}):")
            print(f"   Original:     '{example['original']}'")
            print(f"   Reconstruida: '{example['reconstructed']}'")
    
   
    return accuracy


print(f"📊 Procesando {len(train_tokenized):,} tokens del dataset de entrenamiento...")

# Usar TODO el dataset tokenizado de train
reconstructed_df = detokenize_and_reconstruct(train_tokenized)

# Comparar con originales
accuracy = compare_with_original(reconstructed_df, train_df_asr) 