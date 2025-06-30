"""
EXTRACCIÓN DE EMBEDDINGS BERT PARA TP2 - AA2
============================================

Este script extrae los embeddings de BERT multilingual para los tokens
de los datasets tokenizados (train, val, test) y los guarda en formato
numpy para usar en las redes neuronales.

Basado en el código de ejemplo de la consigna del TP.
"""

import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
import os
from tqdm import tqdm
import pickle

def setup_bert_model():
    """
    Configura el modelo BERT multilingual y tokenizer
    """
    print("🤖 CONFIGURANDO MODELO BERT MULTILINGUAL")
    print("=" * 50)
    
    model_name = "bert-base-multilingual-cased"
    
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.eval()  # Modo evaluación
    
    print(f"✅ Modelo cargado: {model_name}")
    print(f"✅ Vocabulario size: {tokenizer.vocab_size}")
    print(f"✅ Embedding dimension: {model.config.hidden_size}")
    
    return tokenizer, model

def get_token_embedding(token, tokenizer, model):
    """
    Extrae el embedding estático para un token específico
    Basado en el código de ejemplo de la consigna
    """
    # Validar que el token sea string
    if not isinstance(token, str) or pd.isna(token):
        token = "[UNK]"  # Usar token desconocido si no es válido
    
    # Convertir token a ID
    token_id = tokenizer.convert_tokens_to_ids(token)
    
    # Verificar si el token existe en el vocabulario
    if token_id is None or token_id == tokenizer.unk_token_id:
        token_id = tokenizer.unk_token_id
    
    # Extraer embedding del modelo
    with torch.no_grad():
        embedding_vector = model.embeddings.word_embeddings.weight[token_id]
    
    return embedding_vector.detach().numpy()

def process_tokenized_dataset(file_path, tokenizer, model, dataset_name):
    """
    Procesa un dataset tokenizado y extrae embeddings para todos los tokens
    """
    print(f"\n📊 PROCESANDO DATASET: {dataset_name.upper()}")
    print("=" * 50)
    
    # Cargar dataset tokenizado
    print("📂 Cargando dataset...")
    df = pd.read_csv(file_path)
    print(f"✅ Dataset cargado: {len(df):,} tokens")
    
    # Extraer tokens únicos para optimizar
    print("🔍 Analizando tokens únicos...")
    
    # Limpiar tokens inválidos
    df['token'] = df['token'].fillna('[UNK]')  # Reemplazar NaN con [UNK]
    df['token'] = df['token'].astype(str)      # Asegurar que sean strings
    
    unique_tokens = df['token'].unique()
    print(f"📝 Tokens únicos encontrados: {len(unique_tokens):,}")
    
    # Verificar si hay tokens problemáticos
    invalid_tokens = [t for t in unique_tokens if not isinstance(t, str) or pd.isna(t)]
    if invalid_tokens:
        print(f"⚠️  Tokens inválidos encontrados: {len(invalid_tokens)} (serán reemplazados por [UNK])")
    
    # Crear diccionario de embeddings para tokens únicos
    print(f"\n🧠 Extrayendo embeddings de {len(unique_tokens):,} tokens únicos...")
    token_embeddings = {}
    
    # Barra de progreso detallada para embeddings
    pbar_embeddings = tqdm(
        unique_tokens, 
        desc=f"🔄 Embeddings {dataset_name}",
        unit="tokens",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )
    
    for token in pbar_embeddings:
        embedding = get_token_embedding(token, tokenizer, model)
        token_embeddings[token] = embedding
        # Actualizar descripción con progreso
        pbar_embeddings.set_postfix({
            'completado': f"{len(token_embeddings)}/{len(unique_tokens)}",
            'dim': '768'
        })
    
    print(f"✅ Embeddings extraídos para {len(token_embeddings):,} tokens únicos")
    
    # Crear matriz de embeddings para todo el dataset
    print(f"\n🔧 Construyendo matriz de features para {len(df):,} tokens...")
    embeddings_matrix = []
    labels_punt_inicial = []
    labels_punt_final = []
    labels_capitalizacion = []
    
    # Barra de progreso detallada para construcción de matriz
    pbar_matrix = tqdm(
        df.iterrows(), 
        total=len(df),
        desc=f"🏗️  Matriz {dataset_name}",
        unit="tokens",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )
    
    for idx, row in pbar_matrix:
        token = row['token']
        embedding = token_embeddings[token]
        embeddings_matrix.append(embedding)
        
        # Procesar labels
        labels_punt_inicial.append(1 if row['punt_inicial'] == '¿' else 0)
        
        # Labels puntuación final
        punt_final = row['punt_final']
        if punt_final == ',':
            labels_punt_final.append(0)
        elif punt_final == '.':
            labels_punt_final.append(1)
        elif punt_final == '?':
            labels_punt_final.append(2)
        else:  # sin puntuación
            labels_punt_final.append(3)
        
        # Labels capitalización (ya están como números)
        labels_capitalizacion.append(int(row['capitalizacion']))
        
        # Actualizar cada 1000 tokens para no saturar
        if idx % 1000 == 0:
            pbar_matrix.set_postfix({
                'procesados': f"{idx+1}/{len(df)}",
                'shape': f"({len(embeddings_matrix)}, 768)"
            })
    
    print(f"✅ Matriz de features construida: {len(embeddings_matrix):,} × 768")
    
    # Convertir a arrays numpy
    print("🔄 Convirtiendo a arrays numpy...")
    X = np.array(embeddings_matrix)
    y_punt_inicial = np.array(labels_punt_inicial)
    y_punt_final = np.array(labels_punt_final)
    y_capitalizacion = np.array(labels_capitalizacion)
    
    print(f"✅ Features extraídas:")
    print(f"   X shape: {X.shape}")
    print(f"   y_punt_inicial shape: {y_punt_inicial.shape}")
    print(f"   y_punt_final shape: {y_punt_final.shape}")
    print(f"   y_capitalizacion shape: {y_capitalizacion.shape}")
    
    # Estadísticas de labels
    print(f"\n📈 Distribución de labels:")
    print(f"   Punt. inicial - ¿: {(y_punt_inicial == 1).sum():,} ({(y_punt_inicial == 1).mean()*100:.1f}%)")
    print(f"   Punt. inicial - sin: {(y_punt_inicial == 0).sum():,} ({(y_punt_inicial == 0).mean()*100:.1f}%)")
    
    print(f"   Punt. final - coma: {(y_punt_final == 0).sum():,} ({(y_punt_final == 0).mean()*100:.1f}%)")
    print(f"   Punt. final - punto: {(y_punt_final == 1).sum():,} ({(y_punt_final == 1).mean()*100:.1f}%)")
    print(f"   Punt. final - ?: {(y_punt_final == 2).sum():,} ({(y_punt_final == 2).mean()*100:.1f}%)")
    print(f"   Punt. final - sin: {(y_punt_final == 3).sum():,} ({(y_punt_final == 3).mean()*100:.1f}%)")
    
    for i in range(4):
        count = (y_capitalizacion == i).sum()
        pct = (y_capitalizacion == i).mean() * 100
        print(f"   Capitalización {i}: {count:,} ({pct:.1f}%)")
    
    return X, (y_punt_inicial, y_punt_final, y_capitalizacion), df

def save_features(X, y_tuple, df, dataset_name, output_dir="bert_features"):
    """
    Guarda las features y labels extraídas
    """
    print(f"\n💾 GUARDANDO FEATURES: {dataset_name.upper()}")
    print("=" * 50)
    
    # Crear directorio si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar features (embeddings)
    X_file = os.path.join(output_dir, f"X_{dataset_name}.npy")
    np.save(X_file, X)
    print(f"✅ Features guardadas: {X_file}")
    
    # Guardar labels
    y_punt_inicial, y_punt_final, y_capitalizacion = y_tuple
    
    y_punt_inicial_file = os.path.join(output_dir, f"y_punt_inicial_{dataset_name}.npy")
    y_punt_final_file = os.path.join(output_dir, f"y_punt_final_{dataset_name}.npy")
    y_capitalizacion_file = os.path.join(output_dir, f"y_capitalizacion_{dataset_name}.npy")
    
    np.save(y_punt_inicial_file, y_punt_inicial)
    np.save(y_punt_final_file, y_punt_final)
    np.save(y_capitalizacion_file, y_capitalizacion)
    
    print(f"✅ Labels guardadas:")
    print(f"   {y_punt_inicial_file}")
    print(f"   {y_punt_final_file}")
    print(f"   {y_capitalizacion_file}")
    
    # Guardar información adicional
    info = {
        'dataset_name': dataset_name,
        'num_samples': len(X),
        'embedding_dim': X.shape[1],
        'num_instances': df['instancia_id'].nunique(),
        'label_distributions': {
            'punt_inicial': {
                'sin_puntuacion': int((y_punt_inicial == 0).sum()),
                'pregunta_apertura': int((y_punt_inicial == 1).sum())
            },
            'punt_final': {
                'coma': int((y_punt_final == 0).sum()),
                'punto': int((y_punt_final == 1).sum()),
                'pregunta': int((y_punt_final == 2).sum()),
                'sin_puntuacion': int((y_punt_final == 3).sum())
            },
            'capitalizacion': {
                'minusculas': int((y_capitalizacion == 0).sum()),
                'title_case': int((y_capitalizacion == 1).sum()),
                'mixed_case': int((y_capitalizacion == 2).sum()),
                'uppercase': int((y_capitalizacion == 3).sum())
            }
        }
    }
    
    info_file = os.path.join(output_dir, f"info_{dataset_name}.pkl")
    with open(info_file, 'wb') as f:
        pickle.dump(info, f)
    print(f"✅ Info guardada: {info_file}")

def main():
    """
    Función principal para extraer embeddings de todos los datasets
    """
    print("🚀 EXTRACTOR DE EMBEDDINGS BERT MULTILINGUAL")
    print("=" * 60)
    
    # Configurar modelo BERT
    print("⚙️  Inicializando modelo BERT...")
    tokenizer, model = setup_bert_model()
    
    # Datasets a procesar
    datasets = [
        ("tokenized_train.csv", "train"),
        ("tokenized_val.csv", "val"),
        ("tokenized_test.csv", "test")
    ]
    
    print(f"\n📋 PLAN DE PROCESAMIENTO:")
    print(f"   Datasets a procesar: {len(datasets)}")
    for i, (file_path, name) in enumerate(datasets, 1):
        print(f"   {i}. {name.upper()} ({file_path})")
    
    # Procesar cada dataset con barra de progreso general
    print(f"\n🔄 INICIANDO PROCESAMIENTO DE DATASETS...")
    
    pbar_general = tqdm(
        datasets,
        desc="📊 Datasets",
        unit="dataset",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    )
    
    for file_path, dataset_name in pbar_general:
        pbar_general.set_description(f"📊 Procesando {dataset_name.upper()}")
        pbar_general.set_postfix({'actual': dataset_name})
        
        print(f"\n{'='*60}")
        X, y_tuple, df = process_tokenized_dataset(file_path, tokenizer, model, dataset_name)
        save_features(X, y_tuple, df, dataset_name)
        print(f"✅ {dataset_name.upper()} procesado exitosamente")
    
    print(f"\n🎉 EXTRACCIÓN DE EMBEDDINGS COMPLETADA")
    print("=" * 60)
    print("📁 Los archivos se guardaron en el directorio 'bert_features/'")
    print("🔄 Siguiente paso: Entrenar las redes neuronales con neural_networks.py")

if __name__ == "__main__":
    main()
