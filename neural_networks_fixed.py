# neural_networks.py
import os, math, random, pickle
import numpy as np
import pandas as pd
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------- 1.  Dataset ----------
class TokenSeqDataset(Dataset):
    def __init__(self,
                 split:str,
                 data_dir:str="bert_features",
                 csv_dir:str=".",
                 subset:float=1.0,      # 1.0 = 100 % del split
                 max_len:int=None):     # opcional: truncar secuencias largas
        # 1.1  Cargar arrays
        self.X = np.load(os.path.join(data_dir, f"X_{split}.npy"))
        self.y_ini  = np.load(os.path.join(data_dir, f"y_punt_inicial_{split}.npy"))
        self.y_fin  = np.load(os.path.join(data_dir, f"y_punt_final_{split}.npy"))
        self.y_cap  = np.load(os.path.join(data_dir, f"y_capitalizacion_{split}.npy"))
        # 1.2  Re-leer CSV para saber límites de cada instancia
        df = pd.read_csv(os.path.join(csv_dir, f"tokenized_{split}.csv"),
                         usecols=["instancia_id"])   # ahorramos RAM
        inst = df["instancia_id"].values
        # build list of (start, end) pointers
        self.seq_ptrs : List[Tuple[int,int]] = []
        prev, start = inst[0], 0
        for i, cur in enumerate(inst):
            if cur != prev:
                self.seq_ptrs.append((start, i))  # [start, end)
                start, prev = i, cur
        self.seq_ptrs.append((start, len(inst)))
        # 1.3  Sub-sample si hace falta
        if subset < 1.0:
            k = math.ceil(len(self.seq_ptrs) * subset)
            random.shuffle(self.seq_ptrs)
            self.seq_ptrs = self.seq_ptrs[:k]
        # 1.4  Opcional: truncar
        self.max_len = max_len

    def __len__(self):
        return len(self.seq_ptrs)

    def __getitem__(self, idx):
        s, e = self.seq_ptrs[idx]
        X   = torch.as_tensor(self.X[s:e], dtype=torch.float32)
        y_i = torch.as_tensor(self.y_ini[s:e], dtype=torch.long)
        y_f = torch.as_tensor(self.y_fin[s:e], dtype=torch.long)
        y_c = torch.as_tensor(self.y_cap[s:e], dtype=torch.long)
        # trunc / pad handled by collate_fn
        return X, y_i, y_f, y_c, (e - s)

def pad_collate(batch, pad_value=0.0):
    # batch = list of tuples (X, y_i, y_f, y_c, L)
    lengths = [b[4] for b in batch]
    maxL    = max(lengths)
    # tensors to fill
    feats   = []
    y_i_li, y_f_li, y_c_li = [], [], []
    for X, y_i, y_f, y_c, L in batch:
        pad = maxL - L
        feats.append( torch.cat([X,  X.new_full((pad, X.size(1)), pad_value)]) )
        y_i_li.append( torch.cat([y_i, y_i.new_full((pad,), -100)]) )  # -100 → ignorar en loss
        y_f_li.append( torch.cat([y_f, y_f.new_full((pad,), -100)]) )
        y_c_li.append( torch.cat([y_c, y_c.new_full((pad,), -100)]) )
    return (torch.stack(feats),
            torch.stack(y_i_li),
            torch.stack(y_f_li),
            torch.stack(y_c_li),
            torch.as_tensor(lengths, dtype=torch.long))

# ---------- 2.  Modelo ----------
class BiLSTMTagger(nn.Module):
    def __init__(self,
                 embed_dim:int=768,
                 hidden_size:int=256,
                 num_layers:int=1,
                 dropout:float=0.2):
        super().__init__()
        self.bilstm = nn.LSTM(embed_dim,
                              hidden_size,
                              num_layers=num_layers,
                              batch_first=True,
                              dropout=dropout if num_layers>1 else 0,
                              bidirectional=True)
        out_dim = hidden_size*2
        self.head_ini = nn.Linear(out_dim, 2)   # ¿  / sin
        self.head_fin = nn.Linear(out_dim, 4)   # , . ? sin
        self.head_cap = nn.Linear(out_dim, 4)   # 0-3

    def forward(self, x, lengths):
        # pack -> LSTM -> unpack
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths,
                                                   batch_first=True,
                                                   enforce_sorted=False)
        packed_out, _ = self.bilstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out,
                                                  batch_first=True)
        return self.head_ini(out), self.head_fin(out), self.head_cap(out)

# ---------- 3.  Funciones auxiliares ----------
def calculate_accuracy(predictions, targets, ignore_index=-100):
    """Calcula accuracy ignorando tokens de padding"""
    mask = (targets != ignore_index)
    if mask.sum() == 0:
        return 0.0
    correct = (predictions.argmax(-1) == targets) & mask
    return correct.sum().float() / mask.sum().float()

def calculate_f1(predictions, targets, num_classes, ignore_index=-100):
    """Calcula F1 macro ignorando tokens de padding"""
    mask = (targets != ignore_index)
    if mask.sum() == 0:
        return 0.0
    preds = predictions.argmax(-1)[mask]
    targs = targets[mask]
    f1_total = 0.0
    for c in range(num_classes):
        tp = ((preds == c) & (targs == c)).sum().float()
        fp = ((preds == c) & (targs != c)).sum().float()
        fn = ((preds != c) & (targs == c)).sum().float()
        if tp + fp == 0 or tp + fn == 0:
            f1 = 0.0
        else:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = (2 * precision * recall) / (precision + recall)
        f1_total += f1
    return (f1_total / num_classes).item()


def get_class_weights(y_data, num_classes):
    """Calcula pesos para balancear clases"""
    # Contar frecuencias (ignorando -100)
    valid_mask = y_data != -100
    if valid_mask.sum() == 0:
        return torch.ones(num_classes)
    
    y_valid = y_data[valid_mask]
    counts = torch.bincount(y_valid, minlength=num_classes)
    
    # Evitar división por cero
    counts = torch.clamp(counts, min=1)
    weights = len(y_valid) / (num_classes * counts.float())
    return weights

# ---------- 4.  Entrenamiento ----------
def train_model(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Usando dispositivo: {device}")
    
    ds_train = TokenSeqDataset("train", subset=params["subset"])
    ds_val   = TokenSeqDataset("val")
    dl_train = DataLoader(ds_train, batch_size=params["bs"],
                          shuffle=True, collate_fn=pad_collate)
    dl_val   = DataLoader(ds_val,   batch_size=params["bs"],
                          shuffle=False, collate_fn=pad_collate)

    model = BiLSTMTagger(hidden_size=params["hidden"], num_layers=params["layers"],
                         dropout=params["drop"]).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=params["lr"])
    
    # Calcular pesos de clase si está habilitado
    if params.get("use_class_weights", False):
        print("📊 Calculando pesos de clase...")
        # Cargar un batch para calcular pesos
        sample_batch = next(iter(dl_train))
        _, y_i_sample, y_f_sample, y_c_sample, _ = sample_batch
        
        w_ini = get_class_weights(y_i_sample.flatten(), 2).to(device)
        w_fin = get_class_weights(y_f_sample.flatten(), 4).to(device)
        w_cap = get_class_weights(y_c_sample.flatten(), 4).to(device)
        
        print(f"Pesos punt_inicial: {w_ini}")
        print(f"Pesos punt_final: {w_fin}")
        print(f"Pesos capitalización: {w_cap}")
        
        crit_ini = nn.CrossEntropyLoss(ignore_index=-100, weight=w_ini)
        crit_fin = nn.CrossEntropyLoss(ignore_index=-100, weight=w_fin)
        crit_cap = nn.CrossEntropyLoss(ignore_index=-100, weight=w_cap)
    else:
        crit_ini = crit_fin = crit_cap = nn.CrossEntropyLoss(ignore_index=-100)

    # Pesos por tarea (capitalización más importante por estar desbalanceada)
    task_weights = params.get("task_weights", [1.0, 1.0, 2.0])  # [ini, fin, cap]

    best_f1 = -1.0
    patience = 0
    history = {"train_loss": [], "val_loss": [], "val_acc_ini": [], "val_acc_fin": [], "val_acc_cap": [], "val_f1_ini": [], "val_f1_fin": [], "val_f1_cap": []}
    
    for epoch in range(1, params["epochs"]+1):
        # ===== ENTRENAMIENTO =====
        model.train()
        tr_loss = 0.0
        for X, y_i, y_f, y_c, L in dl_train:
            X, y_i, y_f, y_c, L = X.to(device), y_i.to(device), y_f.to(device), y_c.to(device), L.to(device)
            opt.zero_grad()
            p_i, p_f, p_c = model(X, L)
            
            loss_ini = crit_ini(p_i.transpose(1,2), y_i)
            loss_fin = crit_fin(p_f.transpose(1,2), y_f)
            loss_cap = crit_cap(p_c.transpose(1,2), y_c)
            
            # Pérdida ponderada por tarea
            loss = (task_weights[0] * loss_ini + 
                   task_weights[1] * loss_fin + 
                   task_weights[2] * loss_cap) / sum(task_weights)
            
            loss.backward()
            opt.step()
            tr_loss += loss.item()
        
        # ===== VALIDACIÓN =====
        model.eval()
        val_loss = 0.0
        val_acc_ini = val_acc_fin = val_acc_cap = 0.0
        val_f1_ini = val_f1_fin = val_f1_cap = 0.0
        
        with torch.no_grad():
            for X, y_i, y_f, y_c, L in dl_val:
                X, y_i, y_f, y_c, L = X.to(device), y_i.to(device), y_f.to(device), y_c.to(device), L.to(device)
                p_i, p_f, p_c = model(X, L)
                
                loss_ini = crit_ini(p_i.transpose(1,2), y_i)
                loss_fin = crit_fin(p_f.transpose(1,2), y_f)
                loss_cap = crit_cap(p_c.transpose(1,2), y_c)
                
                batch_loss = (task_weights[0] * loss_ini + 
                             task_weights[1] * loss_fin + 
                             task_weights[2] * loss_cap) / sum(task_weights)
                val_loss += batch_loss.item()
                
                # Calcular accuracies
                val_f1_ini += calculate_f1(p_i, y_i, 2)
                val_f1_fin += calculate_f1(p_f, y_f, 4)
                val_f1_cap += calculate_f1(p_c, y_c, 4)
                val_acc_ini += calculate_accuracy(p_i, y_i).item()
                val_acc_fin += calculate_accuracy(p_f, y_f).item()
                val_acc_cap += calculate_accuracy(p_c, y_c).item()
        
        val_f1_ini /= len(dl_val)
        val_f1_fin /= len(dl_val)
        val_f1_cap /= len(dl_val)
        # Promediar métricas
        val_loss /= len(dl_val)
        val_acc_ini /= len(dl_val)
        val_acc_fin /= len(dl_val)
        val_acc_cap /= len(dl_val)
        
        # Guardar historial
        history["train_loss"].append(tr_loss/len(dl_train))
        history["val_loss"].append(val_loss)
        history["val_f1_ini"].append(val_f1_ini)
        history["val_f1_fin"].append(val_f1_fin)
        history["val_f1_cap"].append(val_f1_cap)
        history["val_acc_ini"].append(val_acc_ini)
        history["val_acc_fin"].append(val_acc_fin)
        history["val_acc_cap"].append(val_acc_cap)
        
        print(f"Epoch {epoch:2d} | train {tr_loss/len(dl_train):.4f} | val {val_loss:.4f} | "
              f"f1_ini {val_f1_ini:.3f} | f1_fin {val_f1_fin:.3f} | f1_cap {val_f1_cap:.3f}")
        
        # Early stopping
        avg_f1 = (val_f1_ini + val_f1_fin + val_f1_cap) / 3
        if avg_f1 > best_f1 + 1e-4:
            best_f1 = avg_f1
            patience = 0
            torch.save(model.state_dict(), "best_bilstm.pt")
            print("💾 Modelo guardado (mejor F1)")
        else:
            patience += 1
            if patience >= params["early_stop"]:
                print(f"⏹️  Early stopping en época {epoch}")
                break
    # Guardar historial
    with open("training_history_bilstm.pkl", "wb") as f:
        pickle.dump(history, f)
    
    print("✅ Entrenamiento finalizado. Mejor F1:", best_f1)
    return history

# ---------- 5.  Inferencia ----------
def predict(split="test", model_path="best_bilstm.pt",
            data_dir="bert_features", csv_dir=".", 
            hidden_size=256, num_layers=2, dropout=0.3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = TokenSeqDataset(split, data_dir=data_dir, csv_dir=csv_dir)
    dl = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=pad_collate)
    
    # Crear modelo con la misma arquitectura que se entrenó
    model = BiLSTMTagger(hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    preds_ini, preds_fin, preds_cap = [], [], []
    with torch.no_grad():
        for X, _, _, _, L in dl:
            p_i, p_f, p_c = model(X.to(device), L.to(device))
            # Solo tomar predicciones para tokens válidos (no padding)
            batch_preds_ini = p_i.argmax(-1).cpu().numpy()
            batch_preds_fin = p_f.argmax(-1).cpu().numpy()
            batch_preds_cap = p_c.argmax(-1).cpu().numpy()
            
            # Para cada secuencia en el batch, tomar solo los tokens válidos
            for i, length in enumerate(L.cpu().numpy()):
                preds_ini.extend(batch_preds_ini[i, :length])
                preds_fin.extend(batch_preds_fin[i, :length])
                preds_cap.extend(batch_preds_cap[i, :length])
    
    # Convertir a numpy arrays
    preds_ini = np.array(preds_ini)
    preds_fin = np.array(preds_fin)
    preds_cap = np.array(preds_cap)
    # reconstruir CSV con las columnas predichas…
    df = pd.read_csv(os.path.join(csv_dir, f"tokenized_{split}.csv"))
    df["punt_inicial_pred"] = np.where(preds_ini==1, '¿', '')
    mapa_fin = {0:',', 1:'.', 2:'?', 3:''}
    df["punt_final_pred"]  = [mapa_fin[x] for x in preds_fin]
    df["capitalizacion_pred"] = preds_cap
    df.to_csv(f"predicciones_{split}.csv", index=False)
    print("✅  Archivo predicciones guardado ->", f"predicciones_{split}.csv")
    return df

# ---------- 6.  Ejemplo de uso ----------
if __name__ == "__main__":
    # Hiperparámetros mejorados
    hyper = dict(
        bs=64, 
        hidden=256, 
        layers=2, 
        drop=0.3,
        lr=2e-3, 
        epochs=20, 
        early_stop=3, 
        subset=1.0,  # Usar dataset completo para modelo final
        use_class_weights=True,  # Balancear clases
        task_weights=[1.0, 1.0, 3.0]  # Dar más peso a capitalización
    )
    
    print("🎯 Iniciando entrenamiento con configuración mejorada...")
    history = train_model(hyper)
    
    print("🔮 Generando predicciones en test...")
    predict("test")
