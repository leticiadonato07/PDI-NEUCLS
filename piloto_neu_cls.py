"""
PILOTO - NEU-CLS
Modelo: EfficientNet-B0 | Resolucao: 100x100 | Iluminacao: 1.0

Valida a pipeline completa antes do experimento real:
- dataset/split/leakage
- preprocessamento + grayscale -> RGB
- augmentation somente no treino
- treinamento + validacao + early stopping
- metricas + matriz de confusao
- parametros + MACs/FLOPs estimados (THOP)
- latencia de inferencia (batch 1, 10 warm-ups, 100 medidas)
- salvamento automatico de resultados/configuracao

"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from thop import profile

# ============================================================
# CONFIGURACOES DO PILOTO
# ============================================================
DATASET_ROOT = Path(r"data/raw/NEU-CLS")
SPLITS_CSV = Path(r"data/processed/splits_neu_cls.csv")

OUTPUT_ROOT = Path(r"results/pilot_efficientnet_b0_100_original")

MODEL_NAME = "EfficientNet-B0"
INPUT_SIZE = 100
BRIGHTNESS_FACTOR = 1.0
NUM_CLASSES = 6
CLASS_NAMES = ["Cr", "In", "Pa", "PS", "RS", "Sc"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

SEED_SPLIT = 42
SEED_TRAIN = 42

BATCH_SIZE = 32
MAX_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
SCHEDULER_NAME = "CosineAnnealing"
EARLY_STOPPING_PATIENCE = 5
MIN_DELTA = 0.0
USE_PRETRAINED_WEIGHTS = True

# Augmentation geometrico leve, sem brilho/contraste.
USE_AUGMENTATION = True
ROTATION_DEGREES = 10
TRANSLATE_FRACTION = 0.05
SCALE_RANGE = (0.95, 1.05)
HORIZONTAL_FLIP_P = 0.50

# Proposta para pesos ImageNet.
NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)

INFERENCE_BATCH_SIZE = 1
WARMUP_RUNS = 10
INFERENCE_RUNS = 100
NUM_WORKERS = 0
PIN_MEMORY = False
CPU_THREADS = None

IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# ============================================================
# REPRODUTIBILIDADE / AMBIENTE
# ============================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_environment_info(device: torch.device) -> Dict:
    try:
        import sklearn
        sklearn_version = sklearn.__version__
    except Exception:
        sklearn_version = "desconhecido"
    try:
        import thop
        thop_version = thop.__version__
    except Exception:
        thop_version = "desconhecido"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "pytorch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pillow": Image.__version__,
        "scikit_learn": sklearn_version,
        "thop": thop_version,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_num_threads": torch.get_num_threads(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Nenhuma GPU CUDA detectada",
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
    }

# ============================================================
# DATASET / CSV
# ============================================================
def normalize_key(value: str) -> str:
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def find_column(df: pd.DataFrame, candidates: List[str], required: bool = True) -> str | None:
    normalized = {normalize_key(c): c for c in df.columns}
    for candidate in candidates:
        key = normalize_key(candidate)
        if key in normalized:
            return normalized[key]
    if required:
        raise ValueError(f"Coluna nao encontrada. Candidatas={candidates}; encontradas={list(df.columns)}")
    return None


def normalize_split(value: str) -> str:
    key = normalize_key(value)
    mapping = {
        "train": "train", "training": "train", "treino": "train", "treinamento": "train", "tr": "train",
        "val": "val", "valid": "val", "validation": "val", "validacao": "val", "validacaoo": "val",
        "test": "test", "testing": "test", "teste": "test", "te": "test",
    }
    if key not in mapping:
        raise ValueError(f"Split nao reconhecido: {value!r}")
    return mapping[key]


def canonicalize_label(value: str) -> str:
    key = normalize_key(value)
    mapping = {
        "cr": "Cr", "crazing": "Cr",
        "in": "In", "inclusion": "In",
        "pa": "Pa", "patch": "Pa", "patches": "Pa",
        "ps": "PS", "pitted_surface": "PS", "pittedsurface": "PS",
        "rs": "RS", "rolled_in_scale": "RS", "rolledinscale": "RS",
        "sc": "Sc", "scratch": "Sc", "scratches": "Sc",
    }
    if key in mapping:
        return mapping[key]
    prefix = re.split(r"[_\-\s]", str(value).strip())[0].lower()
    if prefix in mapping:
        return mapping[prefix]
    raise ValueError(f"Classe nao reconhecida: {value!r}; esperadas={CLASS_NAMES}")


def build_image_index(dataset_root: Path) -> Dict[str, List[Path]]:
    if not dataset_root.exists():
        raise FileNotFoundError(f"DATASET_ROOT nao existe: {dataset_root.resolve()}")
    index: Dict[str, List[Path]] = {}
    for path in dataset_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            index.setdefault(path.name.lower(), []).append(path)
    if not index:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em {dataset_root.resolve()}")
    return index


def resolve_image_path(raw_value: str, dataset_root: Path, image_index: Dict[str, List[Path]]) -> Path:
    raw = str(raw_value).strip().replace("\\", os.sep)
    candidate = Path(raw)
    if candidate.is_absolute() and candidate.exists():
        return candidate.resolve()
    rel = (dataset_root / candidate).resolve()
    if rel.exists():
        return rel
    matches = image_index.get(candidate.name.lower(), [])
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        raise ValueError(f"Nome de arquivo ambiguo: {candidate.name}; encontrados={matches}")
    raise FileNotFoundError(f"Imagem nao encontrada: {raw_value!r}")


def infer_label_from_path(path: Path) -> str:
    for source in (path.parent.name, path.stem):
        try:
            return canonicalize_label(source)
        except ValueError:
            pass
    raise ValueError(f"Nao foi possivel inferir a classe de {path}")


def load_and_prepare_split() -> pd.DataFrame:
    if not SPLITS_CSV.exists():
        raise FileNotFoundError(f"SPLITS_CSV nao existe: {SPLITS_CSV.resolve()}")
    df = pd.read_csv(SPLITS_CSV)
    image_col = find_column(df, ["path", "filepath", "file_path", "image_path", "arquivo", "filename", "file_name", "nome_arquivo", "image", "imagem", "file"], True)
    split_col = find_column(df, ["split", "subset", "partition", "conjunto", "divisao", "divisão", "set"], True)
    label_col = find_column(df, ["label", "class", "classe", "defect", "defeito", "category", "categoria"], False)
    image_index = build_image_index(DATASET_ROOT)
    rows = []
    for _, row in df.iterrows():
        path = resolve_image_path(row[image_col], DATASET_ROOT, image_index)
        split = normalize_split(row[split_col])
        label = canonicalize_label(row[label_col]) if label_col else infer_label_from_path(path)
        rows.append({"image_path": str(path), "split": split, "label": label, "label_idx": CLASS_TO_IDX[label], "filename": path.name})
    return pd.DataFrame(rows)


def md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    print("\n=== VALIDACAO DO DATASET ===")
    print(f"Total de imagens no CSV: {len(df)}")
    if len(df) != 1799:
        print(f"AVISO: esperado=1799; encontrado={len(df)}")
    expected_splits = {"train", "val", "test"}
    found_splits = set(df["split"].unique())
    if found_splits != expected_splits:
        raise ValueError(f"Splits encontrados={found_splits}; esperado={expected_splits}")
    missing = set(CLASS_NAMES) - set(df["label"].unique())
    if missing:
        raise ValueError(f"Classes ausentes: {missing}")

    path_sets = {s: set(df.loc[df["split"] == s, "image_path"]) for s in expected_splits}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = path_sets[a] & path_sets[b]
        if overlap:
            raise RuntimeError(f"VAZAMENTO: {a} x {b}: {list(overlap)[:10]}")

    print("Verificando duplicatas exatas por MD5...")
    hash_to_items: Dict[str, List[Tuple[str, str]]] = {}
    for _, row in df.iterrows():
        h = md5_file(Path(row["image_path"]))
        hash_to_items.setdefault(h, []).append((row["split"], row["filename"]))
    cross_split = []
    for items in hash_to_items.values():
        if len(items) > 1 and len({x[0] for x in items}) > 1:
            cross_split.append(items)
    if cross_split:
        raise RuntimeError(f"DUPLICATAS EXATAS ENTRE SPLITS: {cross_split[:10]}")
    print("Nenhuma duplicata exata entre splits.")
    print("\nDistribuicao por split/classe:")
    print(pd.crosstab(df["split"], df["label"]))
    print("\nQuantidade por split:")
    print(df["split"].value_counts())
    return df

# ============================================================
# ILUMINACAO / PREPROCESSAMENTO / AUGMENTATION
# ============================================================
class BrightnessMultiply:
    def __init__(self, factor: float):
        self.factor = float(factor)

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("L")
        if self.factor == 1.0:
            return image
        array = np.asarray(image, dtype=np.float32)
        array = np.clip(array * self.factor, 0, 255).astype(np.uint8)
        return Image.fromarray(array, mode="L")


def build_transforms():
    resize = transforms.Resize((INPUT_SIZE, INPUT_SIZE), antialias=True)
    to_rgb = transforms.Lambda(lambda img: img.convert("RGB"))
    normalize = transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)

    train_ops = [BrightnessMultiply(BRIGHTNESS_FACTOR), resize, to_rgb]
    if USE_AUGMENTATION:
        train_ops += [
            transforms.RandomAffine(
                degrees=ROTATION_DEGREES,
                translate=(TRANSLATE_FRACTION, TRANSLATE_FRACTION),
                scale=SCALE_RANGE,
            ),
            transforms.RandomHorizontalFlip(p=HORIZONTAL_FLIP_P),
        ]
    train_ops += [transforms.ToTensor(), normalize]

    eval_ops = [BrightnessMultiply(BRIGHTNESS_FACTOR), resize, to_rgb, transforms.ToTensor(), normalize]
    return transforms.Compose(train_ops), transforms.Compose(eval_ops)


class NEUCLSDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform):
        self.df = dataframe.reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        with Image.open(Path(row["image_path"])) as image:
            image = image.convert("L")
            tensor = self.transform(image)
        return tensor, int(row["label_idx"])

# ============================================================
# VISUALIZACAO
# ============================================================
def _denormalize(tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(NORMALIZE_MEAN).view(3, 1, 1)
    std = torch.tensor(NORMALIZE_STD).view(3, 1, 1)
    img = (tensor.cpu() * std + mean).clamp(0, 1)
    return np.transpose(img.numpy(), (1, 2, 0))


def save_preprocessing_examples(df: pd.DataFrame, transform, output_path: Path) -> None:
    samples = []
    for cls in CLASS_NAMES:
        cdf = df[df["label"] == cls]
        if cdf.empty:
            continue
        path = Path(cdf.iloc[0]["image_path"])
        with Image.open(path) as image:
            tensor = transform(image.convert("L"))
        samples.append((cls, _denormalize(tensor)))

    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    axes = axes.ravel()
    for ax in axes:
        ax.axis("off")
    for ax, (cls, img) in zip(axes, samples):
        ax.imshow(img)
        ax.set_title(cls)
        ax.axis("off")
    fig.suptitle(f"Pre-processamento - {INPUT_SIZE}x{INPUT_SIZE} - iluminacao {BRIGHTNESS_FACTOR}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_augmentation_examples(df: pd.DataFrame, transform, output_path: Path) -> None:
    path = Path(df.iloc[0]["image_path"])
    with Image.open(path) as image:
        image = image.convert("L")
        images = [_denormalize(transform(image)) for _ in range(6)]
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    for ax, img in zip(axes.ravel(), images):
        ax.imshow(img)
        ax.axis("off")
    fig.suptitle("Exemplos de Data Augmentation")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

# ============================================================
# MODELO / EFICIENCIA
# ============================================================
def build_model() -> nn.Module:
    weights = models.EfficientNet_B0_Weights.DEFAULT if USE_PRETRAINED_WEIGHTS else None
    model = models.efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    return model


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def profile_model(model: nn.Module, device: torch.device) -> Dict:
    model.eval()
    params = count_trainable_parameters(model)
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, device=device)
    macs, thop_params = profile(model, inputs=(dummy,), verbose=False)
    flops = 2 * macs
    return {
        "model": MODEL_NAME,
        "input_size": f"{INPUT_SIZE}x{INPUT_SIZE}",
        "parameters": int(params),
        "parameters_millions": params / 1e6,
        "macs": int(macs),
        "macs_millions": macs / 1e6,
        "estimated_flops": int(flops),
        "estimated_flops_millions": flops / 1e6,
        "thop_parameters": int(thop_params),
    }

# ============================================================
# METRICAS
# ============================================================
def calculate_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, labels=list(range(NUM_CLASSES)), average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, labels=list(range(NUM_CLASSES)), average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, labels=list(range(NUM_CLASSES)), average="macro", zero_division=0),
    }


def predict_loader(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    y_true, y_pred = [], []
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    return np.asarray(y_true), np.asarray(y_pred)

# ============================================================
# TREINAMENTO
# ============================================================
def make_optimizer_scheduler(model: nn.Module):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    if SCHEDULER_NAME.lower() == "cosineannealing":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    else:
        raise ValueError(f"Scheduler nao implementado: {SCHEDULER_NAME}")
    return optimizer, scheduler


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    n = 0
    y_true, y_pred = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        bs = labels.size(0)
        total_loss += loss.item() * bs
        n += bs
        y_true.extend(labels.detach().cpu().numpy())
        y_pred.extend(outputs.argmax(1).detach().cpu().numpy())
    metrics = calculate_metrics(np.asarray(y_true), np.asarray(y_pred))
    metrics["loss"] = total_loss / n
    return metrics


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n = 0
    y_true, y_pred = [], []
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            bs = labels.size(0)
            total_loss += loss.item() * bs
            n += bs
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(outputs.argmax(1).cpu().numpy())
    metrics = calculate_metrics(np.asarray(y_true), np.asarray(y_pred))
    metrics["loss"] = total_loss / n
    return metrics


def train_model(model, train_loader, val_loader, device, run_dir: Path):
    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = make_optimizer_scheduler(model)
    best_f1 = -np.inf
    best_epoch = 0
    no_improve = 0
    history = []
    start_all = time.perf_counter()
    process = psutil.Process(os.getpid())
    peak_rss = process.memory_info().rss
    ckpt = run_dir / "checkpoints" / "best_model.pth"
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, MAX_EPOCHS + 1):
        start_epoch = time.perf_counter()
        train_m = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_m = eval_epoch(model, val_loader, criterion, device)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        epoch_time = time.perf_counter() - start_epoch
        rss = process.memory_info().rss
        peak_rss = max(peak_rss, rss)
        row = {
            "epoch": epoch,
            "train_loss": train_m["loss"],
            "train_accuracy": train_m["accuracy"],
            "train_macro_precision": train_m["macro_precision"],
            "train_macro_recall": train_m["macro_recall"],
            "train_macro_f1": train_m["macro_f1"],
            "val_loss": val_m["loss"],
            "val_accuracy": val_m["accuracy"],
            "val_macro_precision": val_m["macro_precision"],
            "val_macro_recall": val_m["macro_recall"],
            "val_macro_f1": val_m["macro_f1"],
            "learning_rate": lr,
            "epoch_time_seconds": epoch_time,
            "process_rss_mb": rss / (1024 ** 2),
        }
        history.append(row)
        print(f"Epoca {epoch:02d}/{MAX_EPOCHS} | train loss={train_m['loss']:.4f} | val loss={val_m['loss']:.4f} | val acc={val_m['accuracy']:.4f} | val macro-F1={val_m['macro_f1']:.4f} | LR={lr:.7f} | {epoch_time:.1f}s")
        current = val_m["macro_f1"]
        if current > best_f1 + MIN_DELTA:
            best_f1 = current
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), ckpt)
            print(f"  -> Melhor checkpoint salvo (Macro-F1={best_f1:.4f})")
        else:
            no_improve += 1
            print(f"  -> Sem melhora: {no_improve}/{EARLY_STOPPING_PATIENCE}")
            if no_improve >= EARLY_STOPPING_PATIENCE:
                print("  -> Early stopping acionado.")
                break

    total_time = time.perf_counter() - start_all
    history_df = pd.DataFrame(history)
    history_df.to_csv(run_dir / "results" / "historico_treinamento.csv", index=False)
    summary = {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "epochs_executed": len(history),
        "training_time_seconds": total_time,
        "training_time_minutes": total_time / 60,
        "peak_process_rss_mb": peak_rss / (1024 ** 2),
        "best_checkpoint": str(ckpt),
    }
    with (run_dir / "results" / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model, history_df, summary

# ============================================================
# GRAFICOS / TESTE
# ============================================================
def save_training_plots(history: pd.DataFrame, output_dir: Path):
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for name, cols, ylabel, title in [
        ("loss", ["train_loss", "val_loss"], "Loss", "Loss de treino e validacao"),
        ("accuracy", ["train_accuracy", "val_accuracy"], "Accuracy", "Accuracy de treino e validacao"),
        ("macro_f1", ["train_macro_f1", "val_macro_f1"], "Macro-F1", "Macro-F1 de treino e validacao"),
    ]:
        plt.figure(figsize=(8, 5))
        plt.plot(history["epoch"], history[cols[0]], label=cols[0])
        plt.plot(history["epoch"], history[cols[1]], label=cols[1])
        plt.xlabel("Epoca")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close()


def evaluate_test_set(model, test_loader, run_dir: Path, device: torch.device):
    y_true, y_pred = predict_loader(model, test_loader, device)
    metrics = calculate_metrics(y_true, y_pred)
    with (run_dir / "results" / "metricas_teste.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    report = classification_report(y_true, y_pred, labels=list(range(NUM_CLASSES)), target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(run_dir / "results" / "relatorio_classificacao.csv")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(run_dir / "results" / "matriz_confusao.csv")
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Matriz de confusao - teste")
    plt.colorbar()
    ticks = np.arange(NUM_CLASSES)
    plt.xticks(ticks, CLASS_NAMES, rotation=45)
    plt.yticks(ticks, CLASS_NAMES)
    threshold = cm.max() / 2.0 if cm.max() else 0.0
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            plt.text(j, i, int(cm[i, j]), ha="center", color="white" if cm[i, j] > threshold else "black")
    plt.ylabel("Classe verdadeira")
    plt.xlabel("Classe predita")
    plt.tight_layout()
    plt.savefig(run_dir / "results" / "matriz_confusao.png", dpi=150, bbox_inches="tight")
    plt.close()
    return metrics

# ============================================================
# INFERENCIA
# ============================================================
def synchronize_if_needed(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def measure_inference_time(model: nn.Module, device: torch.device) -> Dict:
    model.eval()
    dummy = torch.randn(INFERENCE_BATCH_SIZE, 3, INPUT_SIZE, INPUT_SIZE, device=device)
    with torch.inference_mode():
        for _ in range(WARMUP_RUNS):
            _ = model(dummy)
        synchronize_if_needed(device)
        times = []
        for _ in range(INFERENCE_RUNS):
            synchronize_if_needed(device)
            start = time.perf_counter()
            _ = model(dummy)
            synchronize_if_needed(device)
            end = time.perf_counter()
            times.append((end - start) * 1000.0)
    arr = np.asarray(times)
    return {
        "batch_size": INFERENCE_BATCH_SIZE,
        "warmup_runs": WARMUP_RUNS,
        "measured_runs": INFERENCE_RUNS,
        "mean_ms_per_image": float(arr.mean()),
        "std_ms_per_image": float(arr.std(ddof=1)),
        "min_ms_per_image": float(arr.min()),
        "max_ms_per_image": float(arr.max()),
        "fps_from_mean": float(1000.0 / arr.mean()) if arr.mean() > 0 else None,
    }

# ============================================================
# CONFIGURACAO FINAL
# ============================================================
def save_config(run_dir: Path, environment: Dict, dataset_df: pd.DataFrame, profile_info: Dict, training_summary: Dict, test_metrics: Dict, inference_info: Dict):
    config = {
        "experiment_type": "pilot",
        "model": MODEL_NAME,
        "pretrained_imagenet_weights": USE_PRETRAINED_WEIGHTS,
        "num_classes": NUM_CLASSES,
        "classes": CLASS_NAMES,
        "dataset_total_images": len(dataset_df),
        "dataset_split": {s: int((dataset_df["split"] == s).sum()) for s in ["train", "val", "test"]},
        "split_seed": SEED_SPLIT,
        "training_seed": SEED_TRAIN,
        "input_resolution": [INPUT_SIZE, INPUT_SIZE],
        "brightness_factor": BRIGHTNESS_FACTOR,
        "preprocessing": {
            "grayscale_to_rgb": True,
            "normalization_mean": NORMALIZE_MEAN,
            "normalization_std": NORMALIZE_STD,
        },
        "augmentation": {
            "enabled": USE_AUGMENTATION,
            "rotation_degrees": ROTATION_DEGREES,
            "translate_fraction": TRANSLATE_FRACTION,
            "scale_range": SCALE_RANGE,
            "horizontal_flip_probability": HORIZONTAL_FLIP_P,
            "brightness_augmentation": False,
        },
        "training": {
            "framework": "PyTorch",
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "loss": "CrossEntropyLoss",
            "scheduler": SCHEDULER_NAME,
            "early_stopping_metric": "val_macro_f1",
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "min_delta": MIN_DELTA,
        },
        "efficiency": profile_info,
        "training_summary": training_summary,
        "test_metrics": test_metrics,
        "inference": inference_info,
        "environment": environment,
    }
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

# ============================================================
# MAIN
# ============================================================
def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for sub in ["checkpoints", "results", "plots"]:
        (OUTPUT_ROOT / sub).mkdir(parents=True, exist_ok=True)
    if CPU_THREADS is not None:
        torch.set_num_threads(CPU_THREADS)
    set_seed(SEED_TRAIN)
    device = get_device()

    print("=" * 70)
    print("PILOTO NEU-CLS")
    print("=" * 70)
    print(f"Modelo: {MODEL_NAME}")
    print(f"Resolucao: {INPUT_SIZE}x{INPUT_SIZE}")
    print(f"Iluminacao: {BRIGHTNESS_FACTOR}")
    print(f"Device: {device}")
    print(f"Batch: {BATCH_SIZE}")
    print(f"Max. epocas: {MAX_EPOCHS}")
    print(f"Optimizer: AdamW | LR={LEARNING_RATE} | weight_decay={WEIGHT_DECAY}")
    print(f"Scheduler: {SCHEDULER_NAME}")
    print(f"Early stopping: val_macro_f1 | patience={EARLY_STOPPING_PATIENCE}")

    environment = get_environment_info(device)
    with (OUTPUT_ROOT / "environment.json").open("w", encoding="utf-8") as f:
        json.dump(environment, f, indent=4, ensure_ascii=False)

    # Dataset
    df = validate_dataset(load_and_prepare_split())
    df.to_csv(OUTPUT_ROOT / "results" / "dataset_validado.csv", index=False)

    # Pre-processamento / augmentation
    train_tf, eval_tf = build_transforms()
    save_preprocessing_examples(df, eval_tf, OUTPUT_ROOT / "preprocessamento_exemplos.png")
    if USE_AUGMENTATION:
        save_augmentation_examples(df[df["split"] == "train"].reset_index(drop=True), train_tf, OUTPUT_ROOT / "augmentation_exemplos.png")

    # Datasets / loaders
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    train_loader = DataLoader(NEUCLSDataset(train_df, train_tf), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(NEUCLSDataset(val_df, eval_tf), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader = DataLoader(NEUCLSDataset(test_df, eval_tf), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    # Modelo
    model = build_model().to(device)
    model.eval()
    dummy = torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE, device=device)
    with torch.inference_mode():
        output = model(dummy)
    print("\n=== TESTE DO FORWARD ===")
    print(f"Entrada: {tuple(dummy.shape)}")
    print(f"Saida:   {tuple(output.shape)}")
    if tuple(output.shape) != (2, NUM_CLASSES):
        raise RuntimeError(f"Saida inesperada: {tuple(output.shape)}")
    print("Forward OK.")

    # Eficiencia
    profile_info = profile_model(model, device)
    print("\n=== EFICIENCIA ===")
    print(f"Parametros: {profile_info['parameters']:,} ({profile_info['parameters_millions']:.3f} M)")
    print(f"MACs: {profile_info['macs']:,} ({profile_info['macs_millions']:.3f} M)")
    print(f"FLOPs estimados: {profile_info['estimated_flops']:,} ({profile_info['estimated_flops_millions']:.3f} M)")
    pd.DataFrame([profile_info]).to_csv(OUTPUT_ROOT / "results" / "eficiencia.csv", index=False)

    # Treinamento
    model, history, training_summary = train_model(model, train_loader, val_loader, device, OUTPUT_ROOT)
    save_training_plots(history, OUTPUT_ROOT)

    # Teste
    print("\n=== AVALIACAO NO TESTE ===")
    test_metrics = evaluate_test_set(model, test_loader, OUTPUT_ROOT, device)
    for key, value in test_metrics.items():
        print(f"{key}: {value:.4f}")

    # Inferencia
    print("\n=== TEMPO DE INFERENCIA ===")
    inference_info = measure_inference_time(model, device)
    for key, value in inference_info.items():
        print(f"{key}: {value}")
    with (OUTPUT_ROOT / "results" / "inferencia.json").open("w", encoding="utf-8") as f:
        json.dump(inference_info, f, indent=4, ensure_ascii=False)

    # Config final
    save_config(OUTPUT_ROOT, environment, df, profile_info, training_summary, test_metrics, inference_info)

    print("\n" + "=" * 70)
    print("PILOTO FINALIZADO")
    print("=" * 70)
    print(f"Resultados: {OUTPUT_ROOT.resolve()}")


if __name__ == "__main__":
    main()
