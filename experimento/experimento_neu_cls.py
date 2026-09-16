from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import thop

# ============================================================
# 1. CONFIGURAÇÕES DO EXPERIMENTO
# ============================================================

# AJUSTAR ESTES DOIS CAMINHOS PARA O COMPUTADOR DA DUPLA
DATASET_ROOT = Path(r"C:\Projetos\PDI-NEUCLS\data\raw\NEU-CLS\NEU-CLS")
SPLITS_CSV = Path(r"C:\Projetos\PDI-NEUCLS\data\processed\splits_neu_cls.csv")
OUTPUT_ROOT = Path(r"C:\Projetos\PDI-NEUCLS\results\experimento_completo")

MAX_EXPERIMENTS: Optional[int] = None
RESUME = True

MODELS_TO_RUN = [
    "MobileNetV3-Small",
    "EfficientNet-B0",
    "ShuffleNetV2 1.0x",
]
RESOLUTIONS = [50, 100, 200]
BRIGHTNESS_FACTORS = [0.7, 1.0, 1.15]
TRAINING_SEEDS = [42, 123, 2026]

NUM_CLASSES = 6
CLASS_NAMES = ["Cr", "In", "Pa", "PS", "RS", "Sc"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

SPLIT_SEED = 42
BATCH_SIZE = 32
MAX_EPOCHS = 50

OPTIMIZER_NAME = "AdamW"
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
LOSS_NAME = "CrossEntropyLoss"
SCHEDULER_NAME = "CosineAnnealing"
EARLY_STOPPING_METRIC = "val_macro_f1"
EARLY_STOPPING_PATIENCE = 5
MIN_DELTA = 0.0

USE_PRETRAINED_WEIGHTS = True

AUGMENTATION_ENABLED = True
HORIZONTAL_FLIP_P = 0.5

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
# 2. UTILITÁRIOS
# ============================================================

def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))
    return value.strip("._") or "item"


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


def get_sklearn_version() -> str:
    try:
        import sklearn
        return sklearn.__version__
    except Exception:
        return "desconhecido"


def get_environment_info(device: torch.device) -> Dict:
    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "pytorch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pillow": Image.__version__,
        "scikit_learn": get_sklearn_version(),
        "thop": getattr(thop, "__version__", "desconhecido"),
        "thop_path": str(Path(thop.__file__).resolve()),
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_num_threads": torch.get_num_threads(),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
    else:
        info["gpu_name"] = "Nenhuma GPU CUDA detectada"
        info["cuda_version"] = None
    return info


def atomic_json_dump(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False, default=str)
    tmp.replace(path)


def atomic_csv_save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


# ============================================================
# 3. SPLIT / DATASET
# ============================================================

def normalize_column_name(value: str) -> str:
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def find_column(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    normalized = {normalize_column_name(c): c for c in df.columns}
    for candidate in candidates:
        key = normalize_column_name(candidate)
        if key in normalized:
            return normalized[key]
    if required:
        raise ValueError(
            f"Coluna não encontrada. Candidatas: {candidates}. "
            f"Colunas do CSV: {list(df.columns)}"
        )
    return None


def normalize_split(value: str) -> str:
    key = normalize_column_name(value)
    mapping = {
        "train": "train", "training": "train", "treino": "train", "treinamento": "train",
        "val": "val", "valid": "val", "validation": "val", "validacao": "val",
        "test": "test", "testing": "test", "teste": "test",
    }
    if key not in mapping:
        raise ValueError(f"Split não reconhecido: {value!r}")
    return mapping[key]


def canonicalize_label(value: str) -> str:
    key = normalize_column_name(value)
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
    raise ValueError(f"Classe não reconhecida: {value!r}")


def build_image_index(root: Path) -> Dict[str, List[Path]]:
    if not root.exists():
        raise FileNotFoundError(f"DATASET_ROOT não existe: {root.resolve()}")
    index: Dict[str, List[Path]] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            index.setdefault(path.name.lower(), []).append(path)
    if not index:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em {root.resolve()}")
    return index


def resolve_image_path(raw_value: str, root: Path, index: Dict[str, List[Path]]) -> Path:
    raw = str(raw_value).strip().replace("\\", os.sep)
    candidate = Path(raw)
    if candidate.is_absolute() and candidate.exists():
        return candidate.resolve()
    relative = (root / candidate).resolve()
    if relative.exists():
        return relative
    matches = index.get(candidate.name.lower(), [])
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        raise ValueError(f"Arquivo ambíguo: {candidate.name} -> {matches}")
    raise FileNotFoundError(f"Imagem não encontrada: {raw_value!r}")


def infer_label_from_path(path: Path) -> str:
    for raw in (path.parent.name, path.stem):
        try:
            return canonicalize_label(raw)
        except ValueError:
            pass
    raise ValueError(f"Não foi possível inferir a classe de {path}")


def load_split_dataframe() -> pd.DataFrame:
    if not SPLITS_CSV.exists():
        raise FileNotFoundError(f"SPLITS_CSV não existe: {SPLITS_CSV.resolve()}")

    df = pd.read_csv(SPLITS_CSV)
    image_col = find_column(
        df,
        ["path", "filepath", "file_path", "image_path", "arquivo", "filename", "file_name", "nome_arquivo", "image", "imagem", "file"],
    )
    split_col = find_column(
        df,
        ["split", "subset", "partition", "conjunto", "divisao", "divisão", "set"],
    )
    label_col = find_column(
        df,
        ["label", "class", "classe", "defect", "defeito", "category", "categoria"],
        required=False,
    )

    index = build_image_index(DATASET_ROOT)
    rows = []
    for _, row in df.iterrows():
        path = resolve_image_path(row[image_col], DATASET_ROOT, index)
        split = normalize_split(row[split_col])
        label = canonicalize_label(row[label_col]) if label_col else infer_label_from_path(path)
        rows.append({
            "image_path": str(path),
            "filename": path.name,
            "split": split,
            "label": label,
            "label_idx": CLASS_TO_IDX[label],
        })
    return pd.DataFrame(rows)


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def validate_split(df: pd.DataFrame) -> pd.DataFrame:
    print("\n=== VALIDANDO DATASET E SPLIT ===")

    if len(df) != 1799:
        raise ValueError(f"Esperadas 1799 imagens; encontradas {len(df)}.")
    if set(df["split"].unique()) != {"train", "val", "test"}:
        raise ValueError(f"Splits encontrados: {set(df['split'].unique())}")
    if set(CLASS_NAMES) - set(df["label"].unique()):
        raise ValueError(f"Classes ausentes: {set(CLASS_NAMES) - set(df['label'].unique())}")

    split_paths = {s: set(df.loc[df["split"] == s, "image_path"]) for s in ("train", "val", "test")}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_paths[a] & split_paths[b]
        if overlap:
            raise RuntimeError(f"Vazamento detectado entre {a} e {b}: {list(overlap)[:10]}")

    print("Verificando duplicatas exatas por MD5...")
    hash_map: Dict[str, List[Tuple[str, str]]] = {}
    for _, row in df.iterrows():
        digest = md5_file(Path(row["image_path"]))
        hash_map.setdefault(digest, []).append((row["split"], row["filename"]))
    cross = []
    for items in hash_map.values():
        if len(items) > 1 and len({x[0] for x in items}) > 1:
            cross.append(items)
    if cross:
        raise RuntimeError(f"Duplicatas exatas entre splits: {cross[:5]}")

    print("Nenhuma duplicata exata entre splits.")
    print("\nDistribuição por split/classe:")
    print(pd.crosstab(df["split"], df["label"]))
    print("\nQuantidade por split:")
    print(df["split"].value_counts())
    return df


# ============================================================
# 4. PREPROCESSAMENTO / DATASET
# ============================================================

class BrightnessMultiply:
    def __init__(self, factor: float):
        self.factor = float(factor)

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("L")
        if self.factor == 1.0:
            return image
        arr = np.asarray(image, dtype=np.float32)
        arr = np.clip(arr * self.factor, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="L")


def build_transforms(resolution: int, brightness_factor: float):
    train_ops = [
        BrightnessMultiply(brightness_factor),
        transforms.Resize((resolution, resolution), antialias=True),
        transforms.Lambda(lambda img: img.convert("RGB")),
    ]
    eval_ops = [
        BrightnessMultiply(brightness_factor),
        transforms.Resize((resolution, resolution), antialias=True),
        transforms.Lambda(lambda img: img.convert("RGB")),
    ]

    if AUGMENTATION_ENABLED:
        train_ops.append(transforms.RandomHorizontalFlip(p=HORIZONTAL_FLIP_P))

    train_ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    ])
    eval_ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    ])

    return transforms.Compose(train_ops), transforms.Compose(eval_ops)


class NEUCLSDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform):
        self.df = dataframe.reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        with Image.open(row["image_path"]) as image:
            tensor = self.transform(image.convert("L"))
        return tensor, int(row["label_idx"])


# ============================================================
# 5. MODELOS
# ============================================================

def build_model(model_name: str) -> nn.Module:
    if model_name == "MobileNetV3-Small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if USE_PRETRAINED_WEIGHTS else None
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, NUM_CLASSES)
        return model

    if model_name == "EfficientNet-B0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if USE_PRETRAINED_WEIGHTS else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
        return model

    if model_name == "ShuffleNetV2 1.0x":
        weights = models.ShuffleNet_V2_X1_0_Weights.DEFAULT if USE_PRETRAINED_WEIGHTS else None
        model = models.shufflenet_v2_x1_0(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        return model

    raise ValueError(f"Modelo não suportado: {model_name}")


# ============================================================
# 6. MÉTRICAS
# ============================================================

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    labels = list(range(NUM_CLASSES))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
    }


def predict_loader(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())
    return np.asarray(y_true), np.asarray(y_pred)


# ============================================================
# 7. TREINAMENTO
# ============================================================

def create_optimizer_scheduler(model):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=MAX_EPOCHS,
    )
    return optimizer, scheduler


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total = 0
    all_true, all_pred = [], []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        n = labels.size(0)
        total_loss += loss.item() * n
        total += n
        all_true.extend(labels.detach().cpu().numpy())
        all_pred.extend(logits.argmax(dim=1).detach().cpu().numpy())

    metrics = calculate_metrics(np.asarray(all_true), np.asarray(all_pred))
    metrics["loss"] = total_loss / total
    return metrics


def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0
    all_true, all_pred = [], []

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            n = labels.size(0)
            total_loss += loss.item() * n
            total += n
            all_true.extend(labels.cpu().numpy())
            all_pred.extend(logits.argmax(dim=1).cpu().numpy())

    metrics = calculate_metrics(np.asarray(all_true), np.asarray(all_pred))
    metrics["loss"] = total_loss / total
    return metrics


def train_model(model, train_loader, val_loader, device, run_dir):
    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = create_optimizer_scheduler(model)

    result_dir = run_dir / "results"
    checkpoint_dir = run_dir / "checkpoints"
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_metric = -np.inf
    best_epoch = 0
    epochs_without_improvement = 0
    best_checkpoint = checkpoint_dir / "best_model.pth"

    process = psutil.Process(os.getpid())
    peak_rss = process.memory_info().rss
    training_start = time.perf_counter()

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start = time.perf_counter()

        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        epoch_seconds = time.perf_counter() - epoch_start
        peak_rss = max(peak_rss, process.memory_info().rss)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_precision": train_metrics["macro_precision"],
            "train_macro_recall": train_metrics["macro_recall"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_precision": val_metrics["macro_precision"],
            "val_macro_recall": val_metrics["macro_recall"],
            "val_macro_f1": val_metrics["macro_f1"],
            "learning_rate": current_lr,
            "epoch_time_seconds": epoch_seconds,
            "process_rss_mb": process.memory_info().rss / (1024 ** 2),
        }
        history.append(row)

        history_df = pd.DataFrame(history)
        atomic_csv_save(history_df, result_dir / "historico_treinamento.csv")

        print(
            f"Época {epoch:02d}/{MAX_EPOCHS} | "
            f"train loss={train_metrics['loss']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} | "
            f"val acc={val_metrics['accuracy']:.4f} | "
            f"val Macro-F1={val_metrics['macro_f1']:.4f} | "
            f"LR={current_lr:.7f} | {epoch_seconds:.1f}s"
        )

        current_metric = val_metrics["macro_f1"]
        if current_metric > best_metric + MIN_DELTA:
            best_metric = current_metric
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_checkpoint)
            print(f"  -> Melhor checkpoint salvo (Macro-F1={best_metric:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"  -> Sem melhora: {epochs_without_improvement}/{EARLY_STOPPING_PATIENCE}")
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print("  -> Early stopping acionado.")
                break

    total_seconds = time.perf_counter() - training_start
    history_df = pd.DataFrame(history)
    history_df.to_csv(result_dir / "historico_treinamento.csv", index=False)

    summary = {
        "best_epoch": best_epoch,
        "best_val_macro_f1": float(best_metric),
        "epochs_executed": len(history),
        "training_time_seconds": total_seconds,
        "training_time_minutes": total_seconds / 60.0,
        "peak_process_rss_mb": peak_rss / (1024 ** 2),
        "best_checkpoint": str(best_checkpoint),
    }
    atomic_json_dump(summary, result_dir / "training_summary.json")

    model.load_state_dict(torch.load(best_checkpoint, map_location=device))
    return model, history_df, summary


# ============================================================
# 8. GRÁFICOS E AVALIAÇÃO
# ============================================================

def save_training_plots(history: pd.DataFrame, run_dir: Path):
    if history.empty:
        return
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    charts = [
        ("loss", "Loss", ["train_loss", "val_loss"], ["Train loss", "Validation loss"]),
        ("accuracy", "Accuracy", ["train_accuracy", "val_accuracy"], ["Train accuracy", "Validation accuracy"]),
        ("macro_f1", "Macro-F1", ["train_macro_f1", "val_macro_f1"], ["Train Macro-F1", "Validation Macro-F1"]),
    ]

    for filename, ylabel, cols, labels in charts:
        plt.figure(figsize=(8, 5))
        for col, label in zip(cols, labels):
            plt.plot(history["epoch"], history[col], label=label)
        plt.xlabel("Época")
        plt.ylabel(ylabel)
        plt.title(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / f"{filename}.png", dpi=150, bbox_inches="tight")
        plt.close()


def evaluate_test_set(model, loader, run_dir, device):
    result_dir = run_dir / "results"
    y_true, y_pred = predict_loader(model, loader, device)
    metrics = calculate_metrics(y_true, y_pred)
    atomic_json_dump(metrics, result_dir / "metricas_teste.json")

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(result_dir / "relatorio_classificacao.csv")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(result_dir / "matriz_confusao.csv")

    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Matriz de confusão - teste")
    plt.colorbar()
    ticks = np.arange(NUM_CLASSES)
    plt.xticks(ticks, CLASS_NAMES, rotation=45)
    plt.yticks(ticks, CLASS_NAMES)
    threshold = cm.max() / 2.0 if cm.max() else 0.0
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            plt.text(
                j, i, int(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black",
            )
    plt.ylabel("Classe verdadeira")
    plt.xlabel("Classe predita")
    plt.tight_layout()
    plt.savefig(result_dir / "matriz_confusao.png", dpi=150, bbox_inches="tight")
    plt.close()
    return metrics


# ============================================================
# 9. EFICIÊNCIA E INFERÊNCIA
# ============================================================

def profile_model(model, resolution: int, device: torch.device, model_name: str):
    model.eval()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dummy = torch.randn(1, 3, resolution, resolution, device=device)
    macs, thop_params = thop.profile(model, inputs=(dummy,), verbose=False)
    estimated_flops = 2 * macs
    return {
        "model": model_name,
        "input_size": f"{resolution}x{resolution}",
        "parameters": int(total_params),
        "parameters_millions": total_params / 1e6,
        "macs": int(macs),
        "macs_millions": macs / 1e6,
        "estimated_flops": int(estimated_flops),
        "estimated_flops_millions": estimated_flops / 1e6,
        "thop_parameters": int(thop_params),
        "thop_version": getattr(thop, "__version__", "desconhecido"),
    }


def synchronize(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def measure_inference(model, resolution: int, device: torch.device):
    model.eval()
    dummy = torch.randn(INFERENCE_BATCH_SIZE, 3, resolution, resolution, device=device)
    with torch.inference_mode():
        for _ in range(WARMUP_RUNS):
            model(dummy)
        synchronize(device)
        times_ms = []
        for _ in range(INFERENCE_RUNS):
            synchronize(device)
            start = time.perf_counter()
            model(dummy)
            synchronize(device)
            end = time.perf_counter()
            times_ms.append((end - start) * 1000.0)

    values = np.asarray(times_ms, dtype=float)
    mean_ms = float(values.mean())
    return {
        "batch_size": INFERENCE_BATCH_SIZE,
        "warmup_runs": WARMUP_RUNS,
        "measured_runs": INFERENCE_RUNS,
        "mean_ms_per_image": mean_ms,
        "std_ms_per_image": float(values.std(ddof=1)),
        "min_ms_per_image": float(values.min()),
        "max_ms_per_image": float(values.max()),
        "fps_from_mean": 1000.0 / mean_ms if mean_ms > 0 else None,
    }


# ============================================================
# 10. CONFIGURAÇÃO / EXECUÇÃO / RETOMADA
# ============================================================

def build_run_id(model_name: str, resolution: int, brightness: float, seed: int) -> str:
    bright = str(brightness).replace(".", "p")
    return f"{safe_name(model_name)}__res{resolution}__light{bright}__seed{seed}"


def build_config(model_name, resolution, brightness, seed, environment):
    return {
        "experiment_type": "full_experiment",
        "model": model_name,
        "pretrained_imagenet_weights": USE_PRETRAINED_WEIGHTS,
        "num_classes": NUM_CLASSES,
        "classes": CLASS_NAMES,
        "dataset_total_images": 1799,
        "split": {
            "train_percent": 70,
            "validation_percent": 10,
            "test_percent": 20,
            "split_seed": SPLIT_SEED,
        },
        "training_seed": seed,
        "input_resolution": [resolution, resolution],
        "brightness_factor": brightness,
        "preprocessing": {
            "grayscale_to_rgb": True,
            "normalization_mean": NORMALIZE_MEAN,
            "normalization_std": NORMALIZE_STD,
        },
        "augmentation": {
            "enabled": AUGMENTATION_ENABLED,
            "horizontal_flip_probability": HORIZONTAL_FLIP_P,
            "brightness_augmentation": False,
        },
        "training": {
            "framework": "PyTorch",
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "optimizer": OPTIMIZER_NAME,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "loss": LOSS_NAME,
            "scheduler": SCHEDULER_NAME,
            "early_stopping_metric": EARLY_STOPPING_METRIC,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "min_delta": MIN_DELTA,
        },
        "inference": {
            "batch_size": INFERENCE_BATCH_SIZE,
            "warmup_runs": WARMUP_RUNS,
            "measured_runs": INFERENCE_RUNS,
        },
        "environment": environment,
    }


def run_one_experiment(model_name, resolution, brightness, seed, base_df, device, environment):
    run_id = build_run_id(model_name, resolution, brightness, seed)
    run_dir = OUTPUT_ROOT / run_id
    result_dir = run_dir / "results"
    run_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    completed_path = run_dir / "COMPLETED.json"
    if RESUME and completed_path.exists():
        print(f"[SKIP] {run_id} já concluído.")
        with completed_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    set_seed(seed)
    config = build_config(model_name, resolution, brightness, seed, environment)
    atomic_json_dump(config, run_dir / "config.json")

    start_run = time.perf_counter()
    print("\n" + "=" * 80)
    print(f"EXECUÇÃO: {run_id}")
    print("=" * 80)

    train_tf, eval_tf = build_transforms(resolution, brightness)
    train_df = base_df[base_df["split"] == "train"].reset_index(drop=True)
    val_df = base_df[base_df["split"] == "val"].reset_index(drop=True)
    test_df = base_df[base_df["split"] == "test"].reset_index(drop=True)

    train_ds = NEUCLSDataset(train_df, train_tf)
    val_ds = NEUCLSDataset(val_df, eval_tf)
    test_ds = NEUCLSDataset(test_df, eval_tf)

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    )

    model = build_model(model_name).to(device)
    dummy = torch.randn(2, 3, resolution, resolution, device=device)
    with torch.inference_mode():
        output = model(dummy)
    if tuple(output.shape) != (2, NUM_CLASSES):
        raise RuntimeError(
            f"Forward inválido: entrada={tuple(dummy.shape)}, saída={tuple(output.shape)}"
        )

    efficiency = profile_model(model, resolution, device, model_name)
    atomic_csv_save(pd.DataFrame([efficiency]), result_dir / "eficiencia.csv")

    model, history, training_summary = train_model(
        model, train_loader, val_loader, device, run_dir
    )
    save_training_plots(history, run_dir)

    test_metrics = evaluate_test_set(
        model, test_loader, run_dir, device
    )

    inference = measure_inference(model, resolution, device)
    atomic_json_dump(inference, result_dir / "inferencia.json")

    total_minutes = (time.perf_counter() - start_run) / 60.0
    result = {
        "run_id": run_id,
        "model": model_name,
        "resolution": resolution,
        "brightness_factor": brightness,
        "training_seed": seed,
        "split_seed": SPLIT_SEED,
        "best_epoch": training_summary["best_epoch"],
        "best_val_macro_f1": training_summary["best_val_macro_f1"],
        "epochs_executed": training_summary["epochs_executed"],
        "training_time_minutes": training_summary["training_time_minutes"],
        "total_run_time_minutes": total_minutes,
        "accuracy": test_metrics["accuracy"],
        "macro_precision": test_metrics["macro_precision"],
        "macro_recall": test_metrics["macro_recall"],
        "macro_f1": test_metrics["macro_f1"],
        "parameters": efficiency["parameters"],
        "parameters_millions": efficiency["parameters_millions"],
        "macs": efficiency["macs"],
        "macs_millions": efficiency["macs_millions"],
        "estimated_flops": efficiency["estimated_flops"],
        "estimated_flops_millions": efficiency["estimated_flops_millions"],
        "thop_version": efficiency["thop_version"],
        "inference_mean_ms": inference["mean_ms_per_image"],
        "inference_std_ms": inference["std_ms_per_image"],
        "inference_min_ms": inference["min_ms_per_image"],
        "inference_max_ms": inference["max_ms_per_image"],
        "inference_fps": inference["fps_from_mean"],
    }

    atomic_json_dump(result, result_dir / "resumo_execucao.json")
    atomic_json_dump(result, completed_path)

    print("\nRESULTADO:")
    print(f"  Test Accuracy: {result['accuracy']:.4f}")
    print(f"  Test Macro-F1: {result['macro_f1']:.4f}")
    print(f"  Best Val Macro-F1: {result['best_val_macro_f1']:.4f}")
    print(f"  Épocas: {result['epochs_executed']}")
    print(f"  Treino: {result['training_time_minutes']:.2f} min")
    print(f"  Inferência: {result['inference_mean_ms']:.3f} ms/imagem")
    print(f"  THOP: {result['thop_version']}")

    return result


# ============================================================
# 11. CONSOLIDAÇÃO DAS 81 EXECUÇÕES
# ============================================================

def save_consolidated_results() -> None:
    if not OUTPUT_ROOT.exists():
        return

    rows = []
    for run_dir in OUTPUT_ROOT.iterdir():
        if not run_dir.is_dir():
            continue
        path = run_dir / "COMPLETED.json"
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as f:
                rows.append(json.load(f))
        except Exception:
            continue

    if not rows:
        return

    df = pd.DataFrame(rows)
    sort_cols = ["model", "resolution", "brightness_factor", "training_seed"]
    df = df.sort_values(sort_cols)
    atomic_csv_save(df, OUTPUT_ROOT / "resultados_consolidados.csv")

    group_cols = ["model", "resolution", "brightness_factor"]
    metric_cols = [
        "accuracy", "macro_precision", "macro_recall", "macro_f1",
        "inference_mean_ms", "inference_fps", "training_time_minutes",
        "epochs_executed", "best_val_macro_f1",
    ]
    present = [c for c in metric_cols if c in df.columns]

    grouped = df.groupby(group_cols)[present].agg(["mean", "std"]).reset_index()
    atomic_csv_save(
        grouped,
        OUTPUT_ROOT / "media_desvio_por_configuracao.csv",
    )

    eff_cols = [
        "parameters", "parameters_millions", "macs", "macs_millions",
        "estimated_flops", "estimated_flops_millions", "thop_version",
    ]
    present_eff = [c for c in eff_cols if c in df.columns]
    eff = df.groupby(group_cols)[present_eff].first().reset_index()
    atomic_csv_save(
        eff,
        OUTPUT_ROOT / "eficiencia_por_configuracao.csv",
    )

    simple_rows = []
    for _, group in df.groupby(group_cols):
        record = {
            "model": group["model"].iloc[0],
            "resolution": group["resolution"].iloc[0],
            "brightness_factor": group["brightness_factor"].iloc[0],
            "n_seeds": int(group["training_seed"].nunique()),
        }
        for metric in ["accuracy", "macro_precision", "macro_recall", "macro_f1"]:
            record[f"{metric}_mean"] = group[metric].mean()
            record[f"{metric}_std"] = group[metric].std(ddof=1) if len(group) > 1 else 0.0
        for metric in ["inference_mean_ms", "inference_fps", "training_time_minutes"]:
            record[f"{metric}_mean"] = group[metric].mean()
            record[f"{metric}_std"] = group[metric].std(ddof=1) if len(group) > 1 else 0.0
        record["parameters_millions"] = group["parameters_millions"].iloc[0]
        record["macs_millions"] = group["macs_millions"].iloc[0]
        record["estimated_flops_millions"] = group["estimated_flops_millions"].iloc[0]
        record["thop_version"] = group["thop_version"].iloc[0]
        simple_rows.append(record)

    atomic_csv_save(
        pd.DataFrame(simple_rows),
        OUTPUT_ROOT / "resumo_final_por_configuracao.csv",
    )


# ============================================================
# 12. MAIN
# ============================================================

def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    if CPU_THREADS is not None:
        torch.set_num_threads(CPU_THREADS)

    device = get_device()
    environment = get_environment_info(device)

    print("=" * 80)
    print("EXPERIMENTO COMPLETO - NEU-CLS")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Python: {environment['python'].split()[0]}")
    print(f"PyTorch: {environment['pytorch']}")
    print(f"TorchVision: {environment['torchvision']}")
    print(f"THOP: {environment['thop']}")
    print(f"THOP path: {environment['thop_path']}")
    print(f"CUDA disponível: {environment['cuda_available']}")

    if environment["thop"] != "2.1.6":
        raise RuntimeError(
            "THOP 2.1.6 não está ativo. "
            f"Versão encontrada: {environment['thop']}. "
            "Corrija o ambiente antes do experimento."
        )

    atomic_json_dump(environment, OUTPUT_ROOT / "environment.json")

    df = load_split_dataframe()
    df = validate_split(df)
    atomic_csv_save(df, OUTPUT_ROOT / "dataset_validado.csv")

    experiments = [
        (model, resolution, brightness, seed)
        for model in MODELS_TO_RUN
        for resolution in RESOLUTIONS
        for brightness in BRIGHTNESS_FACTORS
        for seed in TRAINING_SEEDS
    ]

    print(f"\nExecuções planejadas: {len(experiments)}")

    if MAX_EXPERIMENTS is not None:
        experiments = experiments[:MAX_EXPERIMENTS]
        print(f"Modo teste: apenas {len(experiments)} execução(ões).")

    for index, (model, resolution, brightness, seed) in enumerate(experiments, start=1):
        print(f"\n[{index}/{len(experiments)}]")
        try:
            run_one_experiment(
                model,
                resolution,
                brightness,
                seed,
                df,
                device,
                environment,
            )
        except KeyboardInterrupt:
            print("\nExecução interrompida. Resultados já concluídos foram preservados.")
            save_consolidated_results()
            raise
        except Exception as exc:
            run_id = build_run_id(model, resolution, brightness, seed)
            error_dir = OUTPUT_ROOT / run_id
            error_dir.mkdir(parents=True, exist_ok=True)
            atomic_json_dump(
                {
                    "run_id": run_id,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                },
                error_dir / "ERROR.json",
            )
            print(f"ERRO em {run_id}: {type(exc).__name__}: {exc}")
            print("Continuando para a próxima configuração.")
        finally:
            # Consolida após cada tentativa para não perder progresso.
            save_consolidated_results()

    save_consolidated_results()
    print("\n" + "=" * 80)
    print("FIM DO PROCESSAMENTO")
    print("=" * 80)
    print(f"Resultados: {OUTPUT_ROOT.resolve()}")
    print("RESUME=True permite executar novamente e pular execuções concluídas.")


if __name__ == "__main__":
    main()
