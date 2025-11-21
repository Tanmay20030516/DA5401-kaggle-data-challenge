import os
import json
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download

from sklearn.model_selection import train_test_split
from math import sqrt

warnings.filterwarnings("ignore")
os.environ['TOKENIZERS_PARALLELISM'] = 'true'


# ===========================
# 0. CONFIG & SEEDING
# ===========================
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

set_seed(42)

CONFIG = {
    "encoder_name": "l3cube-pune/indic-sentence-similarity-sbert",
    "train_data_path": "/kaggle/input/processed-da5401-challenge-data/kaggle/working/train_df.parquet",
    "test_data_path": "/kaggle/input/processed-da5401-challenge-data/kaggle/working/test_df.parquet",
    "metric_names_path": "/kaggle/input/da5401-2025-data-challenge/metric_names.json",
    "metric_embeddings_path": "/kaggle/input/da5401-2025-data-challenge/metric_name_embeddings.npy",
    "save_dir": "/kaggle/working/checkpoints",
    "submission_path": "/kaggle/working/submission.csv",
    "batch_size_encode": 32,
    "batch_size": 64,
    "learning_rate": 2e-4,
    "num_epochs": 25,
    "hidden_dims": [1024, 512, 256, 128],
    "dropout": 0.3,
    "val_split": 0.1,
    "margin": 0.6,
    "neg_samples_per_positive": 4,
    "score_threshold": 7,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# ===========================
# 1. EMBEDDING HELPERS
# ===========================
def precompute_response_embeddings(data_df, encoder_model, batch_size=64):
    print("Precomputing response embeddings (SentenceTransformer)...")
    unique_responses = data_df["response"].astype(str).unique()
    unique_responses = list(unique_responses)
    n = len(unique_responses)
    print(f"  Unique responses: {n}")
    response_to_idx = {resp: idx for idx, resp in enumerate(unique_responses)}
    all_embeddings = encoder_model.encode(
        unique_responses,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    all_embeddings = np.asarray(all_embeddings)
    print(f"  Embeddings shape: {all_embeddings.shape}")
    return all_embeddings, response_to_idx, unique_responses

# ===========================
# 2. DATA LOADING
# ===========================
def load_and_prepare_data(data_path, metric_names_path, metric_embeddings_path, is_test=False):
    print(f"Loading data from: {data_path}")
    data_df = pd.read_parquet(data_path)
    print(f"  Rows: {len(data_df)}, Columns: {len(data_df.columns)}")

    print(f"Loading metric names from: {metric_names_path}")
    with open(metric_names_path, "r") as f:
        metric_names = json.load(f)

    print(f"Loading metric embeddings from: {metric_embeddings_path}")
    metric_embeddings = np.load(metric_embeddings_path)

    metric_lookup_df = pd.DataFrame(
        {"metric_name": metric_names, "metric_embedding": list(metric_embeddings)}
    )
    print(f"Metric lookup: {len(metric_lookup_df)} entries")

    data_with_embeddings = data_df.merge(metric_lookup_df, on="metric_name", how="left")
    missing_embeddings = data_with_embeddings["metric_embedding"].isna().sum()
    if missing_embeddings > 0:
        print(f"⚠ Warning: {missing_embeddings} rows missing metric embeddings")

    if not is_test:
        print("\nScore statistics (train):")
        print(data_with_embeddings["score"].describe())
        data_with_embeddings["label"] = (data_with_embeddings["score"] >= CONFIG["score_threshold"]).astype(int)
        print("\nLabel distribution:")
        print(data_with_embeddings["label"].value_counts())

    return data_with_embeddings

# ===========================
# 3. SYNTHETIC HARD NEGATIVES
# ===========================
def generate_hard_negatives(data_df, neg_samples_per_positive=4, seed=42):
    print(f"\nGenerating hard negatives ({neg_samples_per_positive} per sample)...")
    np.random.seed(seed)
    synthetic_negatives = []
    metric_groups = data_df.groupby("metric_name")["response"].apply(list).to_dict()
    all_metrics = list(metric_groups.keys())
    for _, row in tqdm(data_df.iterrows(), total=len(data_df), desc="Creating negatives"):
        current_metric = row["metric_name"]
        other_metrics = [m for m in all_metrics if m != current_metric]
        if len(other_metrics) == 0:
            continue
        for _ in range(neg_samples_per_positive):
            random_metric = np.random.choice(other_metrics)
            random_response = np.random.choice(metric_groups[random_metric])
            neg_sample = {
                "metric_name": row["metric_name"],
                "response": random_response,
                "score": 0,
                "label": 0,
                "metric_embedding": row["metric_embedding"],
                "is_synthetic": True,
            }
            synthetic_negatives.append(neg_sample)

    synthetic_df = pd.DataFrame(synthetic_negatives)
    data_df = data_df.copy()
    data_df["is_synthetic"] = False
    combined_df = pd.concat([data_df, synthetic_df], ignore_index=True)
    print(f"  Original samples: {len(data_df)}")
    print(f"  Synthetic negatives: {len(synthetic_df)}")
    print(f"  Combined total: {len(combined_df)}")
    if "label" in combined_df.columns:
        print("\nFinal label distribution (combined):")
        print(combined_df["label"].value_counts())
    return combined_df

# ===========================
# 4. DATASET
# ===========================
class ContrastiveMetricResponseDataset(Dataset):
    def __init__(self, dataframe, response_embeddings, response_to_idx, is_test=False):
        self.data = dataframe.reset_index(drop=True)
        self.response_embeddings = response_embeddings
        self.response_to_idx = response_to_idx
        self.is_test = is_test
        valid_mask = self.data["metric_embedding"].notna()
        if not valid_mask.all():
            removed = (~valid_mask).sum()
            print(f"Removing {removed} rows with missing metric embeddings")
            self.data = self.data[valid_mask].reset_index(drop=True)
        print(f"Dataset initialized: {len(self.data)} samples (test={is_test})")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        resp = str(row["response"])
        response_idx = self.response_to_idx.get(resp, 0)
        response_embedding = torch.FloatTensor(self.response_embeddings[response_idx])
        metric_embedding = torch.FloatTensor(row["metric_embedding"])
        result = {
            "response_embedding": response_embedding,
            "metric_embedding": metric_embedding,
        }
        if not self.is_test:
            result["label"] = torch.FloatTensor([row["label"]])
            if "score" in row and not pd.isna(row["score"]):
                result["score"] = torch.FloatTensor([row["score"]])
        return result

# ===========================
# 5. MODEL
# ===========================
class MetricResponseMatcher(nn.Module):
    def __init__(self, response_embedding_dim, metric_embedding_dim, hidden_dims=[512,256,128], dropout=0.3):
        super(MetricResponseMatcher, self).__init__()
        combined_dim = response_embedding_dim + metric_embedding_dim
        layers = []
        input_dim = combined_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(0.1), nn.BatchNorm1d(hidden_dim), nn.Dropout(dropout)])
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)
        self.config = {"response_embedding_dim": response_embedding_dim, "metric_embedding_dim": metric_embedding_dim, "hidden_dims": hidden_dims, "dropout": dropout}
        print(f"Model init: response_dim={response_embedding_dim}, metric_dim={metric_embedding_dim}, layers={hidden_dims}")

    def forward(self, response_embedding, metric_embedding):
        combined = torch.cat([response_embedding, metric_embedding], dim=1)
        logits = self.network(combined)
        similarity = torch.sigmoid(logits)
        return similarity

# ===========================
# 6. LOSS
# ===========================
class ContrastiveMarginLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveMarginLoss, self).__init__()
        self.margin = margin
        print(f"ContrastiveMarginLoss initialized (margin={self.margin})")
    def forward(self, similarities, labels):
        distances = 1.0 - similarities
        similar_loss = labels * torch.pow(distances, 2)
        dissimilar_loss = (1 - labels) * torch.pow(torch.clamp(self.margin - distances, min=0.0), 2)
        loss = similar_loss + dissimilar_loss
        return loss.mean()

# ===========================
# 7. HELPERS: METRICS & RMSE
# ===========================
def calculate_metrics(similarities, labels, threshold=0.5):
    similarities_np = similarities.cpu().numpy().flatten()
    labels_np = labels.cpu().numpy().flatten()
    predictions = (similarities_np >= threshold).astype(int)
    accuracy = (predictions == labels_np).mean()
    return {"accuracy": accuracy, "mean_similarity": similarities_np.mean(), "std_similarity": similarities_np.std()}

def compute_rmse_on_loader(model, dataloader, device):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for batch in dataloader:
            response_embedding = batch["response_embedding"].to(device)
            metric_embedding = batch["metric_embedding"].to(device)
            similarities = model(response_embedding, metric_embedding)
            preds.extend((similarities.cpu().numpy().flatten() * 10.0).tolist())
            if "score" in batch:
                trues.extend(batch["score"].cpu().numpy().flatten().tolist())
            else:
                return None, None
    preds = np.array(preds)
    trues = np.array(trues)
    rmse = float(np.sqrt(np.mean((preds - trues) ** 2)))
    return preds, rmse

# ===========================
# 8. TRAIN/EVAL/TEST LOOPS
# ===========================
def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    all_similarities = []
    all_labels = []
    pb = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch in pb:
        response_embedding = batch["response_embedding"].to(device)
        metric_embedding = batch["metric_embedding"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        similarities = model(response_embedding, metric_embedding)
        loss = criterion(similarities, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        all_similarities.append(similarities.detach())
        all_labels.append(labels.detach())
        pb.set_postfix({"loss": loss.item()})
    all_similarities = torch.cat(all_similarities)
    all_labels = torch.cat(all_labels)
    metrics = calculate_metrics(all_similarities, all_labels)
    metrics["loss"] = total_loss / len(dataloader)
    return metrics

def evaluate(model, dataloader, criterion, device, mode="Validation"):
    model.eval()
    total_loss = 0.0
    all_similarities = []
    all_labels = []
    pb = tqdm(dataloader, desc=f"[{mode}]")
    with torch.no_grad():
        for batch in pb:
            response_embedding = batch["response_embedding"].to(device)
            metric_embedding = batch["metric_embedding"].to(device)
            labels = batch["label"].to(device)
            similarities = model(response_embedding, metric_embedding)
            loss = criterion(similarities, labels)
            total_loss += loss.item()
            all_similarities.append(similarities)
            all_labels.append(labels)
            pb.set_postfix({"loss": loss.item()})
    all_similarities = torch.cat(all_similarities)
    all_labels = torch.cat(all_labels)
    metrics = calculate_metrics(all_similarities, all_labels)
    metrics["loss"] = total_loss / len(dataloader)
    return metrics, all_similarities, all_labels

def test_and_save(model, dataloader, device, output_path="submission.csv"):
    model.eval()
    all_similarities = []
    pb = tqdm(dataloader, desc="[Testing]")
    with torch.no_grad():
        for batch in pb:
            response_embedding = batch["response_embedding"].to(device)
            metric_embedding = batch["metric_embedding"].to(device)
            similarities = model(response_embedding, metric_embedding)
            all_similarities.append(similarities)
    all_similarities = torch.cat(all_similarities).cpu().numpy().flatten()
    predicted_scores = all_similarities * 10.0
    submission_df = pd.DataFrame({"score": predicted_scores})
    submission_df.index = range(1, len(submission_df) + 1)
    submission_df.to_csv(output_path, index_label="id")
    print(f"\n✓ Predictions saved to {output_path} (count={len(submission_df)})")
    print(f"  Score range: [{predicted_scores.min():.4f}, {predicted_scores.max():.4f}]  Mean: {predicted_scores.mean():.4f}")
    return predicted_scores

def save_model_complete(model, optimizer, epoch, metrics, config, save_path):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": model.config,
        "training_config": config,
        "metrics": metrics,
    }
    torch.save(checkpoint, save_path)
    print(f"✓ Model saved to {save_path}")

def load_model_for_inference(checkpoint_path, device="cuda"):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint["model_config"]
    model = MetricResponseMatcher(
        response_embedding_dim=model_config["response_embedding_dim"],
        metric_embedding_dim=model_config["metric_embedding_dim"],
        hidden_dims=model_config["hidden_dims"],
        dropout=model_config["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"✓ Model loaded from {checkpoint_path} (trained_epochs={checkpoint.get('epoch','?')})")
    return model, checkpoint

# ===========================
# 9. MAIN
# ===========================
def main():
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    print("=" * 70)
    print("METRIC-RESPONSE MATCHING VIA CONTRASTIVE LEARNING (SentenceTransformer embeddings)")
    print("=" * 70)
    print(f"Device: {CONFIG['device']}")
    print(f"Encoder: {CONFIG['encoder_name']}")
    print(f"Strategy: score >= {CONFIG['score_threshold']} -> positive (label=1)")
    print("=" * 70)

    # download snapshot and load SentenceTransformer from local path
    print("\n[0/9] Snapshot-downloading encoder repo:", CONFIG["encoder_name"])
    model_path = snapshot_download(repo_id=CONFIG["encoder_name"], cache_dir="./model_cache", ignore_patterns=["*.md", "*.txt"])
    print(f"  snapshot_download -> {model_path}")

    print("\n[1/9] Loading SentenceTransformer from snapshot...")
    encoder_model = SentenceTransformer(model_path, device=CONFIG["device"])
    print("✓ Encoder loaded")

    # load original train data (unaugmented)
    print("\n[2/9] Loading training data & metric embeddings...")
    train_data_orig = load_and_prepare_data(CONFIG["train_data_path"], CONFIG["metric_names_path"], CONFIG["metric_embeddings_path"], is_test=False)
    print(f"  Loaded train rows: {len(train_data_orig)}")

    # precompute response embeddings from original train (covers responses used in augmentation)
    print("\n[3/9] Precomputing response embeddings (train unique responses)...")
    response_embeddings, response_to_idx, unique_responses = precompute_response_embeddings(train_data_orig, encoder_model, batch_size=CONFIG["batch_size_encode"])

    # generate augmented training data
    print("\n[4/9] Generating hard negatives and augmenting train set...")
    train_data_aug = generate_hard_negatives(train_data_orig, neg_samples_per_positive=CONFIG["neg_samples_per_positive"])

    # split augmented data for training
    print("\n[5/9] Creating train/val splits (augmented)...")
    train_df_aug, val_df_aug = train_test_split(train_data_aug, test_size=CONFIG["val_split"], random_state=42)
    print(f"  Augmented Train: {len(train_df_aug)}, Augmented Val: {len(val_df_aug)}")

    # also create unaugmented train/val splits (for final eval + saving predictions)
    train_df_orig, val_df_orig = train_test_split(train_data_orig, test_size=CONFIG["val_split"], random_state=42)
    print(f"  Unaugmented Train: {len(train_df_orig)}, Unaugmented Val: {len(val_df_orig)}")

    metric_embedding_dim = len(train_df_orig.iloc[0]["metric_embedding"])
    response_embedding_dim = response_embeddings.shape[1]
    print(f"\nEmbedding dims -> response: {response_embedding_dim}, metric: {metric_embedding_dim}")

    train_dataset = ContrastiveMetricResponseDataset(train_df_aug, response_embeddings, response_to_idx, is_test=False)
    val_dataset = ContrastiveMetricResponseDataset(val_df_aug, response_embeddings, response_to_idx, is_test=False)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=2, pin_memory=True if CONFIG["device"] == "cuda" else False)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2, pin_memory=True if CONFIG["device"] == "cuda" else False)

    # 7) Model / optimizer / loss
    print("\n[6/9] Initializing model, optimizer, loss, scheduler...")
    model = MetricResponseMatcher(response_embedding_dim=response_embedding_dim, metric_embedding_dim=metric_embedding_dim, hidden_dims=CONFIG["hidden_dims"], dropout=CONFIG["dropout"]).to(CONFIG["device"])

    criterion = ContrastiveMarginLoss(margin=CONFIG["margin"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, verbose=True)

    history = {"train_loss": [], "train_accuracy": [], "train_rmse": [], "val_loss": [], "val_accuracy": [], "val_rmse": []}
    best_val_loss = float("inf")

    # 8) Training loop with RMSE logging per epoch (on augmented datasets)
    print("\n[7/9] Starting training...")
    for epoch in range(1, CONFIG["num_epochs"] + 1):
        print(f"\n--- Epoch {epoch}/{CONFIG['num_epochs']} ---")
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, CONFIG["device"], epoch)
        val_metrics, _, _ = evaluate(model, val_loader, criterion, CONFIG["device"], mode="Validation")

        # compute RMSE on augmented train & val
        _, train_rmse = compute_rmse_on_loader(model, train_loader, CONFIG["device"])
        _, val_rmse = compute_rmse_on_loader(model, val_loader, CONFIG["device"])

        scheduler.step(val_metrics["loss"])

        print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, RMSE: {train_rmse:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, RMSE: {val_rmse:.4f}")

        history["train_loss"].append(train_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["train_rmse"].append(train_rmse)
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_rmse"].append(val_rmse)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_model_complete(model, optimizer, epoch, {"val_loss": best_val_loss, "val_accuracy": val_metrics["accuracy"]}, CONFIG, os.path.join(CONFIG["save_dir"], "best_model.pt"))
            print(f"  -> New best model saved (val_loss={best_val_loss:.4f})")

    pd.DataFrame(history).to_csv(os.path.join(CONFIG["save_dir"], "training_history.csv"), index=False)
    print(f"\nTraining finished. History saved to {os.path.join(CONFIG['save_dir'], 'training_history.csv')}")

    # 9) Final: load best model and evaluate on unaugmented train/val and save predictions + RMSE
    print("\n[8/9] Loading best model for final unaugmented evaluation...")
    best_model, _ = load_model_for_inference(os.path.join(CONFIG["save_dir"], "best_model.pt"), CONFIG["device"])

    # create unaugmented datasets + loaders (they contain true 'score' field)
    print("Creating unaugmented datasets for final evaluation...")
    train_orig_dataset = ContrastiveMetricResponseDataset(train_df_orig, response_embeddings, response_to_idx, is_test=False)
    val_orig_dataset = ContrastiveMetricResponseDataset(val_df_orig, response_embeddings, response_to_idx, is_test=False)

    train_orig_loader = DataLoader(train_orig_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2, pin_memory=True if CONFIG["device"] == "cuda" else False)
    val_orig_loader = DataLoader(val_orig_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2, pin_memory=True if CONFIG["device"] == "cuda" else False)

    # compute predictions and RMSE for unaugmented splits and save CSVs
    def save_preds_and_rmse(model, df_orig, loader, out_csv):
        preds = []
        with torch.no_grad():
            for batch in loader:
                response_embedding = batch["response_embedding"].to(CONFIG["device"])
                metric_embedding = batch["metric_embedding"].to(CONFIG["device"])
                sims = model(response_embedding, metric_embedding)
                preds.extend((sims.cpu().numpy().flatten() * 10.0).tolist())
        df_out = df_orig.reset_index(drop=True).copy()
        df_out["predicted_score"] = preds
        rmse = float(np.sqrt(((df_out["predicted_score"] - df_out["score"]) ** 2).mean()))
        df_out.to_csv(out_csv, index=False)
        print(f"Saved predictions to {out_csv} (rmse={rmse:.4f})")
        return rmse, out_csv

    print("\n[9/9] Evaluating best model on unaugmented train/val and saving predictions...")
    train_rmse_final, train_pred_path = save_preds_and_rmse(best_model, train_df_orig, train_orig_loader, os.path.join(CONFIG["save_dir"], "train_unaugmented_predictions.csv"))
    val_rmse_final, val_pred_path = save_preds_and_rmse(best_model, val_df_orig, val_orig_loader, os.path.join(CONFIG["save_dir"], "val_unaugmented_predictions.csv"))

    print("\nFinal unaugmented RMSEs:")
    print(f"  Train RMSE: {train_rmse_final:.4f}  -> saved: {train_pred_path}")
    print(f"  Val   RMSE: {val_rmse_final:.4f}  -> saved: {val_pred_path}")

    # Test inference
    print("\nRunning test-time inference and saving submission...")
    test_data = load_and_prepare_data(CONFIG["test_data_path"], CONFIG["metric_names_path"], CONFIG["metric_embeddings_path"], is_test=True)
    print("Precomputing response embeddings for test data...")
    test_response_embeddings, test_response_to_idx, test_unique_responses = precompute_response_embeddings(test_data, encoder_model, batch_size=CONFIG["batch_size_encode"])
    test_dataset = ContrastiveMetricResponseDataset(test_data, test_response_embeddings, test_response_to_idx, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2, pin_memory=True if CONFIG["device"] == "cuda" else False)
    predicted_scores = test_and_save(best_model, test_loader, CONFIG["device"], output_path=CONFIG["submission_path"])

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\nOutputs:\n  - Best model: {os.path.join(CONFIG['save_dir'], 'best_model.pt')}\n  - Training history: {os.path.join(CONFIG['save_dir'], 'training_history.csv')}\n  - Train preds: {train_pred_path}\n  - Val preds: {val_pred_path}\n  - Submission: {CONFIG['submission_path']}")

if __name__ == "__main__":
    main()
