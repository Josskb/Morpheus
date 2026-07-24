"""
Module 4: XGBoost Model Training & Evaluation
Entraîne un modèle pour prédire la profitabilité des calls
"""

from __future__ import annotations

import sqlite3
import json
import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from loguru import logger


class ModelMetrics(NamedTuple):
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float
    confusion: dict


class XGBoostTrainer:
    """Entraîne et évalue un modèle XGBoost sur les données tweets."""

    MODEL_DIR = Path(__file__).parent.parent.parent / "models"

    def __init__(self, db_path: str, window_hours: int = 24):
        """
        Args:
            db_path: Chemin vers la base SQLite
            window_hours: Heures pour évaluer profitabilité (1h, 4h, 24h)
        """
        self.db_path = db_path
        self.window_hours = window_hours
        self.model = None
        self.scaler = None
        self.feature_names = None

    def load_data(self) -> tuple[pd.DataFrame, np.ndarray]:
        """Charge et prépare les données de la base de données."""
        logger.info(f"Chargement données depuis {self.db_path}")

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        # Requête pour récupérer tweets + snapshots
        query = """
        SELECT
            t.id,
            t.confidence,
            t.sentiment,
            t.urgency_score,
            t.call_type,
            t.like_count,
            t.retweet_count,
            t.reply_count,
            a.reliability_score,
            a.win_rate,
            a.avg_roi,
            COALESCE(ms.change_24h, 0) as price_change_24h,
            COALESCE(ms.change_4h, 0) as price_change_4h,
            COALESCE(ms.change_1h, 0) as price_change_1h
        FROM tweets t
        LEFT JOIN accounts a ON t.account_id = a.id
        LEFT JOIN market_snapshots ms ON t.id = ms.tweet_id
        WHERE t.nlp_processed = 1
            AND t.confidence IS NOT NULL
            AND t.call_type IN ('long', 'short')
        GROUP BY t.id
        """

        df = pd.read_sql_query(query, conn)
        conn.close()

        logger.info(f"Chargé {len(df)} tweets avec données complètes")

        # Cible: profitabilité (price_change_24h > 0 pour longs, < 0 pour shorts)
        profitable = []
        for _, row in df.iterrows():
            change = row["price_change_24h"]
            call_type = row["call_type"]

            if change is None or np.isnan(change):
                profitable.append(0)  # Neutral
            elif call_type == "long":
                profitable.append(1 if change > 0 else 0)
            else:  # short
                profitable.append(1 if change < 0 else 0)

        y = np.array(profitable)

        logger.info(f"Distribution profitabilité: {np.bincount(y)}")

        return df, y

    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prépare les features pour le modèle."""
        features_to_use = [
            "confidence",
            "sentiment",
            "urgency_score",
            "like_count",
            "retweet_count",
            "reply_count",
            "reliability_score",
            "win_rate",
            "avg_roi",
            "price_change_4h",
            "price_change_1h",
        ]

        # Encode call_type
        call_type_mapping = {"long": 1, "short": -1}
        df["call_type_encoded"] = df["call_type"].map(call_type_mapping)
        features_to_use.append("call_type_encoded")

        # Fill NaN
        df_features = df[features_to_use].fillna(0)

        # Normalize
        df_features = (df_features - df_features.mean()) / (df_features.std() + 1e-8)

        self.feature_names = features_to_use
        logger.info(f"Features: {self.feature_names}")

        return df_features.values

    def train(self, test_size: float = 0.2) -> ModelMetrics:
        """Entraîne le modèle XGBoost."""
        logger.info("Chargement et préparation des données...")
        df, y = self.load_data()

        logger.info("Préparation des features...")
        X = self.prepare_features(df)

        logger.info(f"Train/test split: {1-test_size:.0%}/{test_size:.0%}")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        logger.info("Entraînement XGBoost...")
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        )

        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=10,
            verbose=False,
        )

        logger.info("Évaluation du modèle...")
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        metrics = ModelMetrics(
            accuracy=float(accuracy_score(y_test, y_pred)),
            precision=float(precision_score(y_test, y_pred, zero_division=0)),
            recall=float(recall_score(y_test, y_pred, zero_division=0)),
            f1=float(f1_score(y_test, y_pred, zero_division=0)),
            auc=float(roc_auc_score(y_test, y_pred_proba)),
            confusion={
                "tn": int(confusion_matrix(y_test, y_pred)[0, 0]),
                "fp": int(confusion_matrix(y_test, y_pred)[0, 1]),
                "fn": int(confusion_matrix(y_test, y_pred)[1, 0]),
                "tp": int(confusion_matrix(y_test, y_pred)[1, 1]),
            },
        )

        logger.info(f"Accuracy:  {metrics.accuracy:.3f}")
        logger.info(f"Precision: {metrics.precision:.3f}")
        logger.info(f"Recall:    {metrics.recall:.3f}")
        logger.info(f"F1:        {metrics.f1:.3f}")
        logger.info(f"AUC:       {metrics.auc:.3f}")

        logger.info(f"Confusion Matrix: TP={metrics.confusion['tp']}, "
                   f"FP={metrics.confusion['fp']}, "
                   f"FN={metrics.confusion['fn']}, "
                   f"TN={metrics.confusion['tn']}")

        # Feature importance
        logger.info("Feature Importance:")
        for name, importance in zip(self.feature_names, self.model.feature_importances_):
            logger.info(f"  {name:20} {importance:.4f}")

        return metrics

    def save_model(self, version: str = "v1") -> Path:
        """Sauvegarde le modèle entraîné."""
        if self.model is None:
            raise ValueError("Model not trained yet")

        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)

        model_path = self.MODEL_DIR / f"xgboost_{version}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "feature_names": self.feature_names,
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            }, f)

        logger.info(f"Modèle sauvegardé: {model_path}")
        return model_path

    def load_model(self, version: str = "v1") -> None:
        """Charge un modèle pré-entraîné."""
        model_path = self.MODEL_DIR / f"xgboost_{version}.pkl"

        with open(model_path, "rb") as f:
            data = pickle.load(f)

        self.model = data["model"]
        self.feature_names = data["feature_names"]

        logger.info(f"Modèle chargé: {model_path}")

    def predict(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Prédit la profitabilité."""
        if self.model is None:
            raise ValueError("Model not loaded")

        predictions = self.model.predict(features)
        probabilities = self.model.predict_proba(features)[:, 1]

        return predictions, probabilities


def main():
    """Entraîne et évalue le modèle XGBoost."""
    db_path = Path(__file__).parent.parent.parent / "data" / "radar.db"

    logger.info("=" * 80)
    logger.info("MODULE 4: XGBOOST TRAINING")
    logger.info("=" * 80)

    trainer = XGBoostTrainer(str(db_path), window_hours=24)
    metrics = trainer.train(test_size=0.2)

    # Save model
    model_path = trainer.save_model("v1")

    logger.info("\n" + "=" * 80)
    logger.info("RÉSULTATS")
    logger.info("=" * 80)
    logger.info(f"Accuracy:  {metrics.accuracy:.1%}")
    logger.info(f"Precision: {metrics.precision:.1%}")
    logger.info(f"Recall:    {metrics.recall:.1%}")
    logger.info(f"F1:        {metrics.f1:.1%}")
    logger.info(f"AUC:       {metrics.auc:.3f}")
    logger.info(f"\nModèle sauvegardé: {model_path}")


if __name__ == "__main__":
    main()
