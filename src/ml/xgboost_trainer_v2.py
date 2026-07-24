"""
Phase 2: XGBoost Model Training with Feature Engineering
Improved features: engagement rates, recency, account age, outlier handling
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
from sklearn.preprocessing import StandardScaler
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
    samples_train: int
    samples_test: int
    outliers_removed: int


class XGBoostTrainerV2:
    """Phase 2: Enhanced XGBoost trainer with feature engineering."""

    MODEL_DIR = Path(__file__).parent.parent.parent / "models"

    def __init__(self, db_path: str, window_hours: int = 24):
        self.db_path = db_path
        self.window_hours = window_hours
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.outliers_removed = 0

    def load_data(self) -> tuple[pd.DataFrame, np.ndarray]:
        """Load tweets with enhanced data."""
        logger.info(f"Loading data from {self.db_path}")

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        query = """
        SELECT
            t.id,
            t.tweet_id,
            t.created_at,
            t.confidence,
            t.sentiment,
            t.urgency_score,
            COALESCE(t.call_type, 'long') as call_type,
            t.like_count,
            t.retweet_count,
            t.reply_count,
            a.reliability_score,
            a.win_rate,
            a.avg_roi,
            a.created_at as account_created_at,
            COALESCE(ms.change_24h, 0) as price_change_24h,
            COALESCE(ms.change_4h, 0) as price_change_4h,
            COALESCE(ms.change_1h, 0) as price_change_1h
        FROM tweets t
        LEFT JOIN accounts a ON t.account_id = a.id
        LEFT JOIN market_snapshots ms ON t.id = ms.tweet_id
        WHERE t.nlp_processed = 1
            AND t.confidence IS NOT NULL
            AND t.confidence >= 0.3
        GROUP BY t.id
        ORDER BY t.created_at DESC
        """

        df = pd.read_sql_query(query, conn)
        conn.close()

        logger.info(f"Loaded {len(df)} tweets with complete data")

        # Define profitability: +2% = profitable for long, -2% = profitable for short
        PROFIT_THRESHOLD = 2.0
        profitable = []

        for _, row in df.iterrows():
            change = row["price_change_24h"]
            call_type = row["call_type"]

            if change is None or np.isnan(change):
                profitable.append(0)
            elif call_type == "long":
                profitable.append(1 if change >= PROFIT_THRESHOLD else 0)
            else:  # short
                profitable.append(1 if change <= -PROFIT_THRESHOLD else 0)

        y = np.array(profitable, dtype=int)

        logger.info(f"Target distribution: {np.bincount(y)} (profitable threshold: ±{PROFIT_THRESHOLD}%)")

        return df, y

    def prepare_features(self, df: pd.DataFrame) -> tuple[np.ndarray, int]:
        """Engineer features for Phase 2."""

        # Phase 2 Features
        features_dict = {}

        # 1. Basic NLP features
        features_dict["confidence"] = df["confidence"].fillna(0)
        features_dict["sentiment"] = df["sentiment"].fillna(0)
        features_dict["urgency_score"] = df["urgency_score"].fillna(0)

        # 2. Engagement features (NEW)
        like_count = df["like_count"].fillna(0)
        retweet_count = df["retweet_count"].fillna(0)
        reply_count = df["reply_count"].fillna(0)

        features_dict["like_count"] = like_count
        features_dict["retweet_count"] = retweet_count
        features_dict["reply_count"] = reply_count

        # Engagement ratios (safe division)
        total_engagement = like_count + retweet_count + reply_count
        features_dict["engagement_rate"] = np.where(
            total_engagement > 0,
            (retweet_count + reply_count) / total_engagement,
            0
        )
        features_dict["retweet_ratio"] = np.where(
            like_count > 0,
            retweet_count / like_count,
            0
        )

        # 3. Account features
        features_dict["reliability_score"] = df["reliability_score"].fillna(0.5)
        features_dict["win_rate"] = df["win_rate"].fillna(0.5)
        features_dict["avg_roi"] = df["avg_roi"].fillna(0)

        # Account age in days (NEW)
        df["account_created_at"] = pd.to_datetime(df["account_created_at"], errors="coerce", utc=True)
        df["tweet_created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
        now = pd.Timestamp.now(tz=timezone.utc)

        account_age_days = []
        for _, row in df.iterrows():
            if pd.isna(row["account_created_at"]):
                account_age_days.append(365)  # Default to 1 year
            else:
                age = (now - row["account_created_at"]).days
                account_age_days.append(max(age, 1))

        features_dict["account_age_days"] = np.array(account_age_days)

        # Tweet recency in hours (NEW)
        tweet_age_hours = []
        for _, row in df.iterrows():
            if pd.isna(row["tweet_created_at"]):
                tweet_age_hours.append(0)
            else:
                age = (now - row["tweet_created_at"]).total_seconds() / 3600
                tweet_age_hours.append(max(age, 0))

        features_dict["tweet_age_hours"] = np.array(tweet_age_hours)

        # 4. Call type encoding
        call_type_mapping = {"long": 1, "short": -1}
        features_dict["call_type_encoded"] = df["call_type"].map(call_type_mapping).fillna(0)

        # 5. Market context
        features_dict["price_change_4h"] = df["price_change_4h"].fillna(0)
        features_dict["price_change_1h"] = df["price_change_1h"].fillna(0)

        # Build feature array
        feature_names = [
            "confidence", "sentiment", "urgency_score",
            "like_count", "retweet_count", "reply_count",
            "engagement_rate", "retweet_ratio",
            "reliability_score", "win_rate", "avg_roi",
            "account_age_days", "tweet_age_hours",
            "call_type_encoded",
            "price_change_4h", "price_change_1h",
        ]

        X = np.column_stack([features_dict[name] for name in feature_names])

        self.feature_names = feature_names
        logger.info(f"Features ({len(feature_names)}): {feature_names}")

        # Outlier removal (OUTLIER HANDLING - Phase 2)
        # Remove extreme price changes that break the model
        price_changes = df["price_change_24h"].fillna(0)
        outlier_mask = (price_changes.abs() <= 10.0)

        outliers_removed = (~outlier_mask).sum()
        self.outliers_removed = outliers_removed

        if outliers_removed > 0:
            logger.warning(
                f"Removing {outliers_removed} extreme price changes (>10% or <-10%)"
            )

        return X[outlier_mask], outliers_removed

    def train(self, test_size: float = 0.2) -> ModelMetrics:
        """Train Phase 2 XGBoost model."""
        logger.info("Loading and preparing data...")
        df, y = self.load_data()

        logger.info("Engineering features (Phase 2)...")
        X, outliers_removed = self.prepare_features(df)

        # Remove outliers from target as well
        y = y[np.abs(df["price_change_24h"].fillna(0)) <= 10.0]

        logger.info(f"Final dataset: {len(X)} samples (removed {outliers_removed} outliers)")
        logger.info(f"Train/test split: {1-test_size:.0%}/{test_size:.0%}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        logger.info("Training XGBoost (Phase 2)...")
        self.model = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.7,
            min_child_weight=2,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        )

        self.model.fit(
            X_train_scaled,
            y_train,
            verbose=False,
        )

        logger.info("Evaluating model...")
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]

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
            samples_train=len(X_train),
            samples_test=len(X_test),
            outliers_removed=outliers_removed,
        )

        logger.info("=" * 80)
        logger.info("PHASE 2 MODEL PERFORMANCE")
        logger.info("=" * 80)
        logger.info(f"Accuracy:  {metrics.accuracy:.3f}")
        logger.info(f"Precision: {metrics.precision:.3f}")
        logger.info(f"Recall:    {metrics.recall:.3f}")
        logger.info(f"F1:        {metrics.f1:.3f}")
        logger.info(f"AUC:       {metrics.auc:.3f}")
        logger.info(f"Train samples: {metrics.samples_train}, Test samples: {metrics.samples_test}")
        logger.info(f"Outliers removed: {metrics.outliers_removed}")

        logger.info("\nConfusion Matrix:")
        logger.info(f"  TP={metrics.confusion['tp']}, FP={metrics.confusion['fp']}")
        logger.info(f"  FN={metrics.confusion['fn']}, TN={metrics.confusion['tn']}")

        logger.info("\nTop 10 Feature Importances:")
        importances = list(zip(self.feature_names, self.model.feature_importances_))
        importances.sort(key=lambda x: x[1], reverse=True)
        for name, importance in importances[:10]:
            logger.info(f"  {name:20} {importance:.4f}")

        return metrics

    def save_model(self, version: str = "v2_phase2") -> Path:
        """Save trained model."""
        if self.model is None:
            raise ValueError("Model not trained yet")

        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)

        model_path = self.MODEL_DIR / f"xgboost_{version}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "feature_names": self.feature_names,
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "version": "phase2_engineered_features",
            }, f)

        logger.info(f"Model saved: {model_path}")
        return model_path

    def load_model(self, version: str = "v2_phase2") -> None:
        """Load pre-trained model."""
        model_path = self.MODEL_DIR / f"xgboost_{version}.pkl"

        with open(model_path, "rb") as f:
            data = pickle.load(f)

        self.model = data["model"]
        self.scaler = data["scaler"]
        self.feature_names = data["feature_names"]

        logger.info(f"Model loaded: {model_path}")


def main():
    """Train Phase 2 XGBoost with feature engineering."""
    db_path = Path(__file__).parent.parent.parent / "data" / "radar.db"

    logger.info("=" * 80)
    logger.info("PHASE 2: XGBOOST TRAINING WITH FEATURE ENGINEERING")
    logger.info("=" * 80)
    logger.info("Improvements:")
    logger.info("  • Engagement ratios (retweet/like, engagement rate)")
    logger.info("  • Account age in days")
    logger.info("  • Tweet recency in hours")
    logger.info("  • Outlier handling (remove ±10% price changes)")
    logger.info("  • Better profit threshold (±2% instead of 0%)")
    logger.info("=" * 80)

    trainer = XGBoostTrainerV2(str(db_path), window_hours=24)
    metrics = trainer.train(test_size=0.2)

    model_path = trainer.save_model("v2_phase2")

    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2 TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Model saved: {model_path}")


if __name__ == "__main__":
    main()
