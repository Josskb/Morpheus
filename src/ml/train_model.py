"""
Module 4: ML Model Training
Entraîne un modèle sur les données collectées pour prédire la profitabilité
des tweets (changement de prix 24h).
"""

import sqlite3
import pickle
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from loguru import logger


class PriceChangePredictor:
    """Prédit le changement de prix 24h après un tweet."""

    MODEL_DIR = Path(__file__).parent.parent.parent / "models"

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.model = None
        self.scaler = None
        self.feature_names = None

    def load_training_data(self) -> tuple[pd.DataFrame, np.ndarray]:
        """Charge les données d'entraînement."""
        logger.info(f"Chargement données depuis {self.db_path}")

        conn = sqlite3.connect(self.db_path)

        # Requête pour récupérer tweets + snapshots avec changement 24h
        query = """
        SELECT
            t.id,
            t.confidence,
            t.sentiment,
            t.urgency_score,
            t.like_count,
            t.retweet_count,
            t.reply_count,
            a.reliability_score,
            a.win_rate,
            ms.change_24h
        FROM tweets t
        LEFT JOIN accounts a ON t.account_id = a.id
        LEFT JOIN market_snapshots ms ON t.id = ms.tweet_id
        WHERE t.nlp_processed = 1
            AND t.confidence IS NOT NULL
            AND ms.change_24h IS NOT NULL
        """

        df = pd.read_sql_query(query, conn)
        conn.close()

        logger.info(f"Chargé {len(df)} tweets avec données complètes")

        # Target: changement de prix 24h (en %)
        y = df["change_24h"].values

        logger.info(f"Prix change - Min: {y.min():.2f}%, Max: {y.max():.2f}%, Mean: {y.mean():.2f}%")

        return df, y

    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prépare les features."""
        features_to_use = [
            "confidence",
            "sentiment",
            "urgency_score",
            "like_count",
            "retweet_count",
            "reply_count",
            "reliability_score",
            "win_rate",
        ]

        df_features = df[features_to_use].fillna(0)

        self.feature_names = features_to_use
        logger.info(f"Features: {self.feature_names}")

        return df_features.values

    def train(self) -> dict:
        """Entraîne le modèle XGBoost."""
        logger.info("Chargement données...")
        df, y = self.load_training_data()

        logger.info("Préparation features...")
        X = self.prepare_features(df)

        # Normaliser
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        logger.info("Train/test split (80/20)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        logger.info("Entraînement XGBoost...")
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )

        self.model.fit(X_train, y_train, verbose=False)

        logger.info("Évaluation...")
        y_pred = self.model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        metrics = {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        }

        logger.info(f"MAE:  {mae:.4f}%")
        logger.info(f"RMSE: {rmse:.4f}%")
        logger.info(f"R²:   {r2:.4f}")

        logger.info("Feature Importance:")
        for name, importance in zip(self.feature_names, self.model.feature_importances_):
            logger.info(f"  {name:20} {importance:.4f}")

        return metrics

    def save_model(self, version: str = "v1") -> Path:
        """Sauvegarde le modèle."""
        if self.model is None:
            raise ValueError("Model not trained")

        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)

        model_path = self.MODEL_DIR / f"xgboost_price_predictor_{version}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "feature_names": self.feature_names,
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            }, f)

        logger.info(f"Modèle sauvegardé: {model_path}")
        return model_path

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Prédit le changement de prix 24h."""
        if self.model is None:
            raise ValueError("Model not trained")

        X_scaled = self.scaler.transform(features)
        return self.model.predict(X_scaled)


def main():
    """Entraîne et sauvegarde le modèle."""
    db_path = Path(__file__).parent.parent.parent / "data" / "radar.db"

    logger.info("=" * 80)
    logger.info("MODULE 4: PRICE CHANGE PREDICTION MODEL")
    logger.info("=" * 80)

    predictor = PriceChangePredictor(str(db_path))
    metrics = predictor.train()

    model_path = predictor.save_model("v1")

    logger.info("\n" + "=" * 80)
    logger.info("RÉSULTATS")
    logger.info("=" * 80)
    logger.info(f"Mean Absolute Error: {metrics['mae']:.4f}%")
    logger.info(f"RMSE:                {metrics['rmse']:.4f}%")
    logger.info(f"R² Score:            {metrics['r2']:.4f}")
    logger.info(f"Training samples:    {metrics['train_samples']}")
    logger.info(f"Test samples:        {metrics['test_samples']}")
    logger.info(f"\nModèle sauvegardé: {model_path}")


if __name__ == "__main__":
    main()
