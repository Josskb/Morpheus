from .features import FEATURE_NAMES, build_features, get_primary_snapshot, primary_ticker, to_vector
from .scorer import HEURISTIC_VERSION, MLScorer, ScoreResult


# MLScoringService/OutcomeLabeler/ModelTrainer dépendent de loguru/sqlalchemy
# (et pour le trainer, scikit-learn/xgboost) — import lazy pour éviter de
# casser les tests unitaires qui n'ont besoin que des features/scorer.
def __getattr__(name: str):
    if name == "MLScoringService":
        from .service import MLScoringService
        return MLScoringService
    if name == "OutcomeLabeler":
        from .labeler import OutcomeLabeler
        return OutcomeLabeler
    if name == "ModelTrainer":
        from .trainer import ModelTrainer
        return ModelTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "FEATURE_NAMES", "build_features", "get_primary_snapshot", "primary_ticker", "to_vector",
    "HEURISTIC_VERSION", "MLScorer", "ScoreResult",
    "MLScoringService", "OutcomeLabeler", "ModelTrainer",
]
