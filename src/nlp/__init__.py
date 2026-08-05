from .analyzer import NLPResult, TweetAnalyzer


# NLPProcessorService dépend de loguru/sqlalchemy — import lazy pour éviter
# de casser les tests unitaires qui n'ont pas ces dépendances installées.
def __getattr__(name: str):
    if name == "NLPProcessorService":
        from .processor import NLPProcessorService
        return NLPProcessorService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["TweetAnalyzer", "NLPResult", "NLPProcessorService"]
