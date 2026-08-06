"""
collector/apewisdom_client.py
────────────────────────────────
Client pour l'API publique ApeWisdom (gratuite, sans clé) : classement des
tickers les plus mentionnés sur Reddit + 4chan/biz, avec comparaison à 24h.

Contrairement à StockTwits, ce n'est PAS un flux de messages individuels —
c'est un signal agrégé d'attention/momentum par ticker. Ne produit donc pas
de RawTweet et n'entre pas dans le pipeline NLP habituel ; voir
trending_service.py pour comment ce signal est exploité (alerte directe sur
spike de mentions, pas de scoring NLP).
"""

from __future__ import annotations

from typing import Literal

import httpx
from loguru import logger

BASE_URL = "https://apewisdom.io/api/v1.0/filter"

ApeWisdomFilter = Literal["all-crypto", "all-stocks"]


class ApeWisdomClient:
    """Client pour le classement de tendance ApeWisdom (Reddit + 4chan)."""

    async def get_trending(
        self, filter: ApeWisdomFilter, max_results: int = 100
    ) -> list[dict]:
        """
        Retourne le classement (page 1, jusqu'à `max_results` entrées) pour
        le filtre donné, normalisé en :
        {ticker, name, mentions, mentions_24h_ago, rank, rank_24h_ago}
        """
        url = f"{BASE_URL}/{filter}"

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(url)
                resp.raise_for_status()
        except Exception as e:
            logger.warning("Erreur fetch ApeWisdom {} : {}", filter, e)
            return []

        data = resp.json()
        results = data.get("results", [])[:max_results]

        trending = [
            {
                "ticker": r["ticker"],
                "name": r.get("name", r["ticker"]),
                # `.get(key, 0)` ne suffit pas : l'API renvoie parfois
                # explicitement `null` (nouveau ticker sans historique 24h),
                # pas seulement une clé absente.
                "mentions": r.get("mentions") or 0,
                "mentions_24h_ago": r.get("mentions_24h_ago") or 0,
                "rank": r.get("rank"),
                "rank_24h_ago": r.get("rank_24h_ago"),
            }
            for r in results
        ]

        logger.debug("ApeWisdom {} : {} tickers récupérés", filter, len(trending))
        return trending
