"""
collector/shortseller_client.py
──────────────────────────────────
Surveille les sites des short-sellers activistes et détecte les nouveaux
rapports publiés. Chaque nouveau rapport déclenche une alerte Telegram
immédiate (bypass du récap périodique).

Persistance : data/shortseller_seen.json — stocke les URLs déjà vues par
firme. Premier run : indexe les rapports existants sans alerter (évite
le spam initial).

Ne couvre que le chemin httpx simple (sites sans protection anti-bot,
ex: Grizzly Research, Muddy Waters). Le contournement Cloudflare via
navigateur headless tenté côté VM (Playwright + Xvfb) s'est avéré non
concluant ("indéblocable sans proxy résidentiel payant") et n'est pas
porté ici — un firm nécessitant ça n'est simplement pas configurable
pour l'instant.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from loguru import logger

from ..core.settings import DATA_DIR

DEFAULT_SEEN_FILE = DATA_DIR / "shortseller_seen.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


class ShortSellerClient:
    def __init__(self, seen_file: Path | None = None) -> None:
        self._seen_file = seen_file or DEFAULT_SEEN_FILE

    def _load_seen(self) -> dict[str, list[str]]:
        if self._seen_file.exists():
            return json.loads(self._seen_file.read_text())
        return {}

    def _save_seen(self, seen: dict[str, list[str]]) -> None:
        self._seen_file.parent.mkdir(parents=True, exist_ok=True)
        self._seen_file.write_text(json.dumps(seen, indent=2))

    def _parse_links(
        self, html: str, base: str, link_contains: str
    ) -> list[dict[str, str]]:
        """Extrait les liens matchant link_contains depuis du HTML brut."""
        soup = BeautifulSoup(html, "html.parser")
        seen_urls: set[str] = set()
        results: list[dict[str, str]] = []
        for a in soup.find_all("a", href=True):
            full_url = urljoin(base, a["href"])
            title = a.get_text(" ", strip=True)
            if link_contains in full_url and len(title) >= 15 and full_url not in seen_urls:
                seen_urls.add(full_url)
                results.append({"url": full_url, "title": title})
        return results

    async def _fetch_links(
        self, name: str, report_url: str, link_contains: str
    ) -> list[dict[str, str]]:
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=20) as client:
                resp = await client.get(report_url, headers=HEADERS)
                resp.raise_for_status()
        except Exception as e:
            logger.warning("ShortSeller {} : erreur fetch : {}", name, e)
            return []

        parsed = urlparse(report_url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        results = self._parse_links(resp.text, base, link_contains)
        logger.debug("ShortSeller {} : {} liens trouvés", name, len(results))
        return results

    async def get_new_reports(
        self, name: str, report_url: str, link_contains: str
    ) -> list[dict[str, str]]:
        """
        Retourne les rapports jamais vus pour ce firm.
        Premier appel : indexe sans alerter (évite le spam de l'historique).
        """
        seen = self._load_seen()
        already_seen = set(seen.get(name, []))
        is_first_run = not already_seen

        links = await self._fetch_links(name, report_url, link_contains)
        if not links:
            return []

        new_reports = [] if is_first_run else [
            r for r in links if r["url"] not in already_seen
        ]

        seen[name] = list(already_seen | {r["url"] for r in links})
        self._save_seen(seen)

        if is_first_run:
            logger.info(
                "ShortSeller {} : premier run — {} rapports indexés (pas d'alerte)",
                name, len(links),
            )
        elif new_reports:
            logger.info(
                "ShortSeller {} : {} nouveau(x) rapport(s) !", name, len(new_reports)
            )

        return new_reports
