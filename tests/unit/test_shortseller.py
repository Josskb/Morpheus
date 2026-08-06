"""
tests/unit/test_shortseller.py
─────────────────────────────────
Tests unitaires du collecteur short-sellers : extraction de liens,
logique premier-run/nouveaux-rapports, et le service qui envoie les
alertes Telegram immédiates.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.collector.shortseller_client import ShortSellerClient
from src.collector.shortseller_service import ShortSellerService

SAMPLE_HTML = """
<html><body>
  <a href="https://grizzlyreports.com/reports/acme-corp">ACME Corp — Accounting Irregularities Uncovered</a>
  <a href="https://grizzlyreports.com/reports/acme-corp">ACME Corp — Accounting Irregularities Uncovered</a>
  <a href="https://grizzlyreports.com/about">About us</a>
  <a href="https://grizzlyreports.com/reports/short">Too short</a>
  <a href="https://other-site.com/reports/foo">Unrelated site, long enough title here</a>
</body></html>
"""


class TestParseLinks:
    def test_extracts_matching_links_above_min_title_length(self):
        client = ShortSellerClient.__new__(ShortSellerClient)
        results = client._parse_links(SAMPLE_HTML, "https://grizzlyreports.com", "grizzlyreports.com/reports")
        urls = [r["url"] for r in results]
        assert "https://grizzlyreports.com/reports/acme-corp" in urls
        assert "https://grizzlyreports.com/about" not in urls  # ne match pas link_contains

    def test_deduplicates_urls(self):
        client = ShortSellerClient.__new__(ShortSellerClient)
        results = client._parse_links(SAMPLE_HTML, "https://grizzlyreports.com", "grizzlyreports.com/reports")
        urls = [r["url"] for r in results]
        assert urls.count("https://grizzlyreports.com/reports/acme-corp") == 1

    def test_ignores_short_titles(self):
        client = ShortSellerClient.__new__(ShortSellerClient)
        results = client._parse_links(SAMPLE_HTML, "https://grizzlyreports.com", "grizzlyreports.com/reports")
        urls = [r["url"] for r in results]
        assert "https://grizzlyreports.com/reports/short" not in urls

    def test_ignores_other_domains(self):
        client = ShortSellerClient.__new__(ShortSellerClient)
        results = client._parse_links(SAMPLE_HTML, "https://grizzlyreports.com", "grizzlyreports.com/reports")
        urls = [r["url"] for r in results]
        assert not any("other-site.com" in u for u in urls)


class TestGetNewReports:
    def _mock_response(self, html: str) -> MagicMock:
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.text = html
        return resp

    @pytest.mark.asyncio
    async def test_first_run_indexes_without_alerting(self, tmp_path):
        client = ShortSellerClient(seen_file=tmp_path / "seen.json")
        with patch("httpx.AsyncClient.get", new=AsyncMock(return_value=self._mock_response(SAMPLE_HTML))):
            new_reports = await client.get_new_reports(
                name="Grizzly Research",
                report_url="https://grizzlyreports.com",
                link_contains="grizzlyreports.com/reports",
            )
        assert new_reports == []
        assert (tmp_path / "seen.json").exists()

    @pytest.mark.asyncio
    async def test_second_run_returns_only_new(self, tmp_path):
        seen_file = tmp_path / "seen.json"
        client = ShortSellerClient(seen_file=seen_file)

        with patch("httpx.AsyncClient.get", new=AsyncMock(return_value=self._mock_response(SAMPLE_HTML))):
            await client.get_new_reports("Grizzly Research", "https://grizzlyreports.com", "grizzlyreports.com/reports")

        new_html = SAMPLE_HTML + '<a href="https://grizzlyreports.com/reports/new-target">New Target — Fresh Allegations Published</a>'
        with patch("httpx.AsyncClient.get", new=AsyncMock(return_value=self._mock_response(new_html))):
            new_reports = await client.get_new_reports("Grizzly Research", "https://grizzlyreports.com", "grizzlyreports.com/reports")

        urls = [r["url"] for r in new_reports]
        assert "https://grizzlyreports.com/reports/new-target" in urls
        assert "https://grizzlyreports.com/reports/acme-corp" not in urls

    @pytest.mark.asyncio
    async def test_network_error_returns_empty_list(self, tmp_path):
        client = ShortSellerClient(seen_file=tmp_path / "seen.json")
        with patch("httpx.AsyncClient.get", new=AsyncMock(side_effect=Exception("network error"))):
            new_reports = await client.get_new_reports(
                "Grizzly Research", "https://grizzlyreports.com", "grizzlyreports.com/reports",
            )
        assert new_reports == []


class TestShortSellerService:
    def _make_service(self, firms: list[dict]) -> ShortSellerService:
        mock_client = AsyncMock()
        mock_bot = AsyncMock()
        svc = ShortSellerService(client=mock_client, bot=mock_bot)
        svc._firms = firms
        return svc

    @pytest.mark.asyncio
    async def test_check_once_sends_alert_for_new_report(self):
        svc = self._make_service([
            {"name": "Grizzly Research", "report_url": "https://grizzlyreports.com",
             "link_contains": "grizzlyreports.com/reports", "emoji": "🐻"},
        ])
        svc._client.get_new_reports = AsyncMock(return_value=[
            {"url": "https://grizzlyreports.com/reports/acme", "title": "ACME Corp Report"},
        ])

        sent = await svc.check_once()

        assert sent == 1
        svc._bot.send_shortseller_alert.assert_called_once_with(
            firm="Grizzly Research", emoji="🐻",
            title="ACME Corp Report", url="https://grizzlyreports.com/reports/acme",
        )

    @pytest.mark.asyncio
    async def test_check_once_no_new_reports_sends_nothing(self):
        svc = self._make_service([
            {"name": "Grizzly Research", "report_url": "https://grizzlyreports.com",
             "link_contains": "grizzlyreports.com/reports"},
        ])
        svc._client.get_new_reports = AsyncMock(return_value=[])

        sent = await svc.check_once()

        assert sent == 0
        svc._bot.send_shortseller_alert.assert_not_called()

    @pytest.mark.asyncio
    async def test_check_once_continues_after_firm_error(self):
        svc = self._make_service([
            {"name": "Broken Firm", "report_url": "https://broken.example", "link_contains": "x"},
            {"name": "Grizzly Research", "report_url": "https://grizzlyreports.com",
             "link_contains": "grizzlyreports.com/reports"},
        ])

        async def _get_new_reports(name, report_url, link_contains):
            if name == "Broken Firm":
                raise Exception("boom")
            return [{"url": "https://grizzlyreports.com/reports/acme", "title": "ACME Corp Report"}]

        svc._client.get_new_reports = AsyncMock(side_effect=_get_new_reports)

        sent = await svc.check_once()

        assert sent == 1
