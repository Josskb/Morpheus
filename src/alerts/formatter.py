"""
alerts/formatter.py
────────────────────
Formate un tweet NLP en message Telegram HTML.
"""

from __future__ import annotations

import json

from src.core.database import Account, Tweet

_CALL_EMOJI = {"long": "📈", "short": "📉"}
_CONF_LABEL = {(0.8, 1.0): "FORT", (0.65, 0.8): "MOYEN", (0.0, 0.65): "FAIBLE"}


def _conf_label(conf: float) -> str:
    for (lo, hi), label in _CONF_LABEL.items():
        if lo <= conf <= hi:
            return label
    return "?"


def _price_fmt(price: float) -> str:
    if price >= 1_000:
        return f"${price:,.0f}"
    if price >= 1:
        return f"${price:.2f}"
    return f"${price:.4f}"


def format_alert(tweet: Tweet, account: Account) -> str:
    """
    Retourne un message HTML prêt à envoyer via Telegram.
    """
    tickers: list[str] = json.loads(tweet.tickers or "[]")
    ticker_str = " · ".join(f"<b>${t}</b>" for t in tickers) if tickers else "—"
    emoji = _CALL_EMOJI.get(tweet.call_type or "", "📊")
    call_upper = (tweet.call_type or "signal").upper()
    conf = tweet.confidence or 0.0
    conf_lbl = _conf_label(conf)

    # Résumé du tweet (max 200 chars)
    preview = tweet.text.strip()
    if len(preview) > 200:
        preview = preview[:197] + "…"

    username = account.username
    tweet_url = f"https://twitter.com/{username}/status/{tweet.tweet_id}"

    lines: list[str] = [
        f"{emoji} <b>{call_upper} — {ticker_str}</b>  |  conf <code>{conf:.2f}</code> [{conf_lbl}]",
        f'👤 <a href="https://twitter.com/{username}">@{username}</a>',
        "",
        f'<i>"{preview}"</i>',
        "",
    ]

    # Prix optionnels
    details: list[str] = []
    if tweet.target_price:
        details.append(f"💰 Cible : <b>{_price_fmt(tweet.target_price)}</b>")
    if tweet.stop_loss:
        details.append(f"🛑 Stop  : <b>{_price_fmt(tweet.stop_loss)}</b>")
    if tweet.sentiment is not None:
        sign = "+" if tweet.sentiment >= 0 else ""
        details.append(f"📊 Sentiment <code>{sign}{tweet.sentiment:.2f}</code>")
    if tweet.urgency_score and tweet.urgency_score > 0:
        details.append(f"⚡ Urgence <code>{tweet.urgency_score:.2f}</code>")

    if details:
        lines.extend(details)
        lines.append("")

    lines.append(f'🔗 <a href="{tweet_url}">Voir le tweet</a>')

    return "\n".join(lines)
