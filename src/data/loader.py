"""Schema-aware loaders for the extracted Anthelion datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


def snake_case(name: str) -> str:
    return (
        name.strip()
        .lower()
        .replace("&", "and")
        .replace("'", "")
        .replace("/", "_")
        .replace("-", "_")
        .replace(" ", "_")
        .replace(",", "")
        .replace(".", "")
    )


def clean_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.loc[:, ~frame.columns.astype(str).str.startswith("Unnamed:")].copy()
    frame.columns = [snake_case(str(column)) for column in frame.columns]
    return frame


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    compression = "zip" if path.suffix == ".zip" else None
    return pd.read_csv(path, compression=compression, **kwargs)


@dataclass
class RawDatasets:
    prices: pd.DataFrame
    fundamentals: pd.DataFrame
    news: pd.DataFrame
    securities: pd.DataFrame


class DataCatalog:
    """Loads active research sources while preserving source and availability metadata."""

    def __init__(self, root: str | Path, data_config: dict) -> None:
        self.root = Path(root)
        self.config = data_config

    def _path(self, filename: str) -> Path:
        path = self.root / filename
        if not path.exists():
            raise FileNotFoundError(f"Configured dataset does not exist: {path}")
        return path

    def load_prices(self) -> pd.DataFrame:
        path = self._path(self.config["prices_file"])
        frame = clean_columns(_read_csv(path))
        required = {"date", "symbol", "open", "close", "low", "high", "volume"}
        missing = required - set(frame)
        if missing:
            raise ValueError(f"Price file missing columns: {sorted(missing)}")
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        frame["symbol"] = frame["symbol"].astype("string").str.upper().str.strip()
        frame = frame.dropna(subset=["date", "symbol", "close"])
        if frame.duplicated(["symbol", "date"]).any():
            raise ValueError("Price data contain duplicate symbol-date keys.")
        return frame.sort_values(["symbol", "date"]).reset_index(drop=True)

    def load_fundamentals(self) -> pd.DataFrame:
        path = self._path(self.config["fundamentals_file"])
        frame = clean_columns(_read_csv(path))
        frame = frame.rename(columns={"ticker_symbol": "symbol"})
        required = {"symbol", "period_ending"}
        missing = required - set(frame)
        if missing:
            raise ValueError(f"Fundamental file missing columns: {sorted(missing)}")
        frame["symbol"] = frame["symbol"].astype("string").str.upper().str.strip()
        frame["period_ending"] = pd.to_datetime(frame["period_ending"], errors="coerce")
        lag = pd.to_timedelta(
            int(self.config["fundamental_reporting_lag_days"]), unit="D"
        )
        frame["fundamental_available_date"] = frame["period_ending"] + lag
        return frame.dropna(subset=["symbol", "period_ending"]).sort_values(
            ["symbol", "fundamental_available_date"]
        )

    def load_securities(self) -> pd.DataFrame:
        path = self._path(self.config["securities_file"])
        frame = clean_columns(_read_csv(path))
        frame = frame.rename(columns={"ticker_symbol": "symbol"})
        if "symbol" not in frame:
            raise ValueError("Security master must contain a ticker symbol column.")
        frame["symbol"] = frame["symbol"].astype("string").str.upper().str.strip()
        return frame.drop_duplicates("symbol", keep="last")

    def _load_news_file(
        self,
        filename: str,
        *,
        symbols: set[str] | None,
        start: pd.Timestamp | None,
        end: pd.Timestamp | None,
    ) -> list[pd.DataFrame]:
        path = self._path(filename)
        compression = "zip" if path.suffix == ".zip" else None
        chunks: list[pd.DataFrame] = []
        for raw in pd.read_csv(
            path,
            compression=compression,
            chunksize=int(self.config.get("news_chunksize", 250_000)),
            low_memory=False,
        ):
            frame = clean_columns(raw)
            if "title" in frame and "headline" not in frame:
                frame = frame.rename(columns={"title": "headline"})
            required = {"stock", "date", "headline"}
            missing = required - set(frame)
            if missing:
                raise ValueError(
                    f"News file {filename} missing columns: {sorted(missing)}"
                )
            frame["symbol"] = frame["stock"].astype("string").str.upper().str.strip()
            if symbols is not None:
                frame = frame.loc[frame["symbol"].isin(symbols)]
            if frame.empty:
                continue
            frame["published_at"] = pd.to_datetime(
                frame["date"], errors="coerce", utc=True
            )
            frame = frame.dropna(subset=["symbol", "published_at", "headline"])
            if start is not None:
                frame = frame.loc[frame["published_at"] >= start]
            if end is not None:
                frame = frame.loc[frame["published_at"] < end]
            if frame.empty:
                continue
            if "publisher" not in frame:
                frame["publisher"] = "unknown"
            frame["source_file"] = filename
            chunks.append(
                frame[
                    ["symbol", "published_at", "headline", "publisher", "source_file"]
                ]
            )
        return chunks

    def load_news(
        self,
        *,
        symbols: Iterable[str] | None = None,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        symbol_set = set(symbols) if symbols is not None else None
        start_ts = pd.Timestamp(start, tz="UTC") if start is not None else None
        end_ts = pd.Timestamp(end, tz="UTC") if end is not None else None
        chunks: list[pd.DataFrame] = []
        for filename in self.config["news_files"]:
            chunks.extend(
                self._load_news_file(
                    filename,
                    symbols=symbol_set,
                    start=start_ts,
                    end=end_ts,
                )
            )
        if not chunks:
            return pd.DataFrame(
                columns=[
                    "symbol",
                    "published_at",
                    "headline",
                    "publisher",
                    "source_file",
                ]
            )
        frame = pd.concat(chunks, ignore_index=True)
        frame["headline_key"] = (
            frame["headline"]
            .astype("string")
            .str.lower()
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        frame = frame.drop_duplicates(
            ["symbol", "published_at", "headline_key"], keep="first"
        ).drop(columns="headline_key")
        lag = pd.to_timedelta(int(self.config["news_availability_lag_days"]), unit="D")
        frame["news_available_date"] = (
            frame["published_at"].dt.tz_convert(None).dt.normalize() + lag
        )
        return frame.sort_values(["symbol", "published_at"]).reset_index(drop=True)

    def load_all(self, *, start: str, end: str) -> RawDatasets:
        prices = self.load_prices()
        securities = self.load_securities()
        symbols = set(prices["symbol"].unique())
        return RawDatasets(
            prices=prices,
            fundamentals=self.load_fundamentals(),
            news=self.load_news(symbols=symbols, start=start, end=end),
            securities=securities,
        )
