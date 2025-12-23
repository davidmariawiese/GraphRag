"""Deterministic aggregation/structuring of Neo4j retrieval results.

`Organizer` consumes a list of Haystack `Document`s. Each document is expected to
contain JSON (as produced by `CypherRetriever`). The output is a structured JSON
object with:

- `raw`: capped list of raw records for list-style questions
- `analysis`: per-sensor/per-metric statistics, trends and bucketed aggregates
- `context`: provenance metadata

Important:
- This module expects `Document.content` to contain JSON-serializable data
    (as produced by a well-behaved Retriever). The retriever normalizes Neo4j
    driver types into plain dicts so the Organizer can rely on JSON parsing.
"""

import json
import logging
import os
import statistics
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from haystack import component
from haystack.dataclasses import Document

from ..config import GRAPH_RAG_DEBUG_ENV


logger = logging.getLogger(__name__)


_DEBUG_JSON_PREVIEW_CHARS = 1000
_NUMERIC_FIELDS = ("temperature", "humidity", "light", "voltage")
_NON_METRIC_KEYS = {
    "moteid",
    "mote_id",
    "sensor_id",
    "epoch_id",
    "epoch",
    "date",
    "week",
    "hour",
    "minute",
    "time",
    "ts",
}
_BUCKET_CANDIDATES_SECONDS = (15 * 60, 60 * 60, 6 * 60 * 60, 24 * 60 * 60)
_TREND_FLAT_STD_FRACTION = 0.1
_MAX_ANOMALIES = 5
_BUCKET_SAFETY_OVERFLOW = 10
_NODE_KEYS = {"Sensor", "Measurement", "Epoch", "Date", "Week", "Hour", "Minute"}


def _debug_enabled() -> bool:
    return os.getenv(GRAPH_RAG_DEBUG_ENV) == "true"


def _coerce_numeric(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        if isinstance(value, bool):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _detect_aggregate_payload(raw_data: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Detect the common aggregate return shape.

    Aggregate Cypher queries are expected to return a single row with a single
    numeric field (usually `value`). In this case, downstream components should
    use that numeric `value` as the answer, and not confuse it with
    `stats_global.count` (which would be 1 because there's only one row).
    """
    # Controlled pipeline: expect a list of rows; only a single-row list can
    # represent an aggregate payload. Fail-fast if not exactly one row.
    if len(raw_data) != 1:
        return None
    row = raw_data[0]

    # Prefer the canonical alias used by QueryProcessor: AS value
    if "value" in row:
        num = _coerce_numeric(row.get("value"))
        if num is None:
            return None
        # If the record only contains the aggregate field, treat as aggregate.
        if len(row.keys()) == 1:
            return {"field": "value", "value": num, "rows": 1}

    # Fallback: if there is exactly one numeric scalar field, treat it as aggregate.
    numeric_fields = []
    for k, v in row.items():
        num = _coerce_numeric(v)
        if num is None:
            continue
        numeric_fields.append((k, num))
    if len(numeric_fields) == 1:
        k, num = numeric_fields[0]
        return {"field": k, "value": num, "rows": 1}

    return None

@component
class Organizer:
    """
    Analysiert Messwerte deterministisch und liefert ein strukturiertes JSON-Objekt (analysis_json).
    Keine Freitext-Zusammenfassung.
    """

    def __init__(self, max_buckets: int = 200, anomaly_threshold: float = 2.0):
        self.max_buckets = max_buckets
        self.anomaly_threshold = anomaly_threshold
        
        # Mapping für Cypher-Aliase
        self.alias_map = {
            "s": "Sensor",
            "m": "Measurement",
            "e": "Epoch",
            "d": "Date",
            "w": "Week",
            "h": "Hour",
            "mi": "Minute"
        }

    @component.output_types(analysis_json=Dict[str, Any])
    def run(self, documents: List[Document]):
        """Parse retrieved documents and optionally compute deterministic analysis.

        The retriever returns Neo4j rows serialized as JSON strings.

        Output contract:
        - analysis_json.context: metadata + optional aggregate hint
        - analysis_json.raw: raw records (capped)
        - analysis_json.analysis: deterministic statistics (only in "full" mode)

        This component is intentionally deterministic (no LLM calls): it prepares
        a structured JSON context for the downstream Generator (LLM).
        """

        debug = _debug_enabled()

        if debug:
            print("\n[organizer] input.documents:")
            total = len(documents)
            print(f"documents_total={total}")

        # 1) Parse & flatten JSON rows.
        raw_data = self._parse_documents(documents)
        if not raw_data:
            return {"analysis_json": {"error": "No data found", "context": {}}}

        # In the controlled pipeline we expect `raw_data` to be the flat list of rows
        # produced by the Retriever (no wrapper objects).

        aggregate = _detect_aggregate_payload(raw_data)

        # 2) Group by sensor + metric.
        # Structure: { sensor_id: { metric_name: [(timestamp, value), ...] } }
        grouped_data = self._group_data(raw_data)

        # 3) Deterministic analysis for each time series.
        analysis_results: Dict[str, Dict[str, Any]] = {}

        for sensor_id, metrics in grouped_data.items():
            analysis_results[sensor_id] = {}
            for metric, series in metrics.items():
                if not series:
                    continue
                series.sort(key=lambda x: x[0])
                analysis_results[sensor_id][metric] = self._analyze_series(series)

        # 4) Construct final JSON
        final_output = {
            "context": {
                "entity_count": len(analysis_results),
                "entities": list(analysis_results.keys()),
                "provenance": "Cypher Query Result",
                # Optional hint for downstream rendering when the Cypher result is an aggregate.
                "aggregate": aggregate,
            },
            "analysis": analysis_results,
            # Keep raw records to answer list-style questions; capped to avoid blow-up.
            "raw": {
                "total_records": len(raw_data),
                "records": raw_data[: self.max_buckets],
            },
        }

        if debug:
            serialized = json.dumps(final_output, ensure_ascii=False)
            preview = serialized[:_DEBUG_JSON_PREVIEW_CHARS]
            suffix = "..." if len(serialized) > _DEBUG_JSON_PREVIEW_CHARS else ""
            print("[organizer] output.analysis_json preview (len={}):".format(len(serialized)))
            print(preview + suffix)

        return {"analysis_json": final_output}

    def _parse_documents(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Parse `Document.content` into Python objects.
        This implementation enforces the Retriever contract: `Document.content`
        must be a JSON string produced by the `CypherRetriever` (i.e. rows
        serialized with `json.dumps`). Any non-string or invalid JSON will
        raise an error. This keeps the Organizer strictly dependent on the
        pipeline and avoids accepting inputs from ad-hoc callers.
        """

        results: List[Dict[str, Any]] = []
        for doc in documents:
            content = getattr(doc, "content", None)
            if not isinstance(content, str):
                raise TypeError("Document.content must be a JSON string from CypherRetriever")

            # json.loads will raise on invalid JSON — fail fast in controlled env
            data = json.loads(content)

            if isinstance(data, list):
                results.extend(data)
            elif isinstance(data, dict):
                results.append(data)
            else:
                raise ValueError("Document content must be a JSON object or array")

        return results

    # NOTE: We assume the retriever provides plain dicts for nested nodes.
    # In this controlled environment we prefer direct `.get(..., {})` calls
    # in `_group_data` instead of keeping a defensive `_as_dict` helper.

    # `_merge_measurement_fields` removed: Retriever guarantees flat `Measurement` dicts
    # and any necessary normalization should be done there. This keeps the
    # Organizer small and focused on analysis only.

    def _extract_sensor_id(self, mapped: Dict[str, Any], sensor: Dict[str, Any], meas: Dict[str, Any]) -> str:
        """Best-effort extraction of the sensor identifier.

        We accept multiple keys because different Cypher queries / datasets use
        different naming conventions (mote_id vs moteid, nested node vs flat).
        """
        sensor_id_val = (
            mapped.get("mote_id")
            or mapped.get("moteid")
            or sensor.get("mote_id")
            or sensor.get("moteid")
            or meas.get("mote_id")
            or meas.get("moteid")
        )
        return str(sensor_id_val) if sensor_id_val is not None else "Unknown"

    def _extract_metric_values(self, meas: Dict[str, Any], mapped: Dict[str, Any]) -> Dict[str, float]:
        """Extract canonical metric values from `meas`.

        With the stricter Retriever contract we prefer explicit canonical fields
        (`temperature`, `humidity`, `light`, `voltage`). If none are present,
        we return an empty dict (do not attempt broad heuristics).
        """

        metrics: Dict[str, float] = {}
        for field in _NUMERIC_FIELDS:
            val = meas.get(field)
            if val is None:
                continue
            try:
                metrics[field] = float(val)
            except (ValueError, TypeError):
                continue
        return metrics

    def _group_data(self, data: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[Tuple[datetime, float]]]]:
        """Group flat Neo4j rows into time series per sensor and metric.

        Heuristics:
        - Accept multiple return shapes: nested nodes (Sensor/Measurement) or flat
          aliased fields (e.g., mote_id, time, temperature).
        - Timestamp fallback: if no time can be extracted, uses a synthetic time
          sequence to keep downstream analysis stable.

                Output shape:
                    { "<sensor_id>": { "<metric>": [(datetime, float), ...], ... }, ... }
        """

        grouped: Dict[str, Dict[str, List[Tuple[datetime, float]]]] = {}
        base_time = datetime.utcnow()

        for idx, record in enumerate(data):
            # Retriever provides normalized rows: use record directly.
            mapped = record

            sensor = mapped.get("Sensor", {})
            meas = mapped.get("Measurement", {})

            sensor_id = self._extract_sensor_id(mapped, sensor, meas)

            ts = self._extract_timestamp(mapped, meas)
            if not ts:
                ts = base_time + timedelta(seconds=idx)

            metric_values = self._extract_metric_values(meas, mapped)
            if not metric_values:
                continue

            sensor_bucket = grouped.setdefault(sensor_id, {})
            for metric, f_val in metric_values.items():
                sensor_bucket.setdefault(metric, []).append((ts, f_val))

        return grouped

    # Alias mapping removed: Retriever produces canonical `Measurement`/`Sensor`
    # nested dicts, so Organizer no longer needs to rewrite record keys.

    def _extract_timestamp(self, record: Dict[str, Any], meas: Dict[str, Any]) -> Optional[datetime]:
        """Extract a best-effort timestamp from a row.

        Preferred order:
        1) Measurement.time
        2) top-level time/ts alias
        3) Date/Hour/Minute nodes (reconstructed)

        Returns a timezone-naive datetime in UTC/local parsing semantics
        (no conversion is done; values are treated as provided by the dataset).
        """
        # 1. Try direct 'time' field in Measurement
        raw_time = meas.get("time")
        if raw_time:
            return self._parse_time(raw_time)

        # 1b. Common flat return shape: time/ts at the top-level
        raw_time = record.get("time") or record.get("ts")
        if raw_time:
            return self._parse_time(raw_time)
            
        # 2. Try Date/Hour/Minute nodes
        date_node = record.get("Date", {})
        hour_node = record.get("Hour", {})
        min_node = record.get("Minute", {})
        
        d_str = date_node.get("date") # "YYYY-MM-DD"
        h_val = hour_node.get("hour")
        m_val = min_node.get("minute")
        
        if d_str:
            dt = datetime.strptime(d_str, "%Y-%m-%d")
            if h_val is not None:
                dt = dt.replace(hour=int(h_val))
            if m_val is not None:
                dt = dt.replace(minute=int(m_val))
            return dt
                
        return None

    def _parse_time(self, t_val: Any) -> Optional[datetime]:
        """Parse timestamps from multiple formats seen in the dataset.

        This is intentionally permissive: Neo4j stores time as strings, and
        different dataset exports include either ISO timestamps or split date/time.
        """
        # Controlled pipeline: timestamps are provided as ISO or common strings.
        if not isinstance(t_val, str):
            return None

        # Strict ISO parser for controlled inputs
        return datetime.fromisoformat(t_val)

    def _analyze_series(self, series: List[Tuple[datetime, float]]) -> Dict[str, Any]:
        """Perform deterministic analysis on a time series.

        Output fields are consumed by the Generator prompt.

        The analysis is purely numerical (no LLM): global stats, trend via
        linear regression, windowed bucket aggregates, and simple anomaly flags.
        """
        values = [v for _, v in series]
        timestamps = [t for t, _ in series]
        
        # Global Stats
        count = len(values)
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values) if count > 1 else 0.0
            
        min_val = min(values)
        max_val = max(values)
        
        # Find timestamps for min/max
        min_ts = next(t for t, v in series if v == min_val)
        max_ts = next(t for t, v in series if v == max_val)
        
        # Quantiles
        p50 = np.percentile(values, 50)
        p95 = np.percentile(values, 95)
        p05 = np.percentile(values, 5)

        # Trend (Linear Regression)
        # Convert timestamps to seconds from start
        t_start = timestamps[0]
        t_end = timestamps[-1]
        duration_seconds = (t_end - t_start).total_seconds()
        
        trend_info = {"direction": "flat", "slope": 0.0}
        
        if count > 1 and duration_seconds > 0:
            x = np.array([(t - t_start).total_seconds() for t in timestamps])
            y = np.array(values)
            # Polyfit deg=1 -> slope, intercept
            slope, _ = np.polyfit(x, y, 1)
            
            # Normalize slope to "per hour"
            slope_per_hour = slope * 3600
            
            # Simple heuristic for direction
            # If change over total duration is significant compared to std dev
            total_change = slope * duration_seconds
            if abs(total_change) < (_TREND_FLAT_STD_FRACTION * std_val): # Threshold: change < 10% of std
                direction = "flat"
            else:
                direction = "up" if slope > 0 else "down"
                
            trend_info = {
                "direction": direction,
                "slope_per_hour": slope_per_hour,
                "delta_start_end": total_change
            }

        # Windowed Aggregates (Bucketing)
        buckets = []
        if duration_seconds > 0:
            # Determine bucket size
            # Try 15min, 1h, 6h, 1d
            candidates = _BUCKET_CANDIDATES_SECONDS
            bucket_size = candidates[0]
            
            for size in candidates:
                if (duration_seconds / size) <= self.max_buckets:
                    bucket_size = size
                    break
            else:
                # If still too many, calculate exact size to fit max_buckets
                bucket_size = duration_seconds / self.max_buckets

            # Create buckets
            current_t = t_start
            bucket_idx = 0
            
            while current_t <= t_end:
                next_t = current_t + timedelta(seconds=bucket_size)
                
                # Filter points in this bucket
                # (t >= current_t) and (t < next_t)
                b_values = [v for t, v in series if current_t <= t < next_t]
                
                # Include the last point in the last bucket if needed
                if next_t > t_end:
                     b_values.extend([v for t, v in series if t == t_end and t >= current_t])

                if b_values:
                    b_mean = statistics.mean(b_values)
                    b_std = statistics.stdev(b_values) if len(b_values) > 1 else 0.0
                    buckets.append({
                        "t_start": current_t.isoformat(),
                        "count": len(b_values),
                        "mean": b_mean,
                        "min": min(b_values),
                        "max": max(b_values),
                        "std": b_std
                    })
                
                current_t = next_t
                bucket_idx += 1
                if bucket_idx > self.max_buckets + _BUCKET_SAFETY_OVERFLOW: # Safety break
                    break

        # Anomalies (Highlight Top-N buckets by z-score of their mean)
        anomalies = []
        if buckets and std_val > 0:
            # Calculate z-score of bucket means relative to global stats
            for b in buckets:
                z_score = (b["mean"] - mean_val) / std_val
                if abs(z_score) > self.anomaly_threshold:
                    anomalies.append({
                        "t_start": b["t_start"],
                        "score": z_score,
                        "mean": b["mean"]
                    })
            # Sort by absolute score descending and take top 5
            anomalies.sort(key=lambda x: abs(x["score"]), reverse=True)
            anomalies = anomalies[:_MAX_ANOMALIES]

        return {
            "stats_global": {
                "count": count,
                "mean": mean_val,
                "std": std_val,
                "min": {"value": min_val, "timestamp": min_ts.isoformat()},
                "max": {"value": max_val, "timestamp": max_ts.isoformat()},
                "quantiles": {"p05": p05, "p50": p50, "p95": p95}
            },
            "trend": trend_info,
            "windowed_aggregates": {
                "bucket_size_seconds": bucket_size if duration_seconds > 0 else 0,
                "buckets": buckets
            },
            "highlights": {
                "anomalies": anomalies
            }
        }
