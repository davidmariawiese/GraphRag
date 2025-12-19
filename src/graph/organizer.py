"""Deterministic aggregation/structuring of Neo4j retrieval results.

`Organizer` consumes a list of Haystack `Document`s. Each document is expected to
contain JSON (as produced by `CypherRetriever`). The output is a structured JSON
object with:

- `raw`: capped list of raw records for list-style questions
- `analysis`: per-sensor/per-metric statistics, trends and bucketed aggregates
- `context`: provenance metadata

Important:
- `_parse_documents` contains an `eval()` fallback for non-JSON strings. This is
    *not safe* for untrusted inputs and is intentionally left unchanged to avoid
    behavior changes.
"""

import json
import os
import statistics
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from haystack import component
from haystack.dataclasses import Document

from ..config import GRAPH_RAG_DEBUG_ENV


_DEBUG_JSON_PREVIEW_CHARS = 1000
_NUMERIC_FIELDS = ("temperature", "humidity", "light", "voltage")
_BUCKET_CANDIDATES_SECONDS = (15 * 60, 60 * 60, 6 * 60 * 60, 24 * 60 * 60)
_TREND_FLAT_STD_FRACTION = 0.1
_MAX_ANOMALIES = 5
_BUCKET_SAFETY_OVERFLOW = 10

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
        debug = os.getenv(GRAPH_RAG_DEBUG_ENV) == "true"

        if debug:
            print("\n[organizer] input.documents:")
            try:
                total = len(documents) if isinstance(documents, list) else 0
            except Exception:
                total = 0
            print(f"documents_total={total}")

        # 1. Parse & Flatten
        raw_data = self._parse_documents(documents)
        if not raw_data:
            return {"analysis_json": {"error": "No data found", "context": {}}}

        # 2. Group by Sensor & Metric
        # Structure: { sensor_id: { metric_name: [(timestamp, value), ...] } }
        grouped_data = self._group_data(raw_data)

        # 3. Analyze each series
        analysis_results = {}
        
        for sensor_id, metrics in grouped_data.items():
            analysis_results[sensor_id] = {}
            for metric, series in metrics.items():
                if not series:
                    continue
                # Sort by timestamp
                series.sort(key=lambda x: x[0])
                
                analysis_results[sensor_id][metric] = self._analyze_series(series)

        # 4. Construct final JSON
        final_output = {
            "context": {
                "entity_count": len(analysis_results),
                "entities": list(analysis_results.keys()),
                "provenance": "Cypher Query Result"
            },
            "analysis": analysis_results,
            # Keep raw records to answer list-style questions; capped to avoid blow-up.
            "raw": {
                "total_records": len(raw_data),
                "records": raw_data[: self.max_buckets],
            },
        }

        if debug:
            try:
                serialized = json.dumps(final_output, ensure_ascii=False)
                preview = serialized[:_DEBUG_JSON_PREVIEW_CHARS]
                suffix = "..." if len(serialized) > _DEBUG_JSON_PREVIEW_CHARS else ""
                print("[organizer] output.analysis_json preview (len={}):".format(len(serialized)))
                print(preview + suffix)
            except Exception:
                print("[organizer] output.analysis_json: <unserializable>")

        return {"analysis_json": final_output}

    def _parse_documents(self, documents) -> List[Dict[str, Any]]:
        """Parse `Document.content` into Python objects.

        The expected format is JSON. For backwards compatibility, a non-JSON
        `eval()` fallback is retained; this must only be used with trusted input.
        """
        results = []
        for doc in documents:
            content = getattr(doc, "content", "")
            try:
                data = json.loads(content)
            except Exception:
                try:
                    data = eval(content)
                except Exception:
                    continue
            
            if isinstance(data, list):
                results.extend(data)
            elif isinstance(data, dict):
                results.append(data)
        return results

    def _group_data(self, data: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[Tuple[datetime, float]]]]:
        grouped = {}
        base_time = datetime.utcnow()
        
        for idx, record in enumerate(data):
            # Resolve aliases
            mapped = self._map_aliases(record)
            
            # Extract Sensor ID
            sensor = mapped.get("Sensor", {})
            meas = mapped.get("Measurement", {})
            
            # Fallback for flat structures (e.g. "m.temperature")
            flat_meas = {}
            for k, v in mapped.items():
                if "." in k:
                    prefix, suffix = k.split(".", 1)
                    if prefix == "Measurement" or prefix == "m":
                        flat_meas[suffix] = v
            
            # Merge flat measurements into meas dict
            meas.update(flat_meas)

            # Sensor ID
            # Common Cypher query shape returns `mote_id` at the top-level via an alias;
            # handle that first, then fall back to nested Sensor/Measurement props.
            sensor_id_val = (
                mapped.get("mote_id")
                or mapped.get("moteid")
                or sensor.get("mote_id")
                or sensor.get("moteid")
                or meas.get("mote_id")
                or meas.get("moteid")
            )
            sensor_id = str(sensor_id_val) if sensor_id_val is not None else "Unknown"
            
            # Extract Timestamp
            ts = self._extract_timestamp(mapped, meas)
            if not ts:
                ts = base_time + timedelta(seconds=idx)

            if sensor_id not in grouped:
                grouped[sensor_id] = {}

            # Extract Metrics
            added_metric = False
            for field in _NUMERIC_FIELDS:
                val = meas.get(field)
                if val is not None:
                    try:
                        f_val = float(val)
                        if field not in grouped[sensor_id]:
                            grouped[sensor_id][field] = []
                        grouped[sensor_id][field].append((ts, f_val))
                        added_metric = True
                    except (ValueError, TypeError):
                        pass

            # Fallback: take any numeric fields from meas
            if not added_metric:
                for k, v in meas.items():
                    if k in ["time", "date", "epoch", "moteid", "mote_id"]:
                        continue
                    try:
                        f_val = float(v)
                        if k not in grouped[sensor_id]:
                            grouped[sensor_id][k] = []
                        grouped[sensor_id][k].append((ts, f_val))
                        added_metric = True
                    except (ValueError, TypeError):
                        continue

            # Final fallback: take numeric scalars from the mapped record (e.g., aggregates like average_temperature)
            if not added_metric:
                for k, v in mapped.items():
                    if isinstance(v, (dict, list)):
                        continue
                    if k in ["Sensor", "Measurement", "Epoch", "Date", "Week", "Hour", "Minute"]:
                        continue
                    try:
                        f_val = float(v)
                        if k not in grouped[sensor_id]:
                            grouped[sensor_id][k] = []
                        grouped[sensor_id][k].append((ts, f_val))
                        added_metric = True
                    except (ValueError, TypeError):
                        continue
                        
        return grouped

    def _map_aliases(self, record: Dict[str, Any]) -> Dict[str, Any]:
        new_record = {}
        for key, value in record.items():
            # Handle dotted keys like "m.temperature"
            if isinstance(key, str) and "." in key:
                prefix, suffix = key.split(".", 1)
                mapped_prefix = self.alias_map.get(prefix, prefix)
                new_key = f"{mapped_prefix}.{suffix}"
                new_record[new_key] = value
            else:
                new_key = self.alias_map.get(key, key)
                new_record[new_key] = value
        return new_record

    def _extract_timestamp(self, record: Dict[str, Any], meas: Dict[str, Any]) -> Optional[datetime]:
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
            try:
                dt = datetime.strptime(d_str, "%Y-%m-%d")
                if h_val is not None:
                    dt = dt.replace(hour=int(h_val))
                if m_val is not None:
                    dt = dt.replace(minute=int(m_val))
                return dt
            except Exception:
                pass
                
        return None

    def _parse_time(self, t_val: Any) -> Optional[datetime]:
        if isinstance(t_val, datetime):
            return t_val
        if isinstance(t_val, str):
            # Try common formats
            formats = [
                "%Y-%m-%dT%H:%M:%S.%f",
                "%Y-%m-%d %H:%M:%S.%f",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d"
            ]
            # Fast path: ISO parser
            try:
                return datetime.fromisoformat(t_val)
            except Exception:
                pass
            for fmt in formats:
                try:
                    return datetime.strptime(t_val, fmt)
                except ValueError:
                    continue
        return None

    def _analyze_series(self, series: List[Tuple[datetime, float]]) -> Dict[str, Any]:
        """Perform statistical analysis on a time series."""
        values = [v for _, v in series]
        timestamps = [t for t, _ in series]
        
        # Global Stats
        count = len(values)
        mean_val = statistics.mean(values)
        try:
            std_val = statistics.stdev(values) if count > 1 else 0.0
        except Exception:
            std_val = 0.0
            
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
