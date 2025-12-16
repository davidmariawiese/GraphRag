from haystack import component
from haystack.dataclasses import Document
import json
import os
import math
import statistics
from datetime import datetime, timedelta
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

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
        # Debug output
        if os.getenv("GRAPH_RAG_DEBUG") == "true":
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
            "analysis": analysis_results
        }

        return {"analysis_json": final_output}

    def _parse_documents(self, documents) -> List[Dict[str, Any]]:
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

            sensor_id = (
                sensor.get("mote_id")
                or sensor.get("moteid")
                or meas.get("mote_id")
                or meas.get("moteid")
                or "Unknown"
            )
            
            # Extract Timestamp
            # Try explicit time field, or construct from Date/Hour/Minute if available
            ts = self._extract_timestamp(mapped, meas)
            if not ts:
                # If no temporal info, synthesize a stable ordering timestamp
                ts = base_time + timedelta(seconds=idx)

            if sensor_id not in grouped:
                grouped[sensor_id] = {}

            # Extract Metrics
            # We look for known numeric fields
            numeric_fields = ["temperature", "humidity", "light", "voltage"]
            for field in numeric_fields:
                val = meas.get(field)
                if val is not None:
                    try:
                        f_val = float(val)
                        if field not in grouped[sensor_id]:
                            grouped[sensor_id][field] = []
                        grouped[sensor_id][field].append((ts, f_val))
                    except (ValueError, TypeError):
                        pass
                        
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
                "%Y-%m-%d %H:%M:%S.%f",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d"
            ]
            for fmt in formats:
                try:
                    return datetime.strptime(t_val, fmt)
                except ValueError:
                    continue
        return None

    def _analyze_series(self, series: List[Tuple[datetime, float]]) -> Dict[str, Any]:
        """
        Performs statistical analysis on a time series.
        """
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
            if abs(total_change) < (0.1 * std_val): # Threshold: change < 10% of std
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
            candidates = [15*60, 60*60, 6*60*60, 24*60*60]
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
                if bucket_idx > self.max_buckets + 10: # Safety break
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
            anomalies = anomalies[:5]

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
