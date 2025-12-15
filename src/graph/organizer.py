from haystack import component
import json
import os
from statistics import mean
from typing import List, Dict, Any

@component
class Organizer:
    """
    Verarbeitet die Rückgabe des Retrievers.
    - Mappt Cypher-Aliase (s, m, e...) auf volle Namen.
    - Erkennt Aggregationen (z.B. max_humidity).
    - Entscheidet basierend auf 'threshold_k' zwischen Detailansicht und Zusammenfassung.
    """

    def __init__(self, threshold_k: int = 10, float_precision: int | None = None):
        self.threshold_k = threshold_k
        self.float_precision = float_precision
        
        self.alias_map = {
            "s": "Sensor",
            "m": "Measurement",
            "e": "Epoch",
            "d": "Date",
            "w": "Week",
            "h": "Hour",
            "mi": "Minute"
        }

        # When Cypher returns scalar projections like `m.temperature AS m.temperature`
        # (or without an alias), these often appear as dotted keys.
        # We normalize common entity-prefixed keys into stable column names.
        self._flatten_key_prefixes = {
            "Measurement": {"temperature", "humidity", "light", "voltage", "mote_id", "moteid"},
            "Sensor": {"mote_id", "moteid"},
            "Date": {"date"},
            "Week": {"week"},
            "Hour": {"hour"},
            "Minute": {"minute"},
            "Epoch": {"epoch_id"},
        }

    @component.output_types(context=str)
    def run(self, documents: list):
        if os.getenv("GRAPH_RAG_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}:
            print("\n[organizer] input.documents:")
            try:
                total = len(documents) if isinstance(documents, list) else 0
            except Exception:
                total = 0
            print(f"documents_total={total}")
            if isinstance(documents, list) and documents:
                for i, d in enumerate(documents[:5], start=1):
                    content = getattr(d, "content", None)
                    if isinstance(content, str) and len(content) > 500:
                        content = content[:500].rstrip() + "..."
                    print(f"- doc[{i}] content: {content}")

        raw_data = self._parse_documents(documents)
        
        if not raw_data:
            return {"context": "Keine Daten gefunden."}

        mapped_data = [self._map_aliases(record) for record in raw_data]
        count = len(mapped_data)
        
        context_lines = []
        context_lines.append(f"### GRAPH CONTEXT ({count} Datensätze)")
        
        if count <= self.threshold_k:
            context_lines.append("Die folgenden Daten sind detaillierte Messwerte:")
            context_lines.append(self._create_markdown_table(mapped_data))
        else:
            context_lines.append("Aufgrund der großen Menge wurden die Daten statistisch zusammengefasst:")
            context_lines.append(self._create_summary_view(mapped_data))

        final_context = "\n".join(context_lines)
        # print("[ORGANIZER]:")
        # print(final_context)
        return {"context": final_context}

    def _parse_documents(self, documents) -> List[Dict[str, Any]]:
        results = []
        for doc in documents:
            content = doc.content
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

    def _map_aliases(self, record: Dict[str, Any]) -> Dict[str, Any]:
        new_record = {}
        for key, value in record.items():
            # Handle both entity aliases (e.g. `m`) and dotted projections (e.g. `m.temperature`).
            if isinstance(key, str) and "." in key:
                prefix, suffix = key.split(".", 1)
                mapped_prefix = self.alias_map.get(prefix, prefix)
                new_key = f"{mapped_prefix}.{suffix}"
            else:
                new_key = self.alias_map.get(key, key)
            new_record[new_key] = value
        return new_record

    def _format_scalar(self, value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, float):
            # Do not apply any fixed rounding. Use Python's shortest round-trip representation.
            if self.float_precision is None:
                return str(value)

            # Optional: allow rounding if explicitly configured.
            formatted = format(value, f".{self.float_precision}f")
            if "." in formatted:
                formatted = formatted.rstrip("0").rstrip(".")
            return formatted
        return str(value)

    def _normalize_column_key(self, key: str) -> str:
        if not isinstance(key, str) or "." not in key:
            return key
        prefix, suffix = key.split(".", 1)

        allowed_suffixes = self._flatten_key_prefixes.get(prefix)
        if allowed_suffixes and suffix in allowed_suffixes:
            # Prefer stable, user-friendly column names.
            if suffix in {"moteid", "mote_id"}:
                return "SensorID"
            return suffix
        return key

    def _create_markdown_table(self, data: List[Dict[str, Any]]) -> str:
        """
        Erstellt eine Markdown-Tabelle.
        Behandelt sowohl verschachtelte Entitäten (Measurement) als auch flache Aggregationen (max_humidity).
        """
        flattened_rows = []
        all_keys = set()

        for record in data:
            row = {}
            
            # 1. Bekannte Entitäten extrahieren
            sensor = record.get("Sensor", {})
            meas = record.get("Measurement", {})
            
            sensor_id = (
                sensor.get("mote_id")
                or sensor.get("moteid")
                or meas.get("mote_id")
                or meas.get("moteid")
            )
            if sensor_id:
                row["SensorID"] = sensor_id
            
            if meas:
                for k, v in meas.items():
                    if k != "moteid":
                        row[self._normalize_column_key(k)] = v

            # 2. Andere Entitäten und flache Werte (Aggregationen)
            for key, val in record.items():
                if key in ["Sensor", "Measurement"]:
                    continue
                
                if isinstance(val, dict):
                    # Verschachtelte Objekte (z.B. Date {date: ...})
                    for sub_k, sub_v in val.items():
                        row[self._normalize_column_key(f"{key}.{sub_k}")] = sub_v
                else:
                    # Flache Werte (z.B. max_humidity: 41.4)
                    row[self._normalize_column_key(key)] = val
            
            flattened_rows.append(row)
            all_keys.update(row.keys())

        # Spalten sortieren
        priority_cols = ["time", "date", "SensorID", "temperature", "humidity", "light", "voltage"]
        # Füge Aggregations-Spalten (die nicht in priority sind) hinzu
        other_cols = [c for c in all_keys if c not in priority_cols]
        sorted_cols = [c for c in priority_cols if c in all_keys] + sorted(other_cols)

        if not sorted_cols:
            return "Keine tabellarischen Daten extrahierbar."

        # Markdown Header
        header = "| " + " | ".join(sorted_cols) + " |"
        separator = "| " + " | ".join(["---"] * len(sorted_cols)) + " |"
        
        lines = [header, separator]

        for row in flattened_rows:
            values = []
            for col in sorted_cols:
                values.append(self._format_scalar(row.get(col, None)))
            lines.append("| " + " | ".join(values) + " |")

        return "\n".join(lines)

    def _create_summary_view(self, data: List[Dict[str, Any]]) -> str:
        """
        Aggregiert Messwerte nach Sensor.
        Falls keine Sensor/Measurement-Struktur da ist (z.B. nur Aggregationen),
        wird auf Detailansicht zurückgefallen oder eine einfache Liste ausgegeben.
        """
        sensor_stats = {}
        has_measurements = False

        # Check ob wir überhaupt Measurements haben
        for record in data:
            if "Measurement" in record:
                has_measurements = True
                break
        
        # Wenn wir nur Aggregationen haben (z.B. 100 Zeilen mit 'avg_temp'), 
        # macht die Sensor-Gruppierung keinen Sinn -> Fallback auf Tabelle
        if not has_measurements:
            return self._create_markdown_table(data)

        # Normale Sensor-Aggregation
        for record in data:
            meas = record.get("Measurement")
            if not meas: continue
            
            sensor_node = record.get("Sensor", {})
            sensor_id = (
                sensor_node.get("mote_id")
                or sensor_node.get("moteid")
                or meas.get("mote_id")
                or meas.get("moteid")
                or "Unknown"
            )

            if sensor_id not in sensor_stats:
                sensor_stats[sensor_id] = {
                    "temperature": [], "humidity": [], "light": [], "voltage": []
                }
            
            for field in ["temperature", "humidity", "light", "voltage"]:
                val = meas.get(field)
                if val is not None and isinstance(val, (int, float)):
                    sensor_stats[sensor_id][field].append(val)

        lines = []
        for sid, fields in sensor_stats.items():
            lines.append(f"**Sensor {sid}**:")
            stats_found = False
            for field_name, values in fields.items():
                if values:
                    stats_found = True
                    _min = min(values)
                    _max = max(values)
                    _avg = mean(values)
                    lines.append(
                        f"- *{field_name}*: Min={self._format_scalar(_min)} | Max={self._format_scalar(_max)} | Avg={self._format_scalar(_avg)}"
                    )
            if not stats_found:
                lines.append("- Keine numerischen Daten.")
            lines.append("")
        
        return "\n".join(lines)