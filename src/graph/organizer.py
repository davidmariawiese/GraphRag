from haystack import component
import json
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

    def __init__(self, threshold_k: int = 10):
        self.threshold_k = threshold_k
        
        self.alias_map = {
            "s": "Sensor",
            "m": "Measurement",
            "e": "Epoch",
            "d": "Date",
            "w": "Week",
            "h": "Hour",
            "mi": "Minute"
        }

    @component.output_types(context=str)
    def run(self, documents: list):
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
        print("[ORGANIZER]:")
        print(final_context)
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
            new_key = self.alias_map.get(key, key)
            new_record[new_key] = value
        return new_record

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
            
            sensor_id = sensor.get("mote_id") or meas.get("moteid")
            if sensor_id:
                row["SensorID"] = sensor_id
            
            if meas:
                for k, v in meas.items():
                    if k != "moteid":
                        row[k] = v

            # 2. Andere Entitäten und flache Werte (Aggregationen)
            for key, val in record.items():
                if key in ["Sensor", "Measurement"]:
                    continue
                
                if isinstance(val, dict):
                    # Verschachtelte Objekte (z.B. Date {date: ...})
                    for sub_k, sub_v in val.items():
                        row[f"{key}.{sub_k}"] = sub_v
                else:
                    # Flache Werte (z.B. max_humidity: 41.4)
                    row[key] = val
            
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
                val = row.get(col, "-")
                if isinstance(val, float):
                    val = f"{val:.2f}"
                values.append(str(val))
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
            sensor_id = sensor_node.get("mote_id") or meas.get("moteid") or "Unknown"

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
                    lines.append(f"- *{field_name}*: Min={_min:.2f} | Max={_max:.2f} | Avg={_avg:.2f}")
            if not stats_found:
                lines.append("- Keine numerischen Daten.")
            lines.append("")
        
        return "\n".join(lines)