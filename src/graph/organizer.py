# pipeline/organizer.py
from haystack import component
import json

@component
class Organizer:

    def __init__(self, max_records: int = 50):
        self.max_records = max_records

    @component.output_types(context=str)
    def run(self, documents: list):
        """
        Haystack will pass a list of Document objects here.
        We do NOT type this as List[Document] because Pipeline
        would consider that a mismatch.
        """
        
        parsed = []

        # --- parse raw Neo4j document contents ---
        for doc in documents[: self.max_records]:
            content = doc.content  # doc is a Document → but type is list for Haystack
            try:
                rec = json.loads(content)
            except Exception:
                try:
                    rec = eval(content)
                except Exception:
                    rec = {"raw": content}
            parsed.append(rec)

        # --- group records ---
        groups = {}
        for r in parsed:
            sensor = r.get("sensor_id") or r.get("sensorId") or "UNKNOWN_SENSOR"
            epoch = r.get("epoch_id") or r.get("epochId") or "UNKNOWN_EPOCH"
            key = f"Sensor {sensor} (Epoch {epoch})"
            groups.setdefault(key, []).append(r)

        # --- build context block ---
        lines = []
        lines.append("GRAPH CONTEXT\n")

        for group_key, items in groups.items():
            lines.append(group_key + ":")
            for r in items[:10]:
                date = r.get("date")
                time = r.get("time")
                temp = r.get("temperature")
                hum = r.get("humidity")
                light = r.get("light")
                volt = r.get("voltage")

                lines.append(
                    f"- {date or ''} {time or ''} | "
                    f"temp={temp}, hum={hum}, light={light}, volt={volt}"
                )
            lines.append("")

        context = "\n".join(lines)
        print("[KONTEXT]:"+ context)
        return {"context": context}
