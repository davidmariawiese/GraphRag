from haystack import component
import json

@component
class Organizer:

    def __init__(self, max_records: int = 50):
        self.max_records = max_records

    @component.output_types(context=str)
    def run(self, documents: list):
        """
        Erwartet eine Liste von Document-Objekten, deren content
        eine Python-Liste von Dicts wie
        [{'m': {...}}, {'m': {...}}, ...]
        enthält.
        """

        measurements = []

        for doc in documents[: self.max_records]:
            content = doc.content
            try:
                # content -> Python-Objekt (Liste von Dicts)
                data = json.loads(content)
            except Exception:
                try:
                    data = eval(content)
                except Exception:
                    # Fallback: Rohstring mit aufnehmen
                    measurements.append({"raw": content})
                    continue

            # data kann Liste oder einzelnes Dict sein
            if isinstance(data, dict):
                data = [data]

            for item in data:
                if isinstance(item, dict) and "m" in item and isinstance(item["m"], dict):
                    m = item["m"]
                elif isinstance(item, dict):
                    m = item
                else:
                    m = {"raw": str(item)}

                measurements.append(m)

        # --- Kontext-String bauen ---
        lines = []
        lines.append("GRAPH CONTEXT\n")

        for m in measurements[: self.max_records]:
            date_time = m.get("time") or ""
            temperature = m.get("temperature")
            humidity = m.get("humidity")
            light = m.get("light")
            voltage = m.get("voltage")

            lines.append(
                f"- {date_time} | "
                f"temp={temperature}, hum={humidity}, light={light}, volt={voltage}"
            )

        context = "\n".join(lines)
        print("[ORGANIZER]:" + context)
        return {"context": context}