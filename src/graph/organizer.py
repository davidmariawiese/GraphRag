from haystack import component
import json

@component
class Organizer:

    def __init__(self, max_records: int = 5000):
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

        for doc in documents:
            content = doc.content
            try:
                # content -> Python-Objekt (Liste von Dicts)
                # Da content oft ein String-Repräsentation eines Dicts ist (z.B. "{'m': ...}"),
                # versuchen wir es sicher zu parsen.
                # json.loads erwartet striktes JSON (doppelte Anführungszeichen),
                # eval ist hier ein pragmatischer Fallback für Python-Dict-Strings.
                try:
                    data = json.loads(content.replace("'", '"'))
                except json.JSONDecodeError:
                    data = eval(content)
            except Exception:
                # Fallback: Rohstring mit aufnehmen
                measurements.append({"raw": content})
                continue

            # data kann Liste oder einzelnes Dict sein
            if isinstance(data, dict):
                data = [data]

            for item in data:
                # Wir suchen nach typischen Schlüsseln oder nehmen das ganze Item
                # Fall 1: {'m': {...}} (typisch bei RETURN m)
                if isinstance(item, dict) and "m" in item and isinstance(item["m"], dict):
                    m = item["m"]
                    # Optional: Sensor-Daten aus 's' dazunehmen, falls vorhanden
                    if "s" in item and isinstance(item["s"], dict):
                        m.update(item["s"])
                # Fall 2: {'s': {...}, 'm': {...}} (Sensor + Messung)
                elif isinstance(item, dict) and "m" in item:
                     m = item["m"]
                     # Optional: Sensor-Daten aus 's' dazunehmen, falls vorhanden
                     if "s" in item and isinstance(item["s"], dict):
                         m.update(item["s"])
                # Fall 3: Flaches Dict (z.B. RETURN {temp: m.temperature, ...})
                elif isinstance(item, dict):
                    m = item
                else:
                    m = {"raw": str(item)}

                measurements.append(m)

        # --- Kontext-String bauen ---
        lines = []
        lines.append("GRAPH CONTEXT (Messdaten):\n")

        if not measurements:
            lines.append("Keine passenden Messdaten gefunden.")
            return {"context": "\n".join(lines)}
        
        # Wenn wenige Daten: Detaillierte Auflistung
        if len(measurements) <= 15:
            for i, m in enumerate(measurements):
                # Zeit
                date_time = m.get("time") or m.get("timestamp") or "N/A"
                # Sensor
                sensor = m.get("sensor_id") or m.get("moteid") or m.get("mote_id") or m.get("id") or "?"
                
                # Werte
                vals = []
                temp = m.get("temperature")
                hum = m.get("humidity")
                light = m.get("light")
                volt = m.get("voltage")

                if temp is not None: vals.append(f"Temp={temp}°C")
                if hum is not None: vals.append(f"Hum={hum}%")
                if light is not None: vals.append(f"Light={light}")
                if volt is not None: vals.append(f"Volt={volt}V")
                
                val_str = ", ".join(vals) if vals else "Keine Messwerte"
                lines.append(f"[{i+1}] Sensor {sensor} @ {date_time}: {val_str}")

        # Wenn viele Daten: Zusammenfassung
        else:
            lines.append(f"Zusammenfassung von {len(measurements)} Messwerten (reduziert für Kontext):\n")
            
            # Gruppieren nach Sensor
            by_sensor = {}
            for m in measurements:
                s_id = m.get("sensor_id") or m.get("moteid") or m.get("mote_id") or m.get("id") or "Unknown"
                if s_id not in by_sensor:
                    by_sensor[s_id] = []
                by_sensor[s_id].append(m)
            
            # Statistiken pro Sensor
            for s_id, ms in by_sensor.items():
                count = len(ms)
                # Zeitspanne
                times = sorted([str(m.get("time") or m.get("timestamp") or "") for m in ms if m.get("time") or m.get("timestamp")])
                t_range = f"{times[0]} bis {times[-1]}" if times else "Zeit unbekannt"
                
                # Durchschnitte
                stats = []
                for key, label in [("temperature", "Temp"), ("humidity", "Hum"), ("light", "Light"), ("voltage", "Volt")]:
                    vals = []
                    for m in ms:
                        val = m.get(key)
                        if val is not None:
                            try:
                                vals.append(float(val))
                            except (ValueError, TypeError):
                                pass
                    
                    if vals:
                        avg = sum(vals) / len(vals)
                        stats.append(f"Ø {label}={avg:.1f}")
                
                stats_str = ", ".join(stats)
                lines.append(f"- Sensor {s_id}: {count} Messungen, {t_range}. {stats_str}")

        context = "\n".join(lines)
        
        # Debug-Print (kann später entfernt werden)
        print("[ORGANIZER]:" + context)
        
        return {"context": context}