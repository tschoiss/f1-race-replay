import numpy as np

def resample_weather(session, timeline, global_t_min):
    """
    Resample weather data from a FastF1 session onto a common timeline.
    Returns dict with track_temp, air_temp, humidity, wind_speed, wind_direction, rainfall.
    """
    weather_resampled = None
    weather_df = getattr(session, "weather_data", None)
    if weather_df is not None and not weather_df.empty:
        try:
            weather_times = weather_df["Time"].dt.total_seconds().to_numpy() - global_t_min
            if len(weather_times) > 0:
                order = np.argsort(weather_times)
                weather_times = weather_times[order]

                def _maybe_get(name):
                    return weather_df[name].to_numpy()[order] if name in weather_df else None

                def _resample(series):
                    if series is None:
                        return None
                    return np.interp(timeline, weather_times, series)

                track_temp = _resample(_maybe_get("TrackTemp"))
                air_temp = _resample(_maybe_get("AirTemp"))
                humidity = _resample(_maybe_get("Humidity"))
                wind_speed = _resample(_maybe_get("WindSpeed"))
                wind_direction = _resample(_maybe_get("WindDirection"))
                rainfall_raw = _maybe_get("Rainfall")
                rainfall = _resample(rainfall_raw.astype(float)) if rainfall_raw is not None else None

                weather_resampled = {
                    "track_temp": track_temp,
                    "air_temp": air_temp,
                    "humidity": humidity,
                    "wind_speed": wind_speed,
                    "wind_direction": wind_direction,
                    "rainfall": rainfall,
                }
        except Exception as e:
            print(f"Weather data could not be processed: {e}")
    return weather_resampled

def build_weather_snapshot(weather_resampled, i):
    """
    Creates weather-snapshot for frame i.
    Returns {}, if empty or error.
    """
    if not weather_resampled:
        return {}

    try:
        wt = weather_resampled
        rain_val = wt["rainfall"][i] if wt.get("rainfall") is not None else 0.0

        return {
            "track_temp": float(wt["track_temp"][i]) if wt.get("track_temp") is not None else None,
            "air_temp": float(wt["air_temp"][i]) if wt.get("air_temp") is not None else None,
            "humidity": float(wt["humidity"][i]) if wt.get("humidity") is not None else None,
            "wind_speed": float(wt["wind_speed"][i]) if wt.get("wind_speed") is not None else None,
            "wind_direction": float(wt["wind_direction"][i]) if wt.get("wind_direction") is not None else None,
            "rain_state": "RAINING" if rain_val and rain_val >= 0.5 else "DRY",
        }
    except Exception as e:
        print(f"Failed to attach weather data to frame {i}: {e}")
        return {}