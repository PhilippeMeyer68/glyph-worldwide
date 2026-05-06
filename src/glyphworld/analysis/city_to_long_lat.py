# -*- coding: utf-8 -*-
"""
Cities to longitude and latitudes.
"""

import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

from glyphworld.core.paths import DATA_DIR

geolocator = Nominatim(user_agent="glyphworld-geocoder")  # change string to your project name
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.0)  # be polite to the server

_cache = {}


def city_to_latlon(city, country=None, language="en"):
    city = (city or "").strip()
    country = (country or "").strip() if country else None
    if not city:
        return None
    query_parts = [city]
    if country:
        query_parts.append(country)
    query = ", ".join(query_parts).strip()
    if query in _cache:
        return _cache[query]
    loc = geocode(query, exactly_one=True, language=language)
    if loc is None:
        _cache[query] = None
        return None
    latlon = (float(loc.latitude), float(loc.longitude))
    _cache[query] = latlon
    return latlon


def _split_semicolon_cell(x):
    """Split 'A; B; C' into ['A','B','C'] (robust to NaN)."""
    if x is None or (isinstance(x, float) and pd.isna(x)) or (isinstance(x, str) and x.strip() == ""):
        return []
    return [p.strip() for p in str(x).split(";") if p.strip()]


def geocode_row(script, cities_cell, countries_cell, language="en"):
    cities = _split_semicolon_cell(cities_cell)
    countries = _split_semicolon_cell(countries_cell)

    # If user provided 1 country for many cities, broadcast it
    if len(countries) == 1 and len(cities) > 1:
        countries = countries * len(cities)

    # If mismatch remains, pad with None to avoid crash (keeps alignment)
    if len(countries) < len(cities):
        countries = countries + [None] * (len(cities) - len(countries))
    elif len(countries) > len(cities):
        countries = countries[: len(cities)]

    lats = []
    lons = []
    for city, country in zip(cities, countries):
        latlon = city_to_latlon(city, country=country, language=language)
        if latlon is None:
            lats.append("")
            lons.append("")
            print(f"[WARN] Script {script}, Not found: {city!r}, {country!r}")
        else:
            lat, lon = latlon
            lats.append(f"{lat:.6f}")
            lons.append(f"{lon:.6f}")

    return "; ".join(lats), "; ".join(lons)


if __name__ == "__main__":
    for q in [("Paris", "France"), ("Springfield", "USA")]:
        city, country = q
        latlon = city_to_latlon(city, country=country)
        print(f"{city}, {country}: {latlon}")

    SCRIPTS_LOCATION = DATA_DIR / "info" / "scripts_location.xlsx"
    df = pd.read_excel(SCRIPTS_LOCATION)

    # compute per row
    out = df.apply(
        lambda r: geocode_row(r.get("Script"), r.get("Cities"), r.get("Countries"), language="en"),
        axis=1,
        result_type="expand",
    )
    out.columns = ["Latitudes", "Longitudes"]  # <-- FIX: only 2 outputs

    df = pd.concat([df, out], axis=1)

    out_path = SCRIPTS_LOCATION.with_name(SCRIPTS_LOCATION.stem + "_with_latlon.xlsx")
    df.to_excel(out_path, index=False)
    print(f"[OK] Saved: {out_path}")
