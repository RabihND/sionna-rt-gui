#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
Optional geo-referencing for scenes.

A scene is geo-referenced when a ``BBOX.json`` sidecar sits next to its XML
file (written e.g. by osm2sionna) with an ``origin`` of the local ENU frame:

    {"origin": {"lat": 45.5025, "lon": -73.587, "alt": 160.5}, ...}

ENU convention: x = east, y = north, z = up, meters, z = 0 at origin ground.
"""
from __future__ import annotations

import json
import math
import os

METERS_PER_DEG_LAT = 111320.0


def load_geo_sidecar(scene_xml_path: str) -> dict | None:
    """Returns the ENU origin {"lat", "lon", "alt"} or None."""
    try:
        path = os.path.join(
            os.path.dirname(os.path.abspath(scene_xml_path)), "BBOX.json"
        )
        if not os.path.isfile(path):
            return None
        with open(path, "r") as f:
            origin = (json.load(f) or {}).get("origin") or {}
        if "lat" not in origin or "lon" not in origin:
            return None
        return {
            "lat": float(origin["lat"]),
            "lon": float(origin["lon"]),
            "alt": float(origin.get("alt", 0.0)),
        }
    except Exception:
        return None


def enu_to_gps(
    origin: dict, x: float, y: float, z: float | None = None
) -> tuple[float, float, float | None]:
    """Local ENU meters -> (lat, lon, altitude ASL). Inverse of the
    equirectangular projection used when generating geo-referenced scenes."""
    m_lon = METERS_PER_DEG_LAT * math.cos(math.radians(origin["lat"]))
    lat = origin["lat"] + y / METERS_PER_DEG_LAT
    lon = origin["lon"] + x / m_lon
    alt = None if z is None else origin["alt"] + z
    return lat, lon, alt
