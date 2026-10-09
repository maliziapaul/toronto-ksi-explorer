"""Build the compact data payload for the Toronto KSI Explorer.

Inputs (all City of Toronto Open Data, downloaded into this folder):
  ksi.csv            Motor Vehicle Collisions involving KSI persons (EPSG:4326 CSV)
  nbhd.geojson       Neighbourhoods (158)
  wards.geojson      City Wards (25)
  centreline.geojson Toronto Centreline v2
Output: data.json (columnar, dictionary-coded, coordinates quantized to 1e-5 deg)
"""
import json, math, re
from collections import Counter, defaultdict
import pandas as pd
from shapely.geometry import shape, Polygon, MultiPolygon, LineString, MultiLineString
from shapely.ops import unary_union, linemerge

Q = 1e-5
EPOCH = pd.Timestamp("2006-01-01")

# ---------------------------------------------------------------- collisions / persons
d = pd.read_csv("ksi.csv", low_memory=False)
d["when"] = pd.to_datetime(d.accdate)
d = d.sort_values(["when", "collision_id", "per_no"], na_position="last").reset_index(drop=True)


def title_st(s):
    if not isinstance(s, str) or not s.strip():
        return ""
    s = s.strip().title()
    s = re.sub(r"\b(Ii|Iii|Iv)\b", lambda m: m.group(0).upper(), s)
    s = re.sub(r"\bO Connor\b", "O'Connor", s)
    s = re.sub(r"\bBoul\b", "Blvd", s); s = re.sub(r"\bAven\b", "Ave", s)
    s = re.sub(r"\bMc([a-z])", lambda m: "Mc" + m.group(1).upper(), s)
    s = re.sub(r"\bSt Clair\b", "St. Clair", s)
    return s


def group_light(v):
    if not isinstance(v, str): return "Not recorded"
    if v.startswith("Daylight"): return "Daylight"
    if v == "Dark": return "Dark, unlit"
    if v == "Dark with Artificial Lighting": return "Dark, street-lit"
    if v.startswith("Dusk") or v.startswith("Dawn"): return "Dawn or dusk"
    return "Other"


def group_surface(v):
    if not isinstance(v, str): return "Not recorded"
    if v in ("Dry", "Wet"): return v
    if v in ("Loose Snow", "Slush", "Ice", "Packed Snow"): return "Snow, slush or ice"
    return "Other"


def group_visible(v):
    if not isinstance(v, str): return "Not recorded"
    if v == "Clear": return "Clear"
    if v == "Rain": return "Rain"
    if v in ("Snow", "Drifting Snow", "Freezing Rain or Hail"): return "Snow or freezing rain"
    return "Fog, wind or other"


def group_loc(v):
    if not isinstance(v, str): return "Not recorded"
    if v in ("At Intersection", "Four-way", "T Intersection", "Y Intersection", "L Intersection",
             "Offset", "Roundabout", "Other Intersection"): return "At intersection"
    if v == "Intersection-Related": return "Near intersection"
    if v == "Non-Intersection": return "Midblock"
    if v in ("At/Near Private Drive", "Private Driveway", "Laneway"): return "Driveway or laneway"
    return "Other"


def group_ctl(v):
    if not isinstance(v, str) or v == "Unknown": return "Not recorded"
    if v in ("Traffic Signal", "Traffic Controller", "Half Signal", "Pedestrian Signal"): return "Traffic signal"
    if v == "No Control": return "No control"
    if v == "Stop Sign": return "Stop sign"
    if v == "Pedestrian Crossover": return "Pedestrian crossover"
    return "Other"


IMPACT = {"Pedestrian Collision (internal code)": "Pedestrian struck",
          "Cyclist Collision (internal code)": "Cyclist struck",
          "Turning Movement": "Turning movement", "SMV Other": "Single vehicle",
          "Rear End": "Rear end", "Angle": "Angle", "Approaching Head On": "Head-on",
          "Sideswipe": "Sideswipe", "Reversing": "Reversing",
          "SMV Unattended Vehicle": "Hit parked vehicle"}


def group_impact(v):
    if not isinstance(v, str): return "Not recorded"
    return IMPACT.get(v, "Other")


def group_road(v):
    if not isinstance(v, str) or v == "Pending": return "Not recorded"
    if v.startswith("Major Arterial"): return "Major arterial"
    if v == "Minor Arterial": return "Minor arterial"
    if v == "Collector": return "Collector"
    if v == "Local": return "Local"
    if v.startswith("Expressway"): return "Expressway"
    return "Other"


def group_ru(v):
    return {"pedestrian": "Pedestrian", "driver": "Driver", "passenger": "Passenger", "cyclist": "Cyclist",
            "motorcyclist": "Motorcyclist", "other_micromobility": "E-scooter / other micromobility"}.get(v, "Other")


def group_veh(v):
    if not isinstance(v, str) or v in ("Unknown", "Other"): return "Other / unknown"
    if v == "Automobile or Station Wagon": return "Car"
    if v in ("Pickup Truck", "Passenger Van"): return "Pickup or passenger van"
    if v in ("Municipal Transit Bus (TTC)", "Streetcar"): return "TTC bus or streetcar"
    if "Bus" in v: return "Other bus"
    if v.startswith("Truck") or v in ("Delivery Van", "Tow Truck", "Construction Equipment", "Fire Vehicle"): return "Truck or delivery van"
    if v in ("Motorcycle", "Moped"): return "Motorcycle or moped"
    if v in ("Bicycle",): return "Bicycle"
    if v.startswith("E-") or "Scooter" in v or "Wheelchair" in v: return "E-bike, e-scooter or other"
    if "Police" in v or "Ambulance" in v: return "Emergency vehicle"
    return "Other / unknown"


def group_cond(v):
    if not isinstance(v, str): return None
    if v == "Normal": return "Normal"
    if v.startswith("Inattentive"): return "Inattentive"
    if "Alcohol" in v or v == "Had Been Drinking": return "Alcohol"
    if "Drugs" in v: return "Drugs"
    if v.startswith("Medical"): return "Medical or physical"
    if v.startswith("Fatigue"): return "Fatigue / asleep"
    if v == "Unknown": return "Unknown"
    return "Other"


def clean(v):
    return v if isinstance(v, str) and v.strip() else None


AGE_BANDS = ["0–14", "15–24", "25–34", "35–44", "45–54", "55–64", "65–74", "75–84", "85+"]


def age_band(a):
    if pd.isna(a) or a > 105: return -1
    a = int(a)
    if a < 15: return 0
    if a >= 85: return 8
    return (a - 5) // 10


class Dict:
    def __init__(self, fixed=None):
        self.items, self.idx = [], {}
        for f in fixed or []: self.get(f)
    def get(self, v):
        if v is None: return -1
        if v not in self.idx:
            self.idx[v] = len(self.items); self.items.append(v)
        return self.idx[v]


D = {k: Dict() for k in ["street", "loc", "accloc", "ctl", "impact", "light", "surface", "visible", "road",
                          "ru", "veh", "drivact", "drivcond", "pedact", "pedcond", "cyctype", "pedtype", "manoeuvre"]}
D["inj"] = Dict(["Fatal", "Major", "Minor", "Minimal", "None"])
D["ru"] = Dict(["Pedestrian", "Driver", "Passenger", "Cyclist", "Motorcyclist", "E-scooter / other micromobility", "Other"])

wards_gj = json.load(open("wards.geojson"))
nb_gj = json.load(open("nbhd.geojson"))
WARDS = sorted([(f["properties"]["AREA_NAME"], int(f["properties"]["AREA_SHORT_CODE"])) for f in wards_gj["features"]], key=lambda x: x[1])
WARD_IDX = {f"{n} ({c})": i for i, (n, c) in enumerate(WARDS)}
NBS = sorted(f["properties"]["AREA_NAME"] for f in nb_gj["features"])
NB_IDX = {n: i for i, n in enumerate(NBS)}

FLAGS = ["pedestrian", "cyclist", "motorcyclist", "other_micromobility", "heavy_truck", "older_adult",
         "school_child", "aggressive", "distracted", "red_light", "speeding", "impaired"]

# collisions
C = defaultdict(list)
locs = {}  # (latq, lonq) -> idx
loc_names = defaultdict(Counter)
L = {"lat": [], "lon": []}
cidx = {}
for cid, g in d.groupby("collision_id", sort=False):
    r = g.iloc[0]
    i = len(cidx); cidx[cid] = i
    dt = r["when"]
    C["day"].append(int((dt.normalize() - EPOCH).days))
    C["min"].append(int(dt.hour * 60 + dt.minute))
    s1, s2 = title_st(r.stname1), title_st(r.stname2)
    nm = f"{s1} & {s2}" if s1 and s2 else (s1 or s2 or "Unnamed location")
    if pd.isna(r.latitude) or pd.isna(r.longitude):
        C["loc"].append(-1)
    else:
        la, lo = round(r.latitude / Q), round(r.longitude / Q)
        key = (la, lo)
        if key not in locs:
            locs[key] = len(locs); L["lat"].append(la); L["lon"].append(lo)
        C["loc"].append(locs[key])
        loc_names[locs[key]][nm] += 1
    C["st1"].append(D["street"].get(s1 or None)); C["st2"].append(D["street"].get(s2 or None))
    C["ward"].append(WARD_IDX.get(r.wardname, -1)); C["nb"].append(NB_IDX.get(r.neighbourhood, -1))
    C["accloc"].append(D["accloc"].get(group_loc(r.accloc)))
    C["ctl"].append(D["ctl"].get(group_ctl(r.traffictl)))
    C["impact"].append(D["impact"].get(group_impact(r.impactype)))
    C["light"].append(D["light"].get(group_light(r.light)))
    C["surface"].append(D["surface"].get(group_surface(r.rdsfcond)))
    C["visible"].append(D["visible"].get(group_visible(r.visible)))
    C["road"].append(D["road"].get(group_road(r.road_class)))
    speeding = g.drivact.isin(["Exceeding speed limit", "Speed too fast for conditions"]).any()
    impaired = (g.drivcond.fillna("").str.contains("Alcohol|Drinking|Drugs") |
                g.pedcond.fillna("").str.contains("Alcohol|Drinking|Drugs")).any()
    vals = {f: bool(g[f].any()) for f in FLAGS[:10]}
    vals["speeding"], vals["impaired"] = bool(speeding), bool(impaired)
    C["flags"].append(sum(1 << k for k, f in enumerate(FLAGS) if vals[f]))
    C["K"].append(int((g.injury == "Fatal").sum())); C["S"].append(int((g.injury == "Major").sum()))
    C["fatal"].append(1 if r.acclass == "Fatal Injury" else 0)

L["name"] = [D["loc"].get(loc_names[i].most_common(1)[0][0]) for i in range(len(locs))]

# persons
P = defaultdict(list)
for r in d.itertuples():
    P["c"].append(cidx[r.collision_id])
    P["ru"].append(D["ru"].get(group_ru(r.road_user)))
    P["inj"].append(D["inj"].get(r.injury if isinstance(r.injury, str) else "None"))
    P["age"].append(age_band(r.invage))
    P["veh"].append(D["veh"].get(group_veh(r.vehtype)) if isinstance(r.vehtype, str) else -1)
    P["vno"].append(-1 if pd.isna(r.veh_no) else int(r.veh_no))
    P["da"].append(D["drivact"].get(clean(r.drivact)))
    P["dc"].append(D["drivcond"].get(group_cond(r.drivcond)))
    P["pa"].append(D["pedact"].get(clean(r.pedact)))
    P["pc"].append(D["pedcond"].get(group_cond(r.pedcond)))
    P["ct"].append(D["cyctype"].get(clean(r.cyclistype)))
    P["pt"].append(D["pedtype"].get(clean(r.pedtype)))
    P["mv"].append(D["manoeuvre"].get(clean(r.manoeuvre)))

# ---------------------------------------------------------------- geometry
def enc_ring(coords):
    out, px, py = [], 0, 0
    pts = [(round(x / Q), round(y / Q)) for x, y in coords]
    ded = [pts[0]] + [p for a, p in zip(pts, pts[1:]) if p != a]
    out.append(len(ded))
    for x, y in ded:
        out += [x - px, y - py]; px, py = x, y
    return out


def enc_poly(geom, tol):
    geom = geom.simplify(tol, preserve_topology=True)
    polys = [geom] if isinstance(geom, Polygon) else list(getattr(geom, "geoms", []))
    flat = []
    for p in polys:
        if p.is_empty: continue
        for ring in [p.exterior, *p.interiors]:
            if len(ring.coords) >= 4: flat += enc_ring(ring.coords)
    return flat


def area_km2(geom):
    lat0 = geom.centroid.y
    kx, ky = 111.32 * math.cos(math.radians(lat0)), 110.57
    from shapely.affinity import scale
    return scale(geom, xfact=kx, yfact=ky, origin=(0, 0)).area


def label_pt(geom):
    p = geom.representative_point() if not geom.contains(geom.centroid) else geom.centroid
    return [round(p.x / Q), round(p.y / Q)]


ward_geo = []
wg = {int(f["properties"]["AREA_SHORT_CODE"]): shape(f["geometry"]) for f in wards_gj["features"]}
for n, c in WARDS:
    g = wg[c]
    ward_geo.append({"name": n, "code": c, "rings": enc_poly(g, 0.00012), "km2": round(area_km2(g), 2), "lab": label_pt(g)})
nbg = {f["properties"]["AREA_NAME"]: shape(f["geometry"]) for f in nb_gj["features"]}
nb_geo = []
for n in NBS:
    g = nbg[n]
    nb_geo.append({"name": n, "rings": enc_poly(g, 0.00008), "km2": round(area_km2(g), 2), "lab": label_pt(g)})

land = unary_union(list(nbg.values())).buffer(0.00005).buffer(-0.00005)
# Lake Ontario: a polygon that runs along the outer shore to the east and west of the city and
# back through the city's interior; subtracting the city leaves water south of the shoreline.
lake_raw = Polygon([(-79.541, 43.584), (-79.62, 43.53), (-79.70, 43.47), (-79.80, 43.40), (-79.80, 43.20),
                    (-78.85, 43.20), (-78.85, 43.85), (-78.95, 43.83), (-79.05, 43.808), (-79.117, 43.796),
                    (-79.20, 43.76), (-79.33, 43.70), (-79.42, 43.66), (-79.50, 43.62), (-79.541, 43.584)])
lake = lake_raw.difference(land)
if isinstance(lake, MultiPolygon):
    big = max(lake.geoms, key=lambda p: p.area)
    lake = big

ROAD_CLASSES = [("expressway", ["Expressway", "Expressway Ramp"], 0.00006),
                ("major", ["Major Arterial", "Major Arterial Ramp"], 0.00005),
                ("minor", ["Minor Arterial", "Minor Arterial Ramp"], 0.00005),
                ("collector", ["Collector", "Collector Ramp"], 0.00004),
                ("local", ["Local"], 0.00003),
                ("rail", ["Major Railway"], 0.00006),
                ("water", ["River", "Creek/Tributary"], 0.00008)]
cl = json.load(open("centreline.geojson"))
by_desc = defaultdict(list)
for f in cl["features"]:
    if f["geometry"] is None: continue
    by_desc[f["properties"]["FEATURE_CODE_DESC"]].append(shape(f["geometry"]))
roads = {}
for key, descs, tol in ROAD_CLASSES:
    lines = []
    for ds in descs:
        for g in by_desc.get(ds, []):
            lines += list(g.geoms) if isinstance(g, MultiLineString) else [g]
    merged = linemerge(lines) if lines else None
    parts = list(merged.geoms) if isinstance(merged, MultiLineString) else ([merged] if merged else [])
    flat = []
    for ln in parts:
        s = ln.simplify(tol)
        if len(s.coords) >= 2: flat += enc_ring(s.coords)
    roads[key] = flat
    print(key, len(parts), len(flat))

meta = {
    "updated": d["when"].max().strftime("%Y-%m-%d"),
    "first": d["when"].min().strftime("%Y-%m-%d"),
    "epoch": "2006-01-01",
    "refreshed": "2026-10-08",
}
out = {
    "meta": meta, "q": Q, "flags": FLAGS, "ageBands": AGE_BANDS,
    "dict": {k: v.items for k, v in D.items()},
    "wards": [{"name": w["name"], "code": w["code"]} for w in ward_geo],
    "nbs": NBS,
    "C": dict(C), "L": L, "P": dict(P),
    "geo": {"land": enc_poly(land, 0.0001), "lake": enc_poly(lake, 0.0001),
            "wards": [{"rings": w["rings"], "km2": w["km2"], "lab": w["lab"]} for w in ward_geo],
            "nbs": [{"rings": n["rings"], "km2": n["km2"], "lab": n["lab"]} for n in nb_geo],
            "roads": roads},
}
s = json.dumps(out, separators=(",", ":"))
open("data.json", "w").write(s)
print("collisions", len(cidx), "persons", len(P["c"]), "locations", len(locs), "bytes", len(s))
for k in ["C", "P", "L"]:
    print(k, len(json.dumps(out[k], separators=(",", ":"))))
print("geo", {k: len(json.dumps(v, separators=(",", ":"))) for k, v in out["geo"].items()})
print({k: v.items[:40] for k, v in D.items() if k not in ("street", "loc")})
