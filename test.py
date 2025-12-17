# energy_paths.py
import os, glob
from pathlib import Path

import streamlit as st
import geopandas as gpd
import numpy as np
import plotly.graph_objects as go
import networkx as nx

from shapely.geometry import Point, LineString, MultiLineString
from shapely.strtree import STRtree
from pyproj import CRS

try:
    from shapely.validation import make_valid
except Exception:
    make_valid = None

# ---------------- UI ----------------
st.set_page_config(page_title="HV Network Paths", layout="wide")
st.title("⚡ Substations & HV Lines — Path Distance")
st.markdown(
    "Pick **From** and **To** substations, compute distance **along HV lines**, "
    "and place a marker at a given distance from the start. Popup shows Lat/Lon "
    "and Village/Cell/Sector/District if present in the village layer."
)

BASE_DIR = Path(__file__).parent.resolve()
root = st.sidebar.text_input("Folder with layers (.fgb / .geojson / .shp)", value=str(BASE_DIR))
map_zoom   = st.sidebar.slider("Map zoom", 8, 15, 10)
pt_size    = st.sidebar.slider("Substation marker size", 6, 20, 10)
ln_width   = st.sidebar.slider("HV line width", 1, 8, 3)
unit       = st.sidebar.selectbox("Distance unit", ["meters", "kilometers"], index=0)
dist_value = st.sidebar.number_input("Distance from start", min_value=0.0, value=0.0, step=100.0)
hv_color   = st.sidebar.color_picker("HV line color", "#2E86AB")
path_color = st.sidebar.color_picker("Shortest path color", "#16A085")
SNAP_TOL   = st.sidebar.number_input("Endpoint snap tolerance (m)", 0.0, 20.0, 3.0, 1.0)

# ---- CRS (Rwanda TM, meters) if a layer lacks CRS ----
RWANDA_TM30 = CRS.from_proj4(
    "+proj=tmerc +lat_0=0 +lon_0=30 +k=0.9999 +x_0=500000 +y_0=5000000 +datum=WGS84 +units=m +no_defs"
)

def to_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        gdf = gdf.set_crs(RWANDA_TM30)
    return gdf.to_crs(4326)

def to_tm(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        gdf = gdf.set_crs(RWANDA_TM30)
    return gdf.to_crs(RWANDA_TM30)

def prefer_fgb(paths):
    by_stem = {}
    for p in sorted(paths):
        stem = Path(p).stem.lower()
        if stem not in by_stem or p.lower().endswith(".fgb"):
            by_stem[stem] = p
    return list(by_stem.values())

def find_file(folder, keyword):
    """Find files containing keyword (case-insensitive)."""
    keyword_lower = keyword.lower()
    matches = []
    pats = [
        f"**/*{keyword}*.fgb",
        f"**/*{keyword}*.geojson",
        f"**/*{keyword}*.json",
        f"**/*{keyword}*.shp",
    ]
    paths = []
    for pat in pats:
        paths.extend(glob.glob(os.path.join(folder, pat), recursive=True))
    return prefer_fgb(paths)

def safe_make_valid(g):
    if g is None or g.is_empty:
        return None
    if not g.is_valid:
        try:
            return make_valid(g) if make_valid else g.buffer(0)
        except Exception:
            return None
    return g

def pick_first_present(gdf, candidates):
    """Return the first existing column among candidates (case-insensitive)."""
    cols_lower = {c.lower(): c for c in gdf.columns}
    for name in candidates:
        if name.lower() in cols_lower:
            return cols_lower[name.lower()]
    return None

# ------------- Load core layers -------------
sub_files = find_file(root, "substation")
hv_files  = find_file(root, "hvline")

if not sub_files:
    st.error("No **substation** file found (expects name containing 'substation')."); st.stop()
if not hv_files:
    st.error("No **HV line** file found (expects name containing 'hvline')."); st.stop()

sub_path = sub_files[0]
hv_path  = hv_files[0]
st.caption(f"Using **Substations:** {os.path.relpath(sub_path, root)}  •  **HV Lines:** {os.path.relpath(hv_path, root)}")

subs = gpd.read_file(sub_path)
hvl  = gpd.read_file(hv_path)

# drop empties
subs = subs[subs.geometry.notna() & (~subs.geometry.is_empty)].copy()
hvl  = hvl[hvl.geometry.notna() & (~hvl.geometry.is_empty)].copy()

# keep only points / lines
subs = subs[subs.geometry.type.isin(["Point","MultiPoint"])].explode(index_parts=False, ignore_index=True)
subs = subs[subs.geometry.type == "Point"].copy()
hvl  = hvl[hvl.geometry.type.isin(["LineString","MultiLineString"])].copy()

# substation labels
if "Substation" not in subs.columns:
    st.error("Column 'Substation' not found in the substation layer."); st.stop()
subs["Substation"] = subs["Substation"].astype(str).str.strip()
mask_empty = subs["Substation"].eq("") | subs["Substation"].str.lower().eq("none")
subs.loc[mask_empty, "Substation"] = [f"Sub_{i+1}" for i in range(mask_empty.sum())]
if subs["Substation"].duplicated().any():
    subs["Substation"] = subs["Substation"] + (
        subs.groupby("Substation").cumcount().replace({0:""}).astype(str)
        .radd(np.where(subs.groupby("Substation").cumcount()==0,""," #"))
    )
sub_label = "Substation"

# prepare for map (WGS84) and network (TM meters)
subs_ll = to_wgs84(subs)
hvl_ll  = to_wgs84(hvl)
subs_tm = to_tm(subs)
hvl_tm  = to_tm(hvl)

# --------- Optional admin (village) layer ---------
admin_gdf = None
village_files = find_file(root, "village")
if village_files:
    try:
        g = gpd.read_file(village_files[0])
        g = g[g.geometry.notna() & (~g.geometry.is_empty)]
        g = to_wgs84(g)
        g = g[g.geometry.type.isin(["Polygon","MultiPolygon","GeometryCollection"])]
        admin_gdf = g
        st.caption("Admin source → " + os.path.relpath(village_files[0], root))
    except Exception as e:
        st.warning(f"Could not read village admin layer: {e}")

# ---------------- Build graph WITHOUT nodding at crossings ----------------
# Connect along each line's own vertices; different lines only connect at shared endpoints.
def snap_xy(x, y, tol):
    """
    Return a string representation of snapped coordinates.
    Using strings as node IDs to avoid tuple hashing issues.
    """
    if tol <= 0:
        return f"{round(x, 3)},{round(y, 3)}"
    qx = round(x / tol) * tol
    qy = round(y / tol) * tol
    return f"{round(qx, 3)},{round(qy, 3)}"

def string_to_coords(node_str):
    """Convert node string back to (x, y) tuple."""
    if isinstance(node_str, tuple):
        return node_str
    try:
        x_str, y_str = node_str.split(',')
        return (float(x_str), float(y_str))
    except:
        # If already a tuple, return it
        if isinstance(node_str, (tuple, list)) and len(node_str) == 2:
            return (float(node_str[0]), float(node_str[1]))
        return (0.0, 0.0)

G = nx.Graph()
segments = []   # LineString segments (TM meters)
seg_uv   = []   # (u, v) endpoints for each segment (as strings)

def add_polyline_segments(geom):
    """Add edges for consecutive vertices (preserve polyline shape)."""
    if isinstance(geom, MultiLineString):
        lines = list(geom.geoms)
    elif isinstance(geom, LineString):
        lines = [geom]
    else:
        return
    for ln in lines:
        coords = list(ln.coords)
        if len(coords) < 2:
            continue
        for a, b in zip(coords[:-1], coords[1:]):
            u = snap_xy(a[0], a[1], SNAP_TOL)
            v = snap_xy(b[0], b[1], SNAP_TOL)
            
            # Convert string nodes to tuples for LineString creation
            u_coords = string_to_coords(u)
            v_coords = string_to_coords(v)
            
            seg = LineString([u_coords, v_coords])
            seg_len = seg.length
            if seg_len <= 0:
                continue
            G.add_edge(u, v, weight=float(seg_len))
            segments.append(seg)
            seg_uv.append((u, v))

for g in hvl_tm.geometry:
    gg = safe_make_valid(g)
    if gg is None:
        continue
    add_polyline_segments(gg)

st.caption(f"Graph: **{G.number_of_nodes()} nodes**, **{G.number_of_edges()} edges**")
if G.number_of_nodes() == 0:
    st.error("HV line graph is empty—cannot compute paths."); st.stop()

# STRtree index for snapping to the nearest segment (stable via WKB map)
seg_tree = STRtree(segments)
seg_wkb_to_idx = {seg.wkb: i for i, seg in enumerate(segments)}

def insert_snapped_node(Gbase: nx.Graph, pt_tm: Point):
    """
    Snap a point to its nearest network segment and split that segment,
    inserting a new node at the snapped location.
    Returns (G_copy, snapped_node, off_m).
    """
    idx = None
    try:
        geom = seg_tree.nearest(pt_tm)
        idx = seg_wkb_to_idx.get(geom.wkb)
    except Exception:
        idx = None

    if idx is None:
        # fallback: brute-force distance
        dists = [seg.distance(pt_tm) for seg in segments]
        idx = int(np.argmin(dists))

    u_str, v_str = seg_uv[idx]
    line = segments[idx]

    s = line.project(pt_tm)
    p_on = line.interpolate(s)
    pnode_str = snap_xy(p_on.x, p_on.y, SNAP_TOL)
    
    # Convert strings to coordinates for distance calculations
    u_coords = string_to_coords(u_str)
    v_coords = string_to_coords(v_str)
    pnode_coords = string_to_coords(pnode_str)
    
    off_m = float(pt_tm.distance(Point(pnode_coords)))

    # if very close to an endpoint, reuse it
    if Point(pnode_coords).distance(Point(u_coords)) < 0.5:
        return Gbase, u_str, off_m
    if Point(pnode_coords).distance(Point(v_coords)) < 0.5:
        return Gbase, v_str, off_m

    Gnew = Gbase.copy()
    if Gnew.has_edge(u_str, v_str):
        Gnew.remove_edge(u_str, v_str)

    up_len = float(LineString([u_coords, pnode_coords]).length)
    pv_len = float(LineString([pnode_coords, v_coords]).length)

    Gnew.add_node(u_str); Gnew.add_node(v_str); Gnew.add_node(pnode_str)
    Gnew.add_edge(u_str, pnode_str, weight=up_len)
    Gnew.add_edge(pnode_str, v_str, weight=pv_len)

    return Gnew, pnode_str, off_m

# ------------- UI: choose substations -------------
opts = subs_ll[sub_label].tolist()
src_name = st.selectbox("From substation", opts, index=0)
dst_name = st.selectbox("To substation",   opts, index=min(1, len(opts)-1))
btn = st.button("Compute path & point")

# ------------- Map base -------------
fig = go.Figure()

# draw HV lines (uniform color)
first = True
for geom in hvl_ll.geometry:
    if isinstance(geom, LineString):
        xs, ys = zip(*geom.coords)
        fig.add_trace(go.Scattermapbox(
            lon=list(xs), lat=list(ys), mode="lines",
            line=dict(width=ln_width, color=hv_color),
            name="HV Line", showlegend=first, hoverinfo="skip"
        ))
        first = False
    elif isinstance(geom, MultiLineString):
        for part in geom.geoms:
            xs, ys = zip(*part.coords)
            fig.add_trace(go.Scattermapbox(
                lon=list(xs), lat=list(ys), mode="lines",
                line=dict(width=ln_width, color=hv_color),
                name="HV Line", showlegend=first, hoverinfo="skip"
            ))
            first = False

# draw substations WITH LABELS
subs_ll["lat"] = subs_ll.geometry.y
subs_ll["lon"] = subs_ll.geometry.x
fig.add_trace(go.Scattermapbox(
    lat=subs_ll["lat"], lon=subs_ll["lon"], mode="markers+text",
    marker=dict(size=pt_size),
    text=subs_ll[sub_label],
    textposition="top center",
    name="Substation",
    hovertemplate="%{text}<br>Lat %{lat:.5f}, Lon %{lon:.5f}<extra></extra>"
))

# center map
minx, miny, maxx, maxy = hvl_ll.total_bounds if not hvl_ll.empty else subs_ll.total_bounds
center_lon = (minx + maxx)/2; center_lat = (miny + maxy)/2

fig.update_layout(
    mapbox_style="open-street-map",
    mapbox_center={"lat": center_lat, "lon": center_lon},
    mapbox_zoom=map_zoom,
    margin=dict(r=0, t=0, l=0, b=0),
    height=820,
    legend=dict(bgcolor="rgba(255,255,255,0.7)")
)

# ------------- Compute & draw path -------------
if btn and src_name != dst_name:
    src_tm = subs_tm.loc[subs_ll[sub_label] == src_name].geometry.iloc[0]
    dst_tm = subs_tm.loc[subs_ll[sub_label] == dst_name].geometry.iloc[0]

    # snap & insert both endpoints
    Gsnap, src_node, off_src = insert_snapped_node(G, src_tm)
    Gsnap, dst_node, off_dst = insert_snapped_node(Gsnap, dst_tm)

    try:
        # route along the snapped graph (follows HV polyline vertices; no fake crossings)
        path_nodes = nx.shortest_path(Gsnap, src_node, dst_node, weight="weight")
        path_len_m = float(nx.shortest_path_length(Gsnap, src_node, dst_node, weight="weight"))
        total_len_m = path_len_m + off_src + off_dst

        # Convert string nodes to coordinates for plotting
        path_coords = [string_to_coords(node) for node in path_nodes]
        
        # draw path (convert nodes from TM to WGS84)
        pts_tm_gdf = gpd.GeoDataFrame(
            geometry=[Point(x, y) for x, y in path_coords], crs=RWANDA_TM30
        ).to_crs(4326)
        path_lon = [pt.x for pt in pts_tm_gdf.geometry]
        path_lat = [pt.y for pt in pts_tm_gdf.geometry]
        fig.add_trace(go.Scattermapbox(
            lon=path_lon, lat=path_lat, mode="lines+markers",
            line=dict(width=max(ln_width+1, 4), color=path_color),
            name="Shortest path", hoverinfo="skip"
        ))

        # ---- point at requested distance from start (along the path) ----
        path_tm = LineString(path_coords)  # TM meters
        req_m = dist_value if unit == "meters" else dist_value * 1000.0
        req_m = max(0.0, min(req_m, float(path_tm.length)))

        pt_tm = path_tm.interpolate(req_m)
        pt_ll = gpd.GeoDataFrame(geometry=[pt_tm], crs=RWANDA_TM30).to_crs(4326).geometry.iloc[0]
        pop_lon, pop_lat = pt_ll.x, pt_ll.y

        # ---- Admin lookup ----
        admin_values = {}
        if admin_gdf is not None:
            pt_df = gpd.GeoDataFrame(geometry=[pt_ll], crs=4326)
            try:
                hit = gpd.sjoin(pt_df, admin_gdf, how="left", predicate="within")
                if not hit.empty:
                    col_village  = pick_first_present(admin_gdf, ["Village","VillageName"])
                    col_cell     = pick_first_present(admin_gdf, ["Cellule_1","Cell","Cellule"])
                    col_sector   = pick_first_present(admin_gdf, ["Sector_1","Sector"])
                    col_district = pick_first_present(admin_gdf, ["District"])
                    admin_values["Village"]  = str(hit.iloc[0].get(col_village))  if col_village  else None
                    admin_values["Cell"]     = str(hit.iloc[0].get(col_cell))     if col_cell     else None
                    admin_values["Sector"]   = str(hit.iloc[0].get(col_sector))   if col_sector   else None
                    admin_values["District"] = str(hit.iloc[0].get(col_district)) if col_district else None
            except Exception:
                pass

        hover_lines = [f"Lat {pop_lat:.6f}", f"Lon {pop_lon:.6f}", f"Dist {req_m/1000:.3f} km"]
        for nice in ["Village","Cell","Sector","District"]:
            val = admin_values.get(nice)
            if val and str(val).lower() not in ("none","nan"):
                hover_lines.append(f"{nice}: {val}")
        hover_text = "<br>".join(hover_lines)

        label_text = f"{req_m/1000:.3f} km from start" if unit == "kilometers" else f"{int(req_m)} m from start"
        fig.add_trace(go.Scattermapbox(
            lon=[pop_lon], lat=[pop_lat], mode="markers+text",
            marker=dict(size=max(pt_size, 10)),
            text=[label_text],
            textposition="top center",
            name="Point at distance",
            showlegend=True,
            hovertemplate=hover_text + "<extra></extra>"
        ))

        st.success(
            f"**Along HV lines:** {total_len_m/1000:.3f} km  "
            f"(network {path_len_m/1000:.3f} km + off-network {(off_src+off_dst)/1000:.3f} km)"
        )
        extra = "  •  ".join([f"{k}: {v}" for k, v in admin_values.items() if v])
        if extra:
            st.info(f"Location context → {extra}")

    except nx.NetworkXNoPath:
        st.error("No path between the chosen substations in the HV line graph. Check topology or gaps.")
    except Exception as e:
        st.error(f"Error computing path: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
else:
    if btn:
        st.info("Pick two different substations.")

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Tip: Keep shapefile sidecars together (.shp/.shx/.dbf/.prj). Missing CRS assumes Rwanda TM; "
    "display is in WGS84. The graph uses each HV polyline's own vertices and only connects at "
    "shared endpoints (with a small snap tolerance), so the route follows the actual lines."
)