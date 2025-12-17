# energy_paths.py
import os, glob
from pathlib import Path
import sys

import streamlit as st
import pandas as pd
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

# ============== DEBUG SECTION ==============
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 DEBUG: File Discovery")
st.sidebar.write(f"Python: {sys.version.split()[0]}")
st.sidebar.write(f"Current dir: {os.getcwd()}")
st.sidebar.write(f"Base dir: {BASE_DIR}")
st.sidebar.write(f"Root input: {root}")

# List ALL files in current directory
st.sidebar.write("**All files in directory:**")
try:
    all_files = os.listdir('.')
    for i, file in enumerate(sorted(all_files)[:15], 1):
        if os.path.exists(file):
            size = os.path.getsize(file)
            size_str = f"{size/1024/1024:.1f} MB" if size > 1000000 else f"{size/1024:.0f} KB"
            st.sidebar.write(f"{i:2}. {file} ({size_str})")
            if 'hv' in file.lower() and 'line' in file.lower():
                st.sidebar.write(f"   ⚡ HV LINE DETECTED!")
            if 'sub' in file.lower() and 'station' in file.lower():
                st.sidebar.write(f"   ⚡ SUBSTATION DETECTED!")
        else:
            st.sidebar.write(f"{i:2}. {file} (does not exist)")
    if len(all_files) > 15:
        st.sidebar.write(f"... and {len(all_files)-15} more files")
except Exception as e:
    st.sidebar.error(f"Error listing files: {e}")

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
    """Simple case-insensitive file finder."""
    keyword_lower = keyword.lower()
    matches = []
    
    try:
        files = os.listdir(folder)
        for file in files:
            if keyword_lower in file.lower():
                full_path = os.path.join(folder, file)
                if os.path.isfile(full_path):
                    if any(file.lower().endswith(ext) for ext in ['.shp', '.csv', '.geojson', '.fgb', '.json']):
                        matches.append(full_path)
    except Exception as e:
        st.sidebar.warning(f"Could not search {folder}: {e}")
    
    return matches[:1]

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

# ============== EMERGENCY FILE LOADING ==============
st.sidebar.markdown("---")
st.sidebar.subheader("📂 Loading Files")

def try_direct_load(filename, description):
    """Try loading a file directly with multiple path attempts."""
    possible_paths = [
        filename,
        os.path.join('.', filename),
        os.path.join(BASE_DIR, filename),
        filename.lower(),
        filename.upper(),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            st.sidebar.success(f"✅ {description}: Found at '{path}'")
            return path
    return None

# Try to find HV line
hv_path = try_direct_load("HVLine.shp", "HV Line")
if not hv_path:
    hv_path = try_direct_load("hvline.shp", "HV Line (lowercase)")
if not hv_path:
    hv_path = try_direct_load("HV_Lines.shp", "HV Lines")
if not hv_path:
    st.sidebar.write("⚠️ Trying search method...")
    hv_files = find_file(root, "hvline")
    if hv_files:
        hv_path = hv_files[0]
    else:
        st.error("❌ **No HV line file found!**")
        st.sidebar.error("Available files that contain 'hv':")
        for file in os.listdir('.'):
            if 'hv' in file.lower():
                st.sidebar.write(f"  - {file}")
        st.stop()

# Try to find substation
sub_path = try_direct_load("Substation.csv", "Substation")
if not sub_path:
    sub_path = try_direct_load("substation.csv", "Substation (lowercase)")
if not sub_path:
    st.sidebar.write("⚠️ Trying search method...")
    sub_files = find_file(root, "substation")
    if sub_files:
        sub_path = sub_files[0]
    else:
        st.error("❌ **No substation file found!**")
        st.sidebar.error("Available files that contain 'sub':")
        for file in os.listdir('.'):
            if 'sub' in file.lower():
                st.sidebar.write(f"  - {file}")
        st.stop()

# Verify shapefile components
st.sidebar.write("**Checking HVLine shapefile components:**")
required_exts = ['.shp', '.shx', '.dbf', '.prj']
base_name = os.path.splitext(hv_path)[0]
missing = []
for ext in required_exts:
    component = base_name + ext
    if os.path.exists(component):
        size = os.path.getsize(component)
        size_str = f"{size/1024:.0f} KB"
        st.sidebar.write(f"✅ {os.path.basename(component)} ({size_str})")
    else:
        st.sidebar.write(f"❌ {os.path.basename(component)} - MISSING")
        missing.append(ext)

if missing:
    st.warning(f"Missing shapefile components: {', '.join(missing)}")
    st.sidebar.warning("Shapefile may not load properly!")

# ============== LOAD THE FILES ==============
try:
    st.caption(f"Using **Substations:** {os.path.basename(sub_path)}  •  **HV Lines:** {os.path.basename(hv_path)}")
    
    st.sidebar.write("⏳ Loading substations...")
    
    # Check if it's CSV and load differently
    if sub_path.lower().endswith('.csv'):
        # Load as regular CSV first
        subs_df = pd.read_csv(sub_path)
        st.sidebar.write(f"CSV columns: {list(subs_df.columns)}")
        
        # Look for coordinate columns
        lat_col = None
        lon_col = None
        
        for col in subs_df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['lat', 'latitude', 'y']):
                lat_col = col
            if any(keyword in col_lower for keyword in ['lon', 'longitude', 'x', 'lng']):
                lon_col = col
        
        if lat_col and lon_col:
            # Create geometry column
            geometry = [Point(xy) for xy in zip(subs_df[lon_col], subs_df[lat_col])]
            subs = gpd.GeoDataFrame(subs_df, geometry=geometry, crs='EPSG:4326')
            st.sidebar.success(f"✅ Created geometry from {lat_col}/{lon_col}")
        else:
            # Try to find any numeric columns
            st.sidebar.warning("No standard coordinate columns found. Trying alternatives...")
            numeric_cols = subs_df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                # Use first two numeric columns as coordinates
                lon_col, lat_col = numeric_cols[0], numeric_cols[1]
                geometry = [Point(xy) for xy in zip(subs_df[lon_col], subs_df[lat_col])]
                subs = gpd.GeoDataFrame(subs_df, geometry=geometry, crs='EPSG:4326')
                st.sidebar.warning(f"⚠️ Using numeric columns {lon_col}/{lat_col} as coordinates")
            else:
                st.error("❌ Could not find coordinate columns in CSV!")
                st.sidebar.error("File structure:")
                st.sidebar.write(subs_df.head())
                st.stop()
    else:
        # Load as normal GIS file
        subs = gpd.read_file(sub_path)
    
    st.sidebar.success(f"✅ Loaded {len(subs)} substations")
    
    # Load HV lines (should be shapefile)
    st.sidebar.write("⏳ Loading HV lines...")
    hvl = gpd.read_file(hv_path)
    st.sidebar.success(f"✅ Loaded {len(hvl)} HV line features")
    
except Exception as e:
    st.error(f"❌ Error loading files: {str(e)}")
    st.sidebar.error(f"Load error: {e}")
    import traceback
    st.code(traceback.format_exc())
    st.stop()

# Check if geometry column exists
if not hasattr(subs, 'geometry'):
    st.error("❌ Substation file has no geometry column!")
    st.sidebar.write("Substation DataFrame info:")
    st.sidebar.write(f"Type: {type(subs)}")
    st.sidebar.write(f"Columns: {list(subs.columns)}")
    st.stop()

if not hasattr(hvl, 'geometry'):
    st.error("❌ HV line file has no geometry column!")
    st.stop()

# drop empties
if hasattr(subs, 'geometry'):
    subs = subs[subs.geometry.notna() & (~subs.geometry.is_empty)].copy()
if hasattr(hvl, 'geometry'):
    hvl = hvl[hvl.geometry.notna() & (~hvl.geometry.is_empty)].copy()

# keep only points / lines
subs = subs[subs.geometry.type.isin(["Point","MultiPoint"])].explode(index_parts=False, ignore_index=True)
subs = subs[subs.geometry.type == "Point"].copy()
hvl  = hvl[hvl.geometry.type.isin(["LineString","MultiLineString"])].copy()

# substation labels
if "Substation" not in subs.columns:
    st.warning("Column 'Substation' not found. Creating default labels...")
    subs["Substation"] = [f"Sub_{i+1}" for i in range(len(subs))]
else:
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
def snap_xy(x, y, tol):
    if tol <= 0:
        return f"{round(x, 3)},{round(y, 3)}"
    qx = round(x / tol) * tol
    qy = round(y / tol) * tol
    return f"{round(qx, 3)},{round(qy, 3)}"

def string_to_coords(node_str):
    if isinstance(node_str, tuple):
        return node_str
    try:
        x_str, y_str = node_str.split(',')
        return (float(x_str), float(y_str))
    except:
        if isinstance(node_str, (tuple, list)) and len(node_str) == 2:
            return (float(node_str[0]), float(node_str[1]))
        return (0.0, 0.0)

G = nx.Graph()
segments = []
seg_uv   = []

def add_polyline_segments(geom):
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
if len(segments) > 0:
    seg_tree = STRtree(segments)
    seg_wkb_to_idx = {seg.wkb: i for i, seg in enumerate(segments)}
else:
    st.error("No segments in HV lines graph!"); st.stop()

def insert_snapped_node(Gbase: nx.Graph, pt_tm: Point):
    idx = None
    try:
        geom = seg_tree.nearest(pt_tm)
        idx = seg_wkb_to_idx.get(geom.wkb)
    except Exception:
        idx = None

    if idx is None:
        dists = [seg.distance(pt_tm) for seg in segments]
        idx = int(np.argmin(dists))

    u_str, v_str = seg_uv[idx]
    line = segments[idx]

    s = line.project(pt_tm)
    p_on = line.interpolate(s)
    pnode_str = snap_xy(p_on.x, p_on.y, SNAP_TOL)
    
    u_coords = string_to_coords(u_str)
    v_coords = string_to_coords(v_str)
    pnode_coords = string_to_coords(pnode_str)
    
    off_m = float(pt_tm.distance(Point(pnode_coords)))

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

# draw HV lines (uniform color) - FIXED: Changed Scattermap to Scattermapbox
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

# draw substations WITH LABELS - FIXED: Changed Scattermap to Scattermapbox
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

    Gsnap, src_node, off_src = insert_snapped_node(G, src_tm)
    Gsnap, dst_node, off_dst = insert_snapped_node(Gsnap, dst_tm)

    try:
        path_nodes = nx.shortest_path(Gsnap, src_node, dst_node, weight="weight")
        path_len_m = float(nx.shortest_path_length(Gsnap, src_node, dst_node, weight="weight"))
        total_len_m = path_len_m + off_src + off_dst

        path_coords = [string_to_coords(node) for node in path_nodes]
        
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

        path_tm = LineString(path_coords)
        req_m = dist_value if unit == "meters" else dist_value * 1000.0
        req_m = max(0.0, min(req_m, float(path_tm.length)))

        pt_tm = path_tm.interpolate(req_m)
        pt_ll = gpd.GeoDataFrame(geometry=[pt_tm], crs=RWANDA_TM30).to_crs(4326).geometry.iloc[0]
        pop_lon, pop_lat = pt_ll.x, pt_ll.y

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

# Updated to fix the deprecation warning
st.plotly_chart(fig, width='stretch')

st.caption(
    "Tip: Keep shapefile sidecars together (.shp/.shx/.dbf/.prj). Missing CRS assumes Rwanda TM; "
    "display is in WGS84. The graph uses each HV polyline's own vertices and only connects at "
    "shared endpoints (with a small snap tolerance), so the route follows the actual lines."
)