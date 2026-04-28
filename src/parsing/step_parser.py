"""pythonocc-core STEP parsing module.

Converts a STEP file into face-dict and edge-dict lists consumable by
``graph_builder.build_data_object``.

Node features produced (9-dim, STEP path):
    [surface_type (0–5), area, normal_x, normal_y, normal_z,
     cylinder_radius, cylinder_axis_z,
     num_adjacent_faces, num_boundary_edges]
    ``area`` and ``cylinder_radius`` are normalised by the bounding-box diagonal.

Edge features produced (3-dim):
    [dihedral_angle (radians), convexity (+1 convex / −1 concave / 0 smooth),
     edge_length (normalised by bounding-box diagonal)]

Requires pythonocc-core 7.7.x installed via conda-forge:
    conda install -c conda-forge pythonocc-core=7.7.2
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# ── Convexity thresholds ──────────────────────────────────────────────────────
_SMOOTH_CROSS_THRESH = 1e-4   # |n1 × n2| below this → smooth
_SMOOTH_ANGLE_THRESH = 0.017  # ~1° in radians — seam / tangent continuation


# ── OCC imports (deferred so the module is importable without pythonocc) ─────

def _require_occ():
    try:
        import OCC.Core.STEPControl  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "pythonocc-core is required for STEP parsing. "
            "Install via: conda install -c conda-forge pythonocc-core=7.7.2"
        ) from exc


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_step(filepath: str):
    """Read a STEP file and return a TopoDS_Shape."""
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.IFSelect import IFSelect_RetDone

    reader = STEPControl_Reader()
    status = reader.ReadFile(str(filepath))
    if status != IFSelect_RetDone:
        raise IOError(f"STEPControl_Reader failed on '{filepath}' (status={status})")
    reader.TransferRoots()
    shape = reader.OneShape()
    return shape


def _bounding_box_diagonal(shape) -> float:
    """Return the diagonal length of the axis-aligned bounding box."""
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib

    bbox = Bnd_Box()
    brepbndlib.Add(shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    diag = math.sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2)
    return max(diag, 1e-6)


def _surface_type_str(adaptor) -> str:
    """Map BRepAdaptor_Surface type to a string key."""
    from OCC.Core.GeomAbs import (
        GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
        GeomAbs_Sphere, GeomAbs_Torus,
    )
    t = adaptor.GetType()
    return {
        GeomAbs_Plane:    "PLANE",
        GeomAbs_Cylinder: "CYLINDER",
        GeomAbs_Cone:     "CONE",
        GeomAbs_Sphere:   "SPHERE",
        GeomAbs_Torus:    "TORUS",
    }.get(t, "OTHER")


def _face_properties(face, diag: float) -> Dict:
    """Compute area, centroid, outward normal, cylinder params for one face."""
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.BRepTools import breptools
    from OCC.Core.BRepLProp import BRepLProp_SLProps
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.TopAbs import TopAbs_REVERSED
    from OCC.Core.GeomAbs import GeomAbs_Cylinder

    adaptor = BRepAdaptor_Surface(face)
    stype = _surface_type_str(adaptor)

    # Area + centroid
    props = GProp_GProps()
    brepgprop.SurfaceProperties(face, props)
    area = props.Mass() / max(diag ** 2, 1e-12)   # normalise by diag²
    cog = props.CentreOfMass()

    # Outward normal at surface centre
    umin, umax, vmin, vmax = breptools.UVBounds(face)
    u_mid = 0.5 * (umin + umax)
    v_mid = 0.5 * (vmin + vmax)
    surface = BRep_Tool.Surface(face)
    sl_props = BRepLProp_SLProps(surface, u_mid, v_mid, 1, 1e-6)

    nx, ny, nz = 0.0, 0.0, 0.0
    if sl_props.IsNormalDefined():
        n = sl_props.Normal()
        # Flip if face orientation is reversed (outward = into solid)
        if face.Orientation() == TopAbs_REVERSED:
            nx, ny, nz = -n.X(), -n.Y(), -n.Z()
        else:
            nx, ny, nz = n.X(), n.Y(), n.Z()

    # Cylinder-specific params
    cyl_radius = 0.0
    cyl_axis_z = 0.0
    cyl_axis = [0.0, 0.0, 1.0]
    if adaptor.GetType() == GeomAbs_Cylinder:
        cyl = adaptor.Cylinder()
        cyl_radius = cyl.Radius() / max(diag, 1e-6)
        ax = cyl.Axis().Direction()
        cyl_axis = [ax.X(), ax.Y(), ax.Z()]
        cyl_axis_z = ax.Z()

    return {
        "surface_type":   stype,
        "area":           area,
        "centroid_x":     cog.X(),
        "centroid_y":     cog.Y(),
        "centroid_z":     cog.Z(),
        "normal_x":       nx,
        "normal_y":       ny,
        "normal_z":       nz,
        "cylinder_radius": cyl_radius,
        "cylinder_axis":  cyl_axis,
        "cylinder_axis_z": cyl_axis_z,
    }


def _edge_midpoint_and_tangent(edge):
    """Return (midpoint gp_Pnt, tangent gp_Vec) at the parametric midpoint."""
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
    from OCC.Core.gp import gp_Pnt, gp_Vec

    curve = BRepAdaptor_Curve(edge)
    t_first = curve.FirstParameter()
    t_last  = curve.LastParameter()
    t_mid   = 0.5 * (t_first + t_last)

    pt  = gp_Pnt()
    vec = gp_Vec()
    curve.D1(t_mid, pt, vec)
    return pt, vec


def _project_point_to_face_uv(face, point):
    """Project a 3-D point onto a face surface, return (u, v) or None."""
    from OCC.Core.ShapeAnalysis import ShapeAnalysis_Surface
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.gp import gp_Pnt2d

    surface = BRep_Tool.Surface(face)
    sa = ShapeAnalysis_Surface(surface)
    uv = sa.ValueOfUV(point, 1e-4)
    return uv.X(), uv.Y()


def _face_outward_normal_at_uv(face, u: float, v: float):
    """Return outward-pointing gp_Dir at (u, v), accounting for orientation."""
    from OCC.Core.BRepLProp import BRepLProp_SLProps
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.TopAbs import TopAbs_REVERSED
    from OCC.Core.gp import gp_Vec

    surface = BRep_Tool.Surface(face)
    sl = BRepLProp_SLProps(surface, u, v, 1, 1e-6)
    if not sl.IsNormalDefined():
        return None
    n = sl.Normal()
    if face.Orientation() == TopAbs_REVERSED:
        return gp_Vec(-n.X(), -n.Y(), -n.Z())
    return gp_Vec(n.X(), n.Y(), n.Z())


def _convexity(n1, n2, tangent) -> float:
    """Compute convexity sign from outward face normals and edge tangent.

    Convention (matching MFCAD++ E_1/E_2/E_3):
      +1  convex  — (n1 × n2) · tangent < 0
      −1  concave — (n1 × n2) · tangent > 0
       0  smooth  — |n1 × n2| < threshold or dihedral ≈ π
    """
    cross = n1.Crossed(n2)
    mag = cross.Magnitude()
    if mag < _SMOOTH_CROSS_THRESH:
        return 0.0
    dot = cross.Dot(tangent)
    if abs(dot) < _SMOOTH_CROSS_THRESH:
        return 0.0
    return -1.0 if dot > 0 else 1.0


def _dihedral_angle(n1, n2) -> float:
    """Interior dihedral angle between two faces (radians, 0–π)."""
    dot = n1.Dot(n2)
    dot = max(-1.0, min(1.0, dot))
    # Outward normals of adjacent faces point AWAY from each other for convex
    # edges (dot < 0), TOWARD each other for concave (dot > 0).
    # The interior dihedral angle = π − arccos(dot).
    return math.pi - math.acos(dot)


def _edge_length(edge, diag: float) -> float:
    """Return normalised edge arc length."""
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.GProp import GProp_GProps

    props = GProp_GProps()
    brepgprop.LinearProperties(edge, props)
    return props.Mass() / max(diag, 1e-6)


# ── Main parsing pipeline ─────────────────────────────────────────────────────

def parse_step_file(
    filepath: str,
) -> Tuple[List[Dict], List[Dict]]:
    """Parse a STEP file into face-dicts and edge-dicts.

    Args:
        filepath: Path to a .step or .stp file.

    Returns:
        faces: List[Dict] — one dict per B-Rep face, keys:
            face_id, occ_face_id, surface_type, area, centroid_{x,y,z},
            normal_{x,y,z}, cylinder_radius, cylinder_axis, cylinder_axis_z,
            num_adjacent_faces, num_boundary_edges
        edges: List[Dict] — one dict per shared (manifold) edge, keys:
            src, dst, dihedral_angle, convexity, edge_length
    """
    _require_occ()

    from OCC.Core.TopExp import TopExp_Explorer, topexp
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
    from OCC.Core.TopTools import (
        TopTools_IndexedMapOfShape,
        TopTools_IndexedDataMapOfShapeListOfShape,
    )
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.TopoDS import topods

    filepath = str(filepath)
    log.info("Parsing STEP file: %s", filepath)

    shape = _load_step(filepath)
    diag  = _bounding_box_diagonal(shape)
    log.debug("Bounding box diagonal: %.4f", diag)

    # ── Index all faces ───────────────────────────────────────────────────────
    face_map = TopTools_IndexedMapOfShape()
    topexp.MapShapes(shape, TopAbs_FACE, face_map)
    n_faces = face_map.Size()
    log.info("Found %d faces", n_faces)

    # ── Index edge → adjacent faces ───────────────────────────────────────────
    edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
    topexp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_face_map)

    # ── Count adjacency and boundary edges per face ───────────────────────────
    # adj_count[face_idx] = number of manifold neighbour faces
    # boundary_count[face_idx] = number of boundary (naked) edges
    adj_count      = [0] * (n_faces + 1)
    boundary_count = [0] * (n_faces + 1)

    edge_map = TopTools_IndexedMapOfShape()
    topexp.MapShapes(shape, TopAbs_EDGE, edge_map)

    for edge_idx in range(1, edge_map.Size() + 1):
        edge = topods.Edge(edge_map.FindKey(edge_idx))
        face_list = edge_face_map.FindFromKey(edge)
        n_adj = face_list.Size()
        if n_adj == 2:
            f1_idx = face_map.FindIndex(face_list.First())
            f2_idx = face_map.FindIndex(face_list.Last())
            adj_count[f1_idx] += 1
            adj_count[f2_idx] += 1
        elif n_adj == 1:
            fi = face_map.FindIndex(face_list.First())
            boundary_count[fi] += 1

    # ── Build face dicts ──────────────────────────────────────────────────────
    faces: List[Dict] = []
    for fi in range(1, n_faces + 1):
        face = topods.Face(face_map.FindKey(fi))
        try:
            props = _face_properties(face, diag)
        except Exception as exc:
            log.warning("Face %d: property extraction failed (%s); using defaults", fi, exc)
            props = {
                "surface_type": "OTHER", "area": 0.0,
                "centroid_x": 0.0, "centroid_y": 0.0, "centroid_z": 0.0,
                "normal_x": 0.0, "normal_y": 0.0, "normal_z": 0.0,
                "cylinder_radius": 0.0, "cylinder_axis": [0.0, 0.0, 1.0],
                "cylinder_axis_z": 0.0,
            }
        props["face_id"]            = fi - 1          # 0-based integer index
        props["occ_face_id"]        = f"#{fi}"        # 1-based STEP-style ID
        props["num_adjacent_faces"] = adj_count[fi]
        props["num_boundary_edges"] = boundary_count[fi]
        faces.append(props)

    # ── Build edge dicts (manifold shared edges only) ─────────────────────────
    edges: List[Dict] = []
    for edge_idx in range(1, edge_map.Size() + 1):
        edge = topods.Edge(edge_map.FindKey(edge_idx))

        face_list = edge_face_map.FindFromKey(edge)
        if face_list.Size() != 2:
            continue   # boundary or non-manifold edge

        face1 = topods.Face(face_list.First())
        face2 = topods.Face(face_list.Last())
        fi1   = face_map.FindIndex(face_list.First()) - 1   # 0-based
        fi2   = face_map.FindIndex(face_list.Last())  - 1

        try:
            mid_pt, tangent = _edge_midpoint_and_tangent(edge)
            u1, v1 = _project_point_to_face_uv(face1, mid_pt)
            u2, v2 = _project_point_to_face_uv(face2, mid_pt)
            n1 = _face_outward_normal_at_uv(face1, u1, v1)
            n2 = _face_outward_normal_at_uv(face2, u2, v2)

            if n1 is None or n2 is None:
                conv  = 0.0
                angle = 0.0
            else:
                conv  = _convexity(n1, n2, tangent)
                angle = _dihedral_angle(n1, n2)

            length = _edge_length(edge, diag)
        except Exception as exc:
            log.debug("Edge %d: feature extraction failed (%s); using defaults", edge_idx, exc)
            conv, angle, length = 0.0, 0.0, 0.0

        # Store both directions (undirected graph)
        for src, dst in [(fi1, fi2), (fi2, fi1)]:
            edges.append({
                "src":           src,
                "dst":           dst,
                "dihedral_angle": angle,
                "convexity":     conv,
                "edge_length":   length,
            })

    log.info("Parsed %d faces, %d directed edges", len(faces), len(edges))
    return faces, edges


def step_to_graph(filepath: str, model_id: Optional[str] = None):
    """Parse a STEP file and return a PyG Data object ready for inference.

    Convenience wrapper: parse_step_file → merge_seam_faces → build_data_object.

    Args:
        filepath : Path to .step / .stp file.
        model_id : Optional identifier stored in Data.model_id.

    Returns:
        torch_geometric.data.Data with x (9-dim), edge_index, edge_attr,
        face_ids, occ_face_ids.
    """
    from src.parsing.graph_builder import build_data_object

    if model_id is None:
        model_id = Path(filepath).stem

    faces, edges = parse_step_file(filepath)
    return build_data_object(faces, edges, model_id=model_id)
