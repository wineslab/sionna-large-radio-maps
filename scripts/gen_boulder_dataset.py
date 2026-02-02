#!/usr/bin/env python3
"""
Boulder, CO Coverage Map Dataset Generator

Generates path gain and elevation maps by dynamically building 3D scenes 
from OSM buildings + AWS elevation tiles, then ray tracing with Sionna RT.
"""

import os
import gc
import time
import logging
import shutil
import warnings
import uuid
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from xml.etree import ElementTree as ET
from xml.dom import minidom

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image
from tqdm import tqdm

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import pyproj
from pyproj import Transformer
import osmnx as ox
import open3d as o3d
from shapely.geometry import Point

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import drjit as dr
import mitsuba as mi

# Sionna RT requires mono_polarized variant
try:
    mi.set_variant("cuda_ad_mono_polarized")
    print("Using CUDA mono_polarized variant")
except Exception:
    mi.set_variant("llvm_ad_mono_polarized")
    print("Using LLVM mono_polarized variant")

import sionna.rt as rt


# ==================== Configuration ====================

# Boulder, CO bounds
BOULDER_LAT = (39.97, 40.07)
BOULDER_LON = (-105.31, -105.20)

# Scene parameters
RADIUS_RANGE = (500, 2500)  # Scene radius in meters
RESOLUTION_RATIO = 100      # cell_size = radius/ratio -> coverage map is 2*ratio cells (200x200)
ZOOM_LEVEL = 14             # Elevation tile zoom

# Ray tracing
FREQUENCY = 1e9           # 3.5 GHz
MAX_DEPTH = 10
NUM_SAMPLES = int(5e7)
DIFFRACTION = True

# Output
NUM_SAMPLES_TO_GENERATE = 1000
WORKERS_PER_GPU = 1         # Number of workers per GPU
GPU_MEMORY_FRACTION = 0.20  # GPU memory fraction per worker (0.20 = 20%)
OUTPUT_DIR = Path(f"/output/boulder_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
GRID_SIZE = 200             # Output grid size
PLOT = False
MIN_BUILDINGS = 10           # Minimum buildings required per sample

# Heights
TX_HEIGHT = 25.0            # TX height above terrain (meters)
RX_HEIGHT = 1.5             # RX height above terrain (meters, standard pedestrian UE height)
MAX_ELEVATION_RANGE = 200   # Maximum terrain elevation range (meters) - skip samples with more

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==================== HeightMap ====================

class HeightMap:
    """Downloads elevation data from AWS S3 public tiles."""
    
    def __init__(self, utm_zone, bbox, z=14):
        self.s3 = boto3.client("s3", config=Config(
            signature_version=UNSIGNED,
            region_name="us-east-1",
            retries={"max_attempts": 2, "mode": "adaptive"},
        ))
        
        min_lon, min_lat, max_lon, max_lat = bbox
        x0, x1 = int(self._lon2tile(min_lon, z)), int(self._lon2tile(max_lon, z))
        y0, y1 = int(self._lat2tile(max_lat, z)), int(self._lat2tile(min_lat, z))
        
        self.to_utm = Transformer.from_crs("EPSG:4326", utm_zone, always_xy=True)
        self.to_wgs84 = Transformer.from_crs(utm_zone, "EPSG:4326", always_xy=True)
        
        # Download tiles
        tex = np.zeros(((y1-y0+1)*512, (x1-x0+1)*512, 1))
        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = [ex.submit(self._get_tile, z, x0+c, y0+r) 
                      for r in range(y1-y0+1) for c in range(x1-x0+1)]
            for f in tqdm(futures, desc="Downloading elevation tiles"):
                x, y, tile = f.result()
                tex[(y-y0)*512:(y-y0+1)*512, (x-x0)*512:(x-x0+1)*512, 0] = tile
        
        self.heightmap = mi.load_dict({"type": "bitmap", "data": mi.TensorXf(tex), 
                                       "raw": True, "filter_type": "bilinear"})
        self.bounds = (self._tile2lon(x0, z), self._tile2lat(y0, z),
                       self._tile2lon(x1+1, z), self._tile2lat(y1+1, z))
    
    def _get_tile(self, z, x, y):
        buf = BytesIO()
        self.s3.download_fileobj("elevation-tiles-prod", f"geotiff/{z}/{x}/{y}.tif", buf)
        buf.seek(0)
        im = np.array(Image.open(buf), dtype=np.float32)
        if im.shape != (512, 512):
            im = mi.TensorXf(mi.Bitmap(im).resample(mi.ScalarVector2u(512))).numpy().squeeze()
        # Handle invalid pixels
        if np.any((im < -1e6) | np.isnan(im) | (im > 1e4)):
            im = np.nan_to_num(im, nan=0, posinf=0, neginf=0)
            im = np.clip(im, 0, 5000)
        return x, y, im
    
    def height_from_utm(self, x, y):
        lon, lat = self.to_wgs84.transform(x, y)
        u = (lon - self.bounds[0]) / (self.bounds[2] - self.bounds[0])
        v = (lat - self.bounds[1]) / (self.bounds[3] - self.bounds[1])
        si = dr.zeros(mi.SurfaceInteraction3f)
        si.uv = mi.Point2f(mi.Float(u), mi.Float(v))
        dr.make_opaque(si)
        return self.heightmap.eval_1(si)
    
    @staticmethod
    def _lat2tile(lat, z):
        return np.floor(2**z * 0.5 * (1 - np.arcsinh(np.tan(np.radians(lat))) / np.pi))
    
    @staticmethod
    def _lon2tile(lon, z):
        return np.floor(2**z * 0.5 * (1 + np.radians(lon) / np.pi))
    
    @staticmethod
    def _tile2lon(x, z):
        return x / 2**z * 360 - 180
    
    @staticmethod
    def _tile2lat(y, z):
        return np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * y / 2**z))))


def get_utm_crs(lon, lat):
    """Get UTM CRS for given coordinates."""
    crs_list = pyproj.database.query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=pyproj.aoi.AreaOfInterest(lon, lat, lon, lat),
    )
    return pyproj.CRS.from_epsg(crs_list[0].code)


# ==================== Scene Builder ====================

class SceneBuilder:
    """Builds Sionna RT scenes from OSM + elevation data."""
    
    def __init__(self, center, radius, cache_dir):
        self.center = center  # (lat, lon)
        self.radius = radius
        self.cache_dir = Path(cache_dir)
        self.resolution = radius / RESOLUTION_RATIO
        
        lat, lon = center
        self.utm_crs = get_utm_crs(lon, lat)
        self.to_utm = Transformer.from_crs("EPSG:4326", self.utm_crs, always_xy=True)
        self.to_wgs84 = Transformer.from_crs(self.utm_crs, "EPSG:4326", always_xy=True)
        
        # Calculate bbox
        lat_off = radius / 111320.0
        lon_off = radius / (111320.0 * np.cos(np.radians(lat)))
        self.bbox = (lat - lat_off, lon - lon_off, lat + lat_off, lon + lon_off)
        
        # Load heightmap
        min_lat, min_lon, max_lat, max_lon = self.bbox
        self.heightmap = HeightMap(self.utm_crs, (min_lon, min_lat, max_lon, max_lat), ZOOM_LEVEL)
        
        # Calculate origin and z_offset for local coordinate system
        self.origin = self.to_utm.transform(min_lon, min_lat)
        self._compute_z_offset()
        
        # Build and load scene
        self._build_scene()
        self.scene = rt.load_scene(str(self.scene_path))
        self.scene.frequency = FREQUENCY
        self._setup_antennas()
        self._add_transmitter()
    
    def _compute_z_offset(self):
        """Compute minimum terrain elevation to use as Z offset and elevation range."""
        min_lat, min_lon, max_lat, max_lon = self.bbox
        bl = self.to_utm.transform(min_lon, min_lat)
        tr = self.to_utm.transform(max_lon, max_lat)
        w, h = tr[0] - bl[0], tr[1] - bl[1]
        
        # Sample terrain at grid points
        nx, ny = 20, 20
        xx, yy = np.meshgrid(np.linspace(0, w, nx), np.linspace(0, h, ny))
        heights = self.heightmap.height_from_utm(self.origin[0] + xx.flatten(), self.origin[1] + yy.flatten())
        if hasattr(heights, 'numpy'):
            heights = heights.numpy()
        self.z_offset = float(np.nanmin(heights))
        self.elevation_range = float(np.nanmax(heights) - np.nanmin(heights))
        logger.info(f"Z offset: {self.z_offset:.1f}m, elevation range: {self.elevation_range:.1f}m")
    
    def _build_scene(self):
        """Create Mitsuba XML scene with meshes."""
        from triangle import triangulate
        
        min_lat, min_lon, max_lat, max_lon = self.bbox
        scene_dir = self.cache_dir / f"scene_{hash(self.center)}"
        mesh_dir = scene_dir / "mesh"
        mesh_dir.mkdir(parents=True, exist_ok=True)
        self.scene_path = scene_dir / "scene.xml"
        
        # Create XML
        xml = ET.Element("scene", version="2.1.0")
        ET.SubElement(xml, "integrator", type="path", id="integrator")
        
        # Materials
        mat = ET.SubElement(xml, "bsdf", type="itu-radio-material", id="mat-ground")
        ET.SubElement(mat, "string", name="type", value="medium_dry_ground")
        ET.SubElement(mat, "float", name="thickness", value="1.0")
        
        mat = ET.SubElement(xml, "bsdf", type="itu-radio-material", id="mat-concrete")
        ET.SubElement(mat, "string", name="type", value="concrete")
        ET.SubElement(mat, "float", name="thickness", value="0.2")
        
        ET.SubElement(xml, "emitter", type="constant", id="light")
        
        # Unique prefix for this scene (avoids "name already used" errors)
        uid = uuid.uuid4().hex[:8]
        
        # Ground mesh (with z_offset subtracted)
        verts, faces = self._make_ground()
        self._write_ply(mesh_dir / "ground.ply", verts, faces)
        self._add_shape(xml, f"terrain_{uid}", "mesh/ground.ply", "mat-ground")
        
        # Buildings
        self.building_count = 0
        try:
            buildings = ox.features.features_from_bbox(
                bbox=(min_lon, min_lat, max_lon, max_lat), tags={"building": True}
            ).to_crs(self.utm_crs)
            
            verts, faces = self._make_buildings(buildings)
            if len(verts) > 0 and len(faces) > 0:
                # Validate mesh data
                if np.any(np.isnan(verts)) or np.any(np.isinf(verts)):
                    logger.warning("Invalid vertices in building mesh, skipping")
                else:
                    self._write_ply(mesh_dir / "buildings.ply", verts, faces)
                    self._add_shape(xml, f"bldgs_{uid}", "mesh/buildings.ply", "mat-concrete")
                    self.building_count = len(buildings)
                    logger.info(f"Added {self.building_count} buildings")
        except Exception as e:
            logger.warning(f"No buildings: {e}")
        
        # Write XML
        with open(self.scene_path, "w") as f:
            f.write(minidom.parseString(ET.tostring(xml)).toprettyxml(indent="  "))
    
    def _make_ground(self, res=10):
        """Create ground mesh grid with normalized elevations."""
        min_lat, min_lon, max_lat, max_lon = self.bbox
        bl = self.to_utm.transform(min_lon, min_lat)
        tr = self.to_utm.transform(max_lon, max_lat)
        w, h = tr[0] - bl[0], tr[1] - bl[1]
        
        nx, ny = max(10, int(w/res)), max(10, int(h/res))
        xx, yy = np.meshgrid(np.linspace(0, w, nx), np.linspace(0, h, ny))
        
        heights = self.heightmap.height_from_utm(self.origin[0] + xx.flatten(), self.origin[1] + yy.flatten())
        if hasattr(heights, 'numpy'):
            heights = heights.numpy()
        heights = np.array(heights).reshape(xx.shape)
        
        # Subtract z_offset to normalize elevations (local Z=0 at min terrain)
        heights = heights - self.z_offset
        
        verts = np.stack([xx, yy, heights], axis=-1).reshape(-1, 3)
        faces = []
        for i in range(ny-1):
            for j in range(nx-1):
                idx = i * nx + j
                faces.extend([[idx, idx+1, idx+nx+1], [idx, idx+nx+1, idx+nx]])
        return verts.astype(np.float32), np.array(faces, dtype=np.int32)
    
    def _make_buildings(self, buildings):
        """Extrude building footprints to 3D with normalized elevations."""
        from triangle import triangulate
        
        all_verts, all_faces = [], []
        offset = 0
        
        for _, bld in buildings.iterrows():
            try:
                if bld.geometry.geom_type != 'Polygon':
                    continue
                
                height = bld.get('height', np.random.uniform(5, 25))
                if not isinstance(height, (int, float)):
                    height = np.random.uniform(5, 25)
                height = float(height)
                if height <= 0 or height > 500 or np.isnan(height):
                    height = np.random.uniform(5, 25)
                
                coords = np.array(bld.geometry.exterior.coords)[:, :2] - self.origin
                coords = np.unique(coords, axis=0)
                if len(coords) < 3:
                    continue
                
                # Validate coordinates
                if np.any(np.isnan(coords)) or np.any(np.isinf(coords)):
                    continue
                if np.any(np.abs(coords) > 1e6):  # Skip unreasonable coords
                    continue
                
                elev = self.heightmap.height_from_utm(bld.geometry.centroid.x, bld.geometry.centroid.y)
                if hasattr(elev, 'numpy'):
                    elev = float(elev.numpy()[0])
                elif hasattr(elev, '__iter__'):
                    elev = float(elev[0])
                if np.isnan(elev) or np.isinf(elev):
                    continue
                
                # Subtract z_offset to normalize building base elevation
                elev = elev - self.z_offset
                
                # Triangulate
                edges = np.array([[i, (i+1) % len(coords)] for i in range(len(coords))])
                tri = triangulate({'vertices': coords, 'segments': edges}, opts='p')
                v2d, f2d = tri['vertices'], tri['triangles']
                
                if len(v2d) < 3 or len(f2d) < 1:
                    continue
                
                # Extrude
                n = len(v2d)
                bottom = np.column_stack([v2d, np.full(n, elev)])
                top = np.column_stack([v2d, np.full(n, elev + height)])
                verts = np.vstack([bottom, top])
                
                # Final validation
                if np.any(np.isnan(verts)) or np.any(np.isinf(verts)):
                    continue
                
                # Faces
                faces = np.vstack([
                    f2d[:, [0, 2, 1]],  # bottom
                    f2d + n,             # top
                    *[np.array([[i, (i+1)%len(coords), (i+1)%len(coords)+n], 
                               [i, (i+1)%len(coords)+n, i+n]]) for i in range(len(coords))]
                ]) + offset
                
                all_verts.append(verts)
                all_faces.append(faces)
                offset += len(verts)
            except:
                continue
        
        if all_verts:
            return np.vstack(all_verts).astype(np.float32), np.vstack(all_faces).astype(np.int32)
        return np.array([]), np.array([])
    
    def _write_ply(self, path, verts, faces):
        mesh = o3d.t.geometry.TriangleMesh()
        mesh.vertex.positions = o3d.core.Tensor(verts)
        mesh.triangle.indices = o3d.core.Tensor(faces)
        o3d.t.io.write_triangle_mesh(str(path), mesh)
    
    def _add_shape(self, xml, name, filename, material):
        shape = ET.SubElement(xml, "shape", type="ply", id=f"mesh-{name}", name=f"mesh-{name}")
        ET.SubElement(shape, "transform", name="to_world")
        ET.SubElement(shape, "string", name="filename", value=filename)
        ET.SubElement(shape, "boolean", name="face_normals", value="true")
        ET.SubElement(shape, "ref", id=material, name="bsdf")
    
    def _setup_antennas(self):
        self.scene.tx_array = rt.PlanarArray(num_rows=8, num_cols=2, vertical_spacing=0.7,
                                             horizontal_spacing=0.5, pattern="iso", polarization="V")
        self.scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    
    def _add_transmitter(self):
        lat, lon = self.center
        min_lat, min_lon = self.bbox[0], self.bbox[1]
        
        center_utm = self.to_utm.transform(lon, lat)
        origin_utm = self.to_utm.transform(min_lon, min_lat)
        local_pos = (center_utm[0] - origin_utm[0], center_utm[1] - origin_utm[1])
        
        elev = self.heightmap.height_from_utm(center_utm[0], center_utm[1])
        if hasattr(elev, 'numpy'):
            elev = float(elev.numpy()[0])
        elif hasattr(elev, '__iter__'):
            elev = float(elev[0])
        
        # Normalize elevation by subtracting z_offset, then add TX_HEIGHT
        local_elev = elev - self.z_offset
        tx_z = local_elev + TX_HEIGHT
        
        logger.info(f"TX position: ({local_pos[0]:.1f}, {local_pos[1]:.1f}, {tx_z:.1f}) [elev={elev:.1f}, z_offset={self.z_offset:.1f}]")
        self.scene.add(rt.Transmitter("tx", position=[local_pos[0], local_pos[1], tx_z]))
    
    def compute_coverage(self):
        """Run ray tracing to get path gain.
        
        Coverage map is computed at terrain surface level.
        Since terrain is normalized to Z=0 (local coordinate system), the coverage
        map samples at ground level. Buildings sit on this normalized terrain.
        """
        solver = rt.RadioMapSolver()
        rm = solver(self.scene, max_depth=MAX_DEPTH, diffraction=DIFFRACTION,
                    cell_size=(self.resolution, self.resolution), samples_per_tx=NUM_SAMPLES,
                    los=True, specular_reflection=True, refraction=True)
        return rm.path_gain.numpy().squeeze()
    
    def get_elevation(self):
        """Get normalized elevation map including terrain and buildings.
        
        Returns a grid where each cell contains the height above local ground level.
        For terrain-only cells, this is the normalized terrain elevation.
        For building cells, this is terrain + building height.
        """
        min_lat, min_lon, max_lat, max_lon = self.bbox
        bl = self.to_utm.transform(min_lon, min_lat)
        tr = self.to_utm.transform(max_lon, max_lat)
        
        # Get terrain elevation and normalize (subtract z_offset)
        xx, yy = np.meshgrid(np.linspace(bl[0], tr[0], GRID_SIZE),
                             np.linspace(bl[1], tr[1], GRID_SIZE))
        h = self.heightmap.height_from_utm(xx.flatten(), yy.flatten())
        if hasattr(h, 'numpy'):
            h = h.numpy()
        terrain = np.array(h).reshape(GRID_SIZE, GRID_SIZE) - self.z_offset
        
        # Add building heights on top of terrain
        elevation = terrain.copy()
        try:
            buildings = ox.features.features_from_bbox(
                bbox=(min_lon, min_lat, max_lon, max_lat), tags={"building": True}
            ).to_crs(self.utm_crs)
            
            # Rasterize buildings onto grid
            for _, bld in buildings.iterrows():
                try:
                    if bld.geometry.geom_type != 'Polygon':
                        continue
                    
                    height = bld.get('height', np.random.uniform(5, 25))
                    if not isinstance(height, (int, float)):
                        height = np.random.uniform(5, 25)
                    height = float(height)
                    if height <= 0 or height > 500 or np.isnan(height):
                        height = np.random.uniform(5, 25)
                    
                    # Get building footprint bounds in grid coordinates
                    minx, miny, maxx, maxy = bld.geometry.bounds
                    
                    # Convert to grid indices
                    col_min = int((minx - bl[0]) / (tr[0] - bl[0]) * GRID_SIZE)
                    col_max = int((maxx - bl[0]) / (tr[0] - bl[0]) * GRID_SIZE)
                    row_min = int((miny - bl[1]) / (tr[1] - bl[1]) * GRID_SIZE)
                    row_max = int((maxy - bl[1]) / (tr[1] - bl[1]) * GRID_SIZE)
                    
                    # Clip to grid bounds
                    col_min, col_max = max(0, col_min), min(GRID_SIZE-1, col_max)
                    row_min, row_max = max(0, row_min), min(GRID_SIZE-1, row_max)
                    
                    # Add building height to cells within footprint
                    for r in range(row_min, row_max + 1):
                        for c in range(col_min, col_max + 1):
                            # Check if cell center is inside polygon
                            cell_x = bl[0] + (c + 0.5) / GRID_SIZE * (tr[0] - bl[0])
                            cell_y = bl[1] + (r + 0.5) / GRID_SIZE * (tr[1] - bl[1])
                            if bld.geometry.contains(Point(cell_x, cell_y)):
                                elevation[r, c] = terrain[r, c] + height
                except:
                    continue
        except Exception as e:
            logger.warning(f"Could not add buildings to elevation map: {e}")
        
        return elevation
    
    def cleanup(self):
        dr.flush_malloc_cache()
        dr.flush_kernel_cache()
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir, ignore_errors=True)
        gc.collect()


def resize_grid(arr, size=(200, 200)):
    """Resize array to fixed grid size."""
    if arr.ndim == 1:
        arr = arr.reshape((int(np.sqrt(len(arr))), -1))
    if arr.shape == size:
        return arr
    return ndimage.zoom(arr, (size[0]/arr.shape[0], size[1]/arr.shape[1]), order=1)


def generate_sample(args):
    """Generate one sample. args = (idx, output_dir, cache_base, seed)"""
    idx, output_dir, cache_base, seed = args
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Random location and radius
    radius = np.random.uniform(*RADIUS_RANGE)
    margin_lat = radius / 111320.0
    margin_lon = radius / (111320.0 * np.cos(np.radians(np.mean(BOULDER_LAT))))
    
    lat = np.random.uniform(BOULDER_LAT[0] + margin_lat, BOULDER_LAT[1] - margin_lat)
    lon = np.random.uniform(BOULDER_LON[0] + margin_lon, BOULDER_LON[1] - margin_lon)
    
    sample_id = f"{lat:.4f}_{lon:.4f}_{int(radius)}".replace('.', 'p').replace('-', 'm')
    output_file = output_dir / f"data_{sample_id}.h5"
    
    if output_file.exists():
        return sample_id
    
    logger.info(f"[{idx}] ({lat:.4f}, {lon:.4f}), R={radius:.0f}m")
    
    builder = None
    try:
        builder = SceneBuilder((lat, lon), radius, cache_base / f"s{idx}")
        
        # Check elevation range (skip mountainous terrain)
        if builder.elevation_range > MAX_ELEVATION_RANGE:
            logger.warning(f"Skipping: elevation range {builder.elevation_range:.0f}m exceeds {MAX_ELEVATION_RANGE}m")
            builder.cleanup()
            return None
        
        # Check minimum buildings
        if builder.building_count < MIN_BUILDINGS:
            logger.warning(f"Skipping: only {builder.building_count} buildings (need {MIN_BUILDINGS})")
            builder.cleanup()
            return None
        
        path_gain = resize_grid(builder.compute_coverage(), (GRID_SIZE, GRID_SIZE))
        elevation = builder.get_elevation()
        
        # Save
        with h5py.File(output_file, 'w') as f:
            f.attrs.update(center_lat=lat, center_lon=lon, radius=radius, frequency=FREQUENCY,
                          building_count=builder.building_count, tx_height=TX_HEIGHT)
            f.create_dataset('elevation_map', data=elevation)
            f.create_dataset('path_gain', data=path_gain)
        
        # Plot
        if PLOT:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            ax1.imshow(elevation, cmap='terrain', origin='lower')
            ax1.set_title(f'Elevation ({lat:.3f}, {lon:.3f})')
            plt.colorbar(ax1.images[0], ax=ax1, label='m')
            
            pg_db = 10 * np.log10(np.maximum(path_gain, 1e-15))
            ax2.imshow(pg_db, cmap='viridis', origin='lower', vmin=-150, vmax=-50)
            ax2.set_title(f'Path Gain (R={radius:.0f}m)')
            plt.colorbar(ax2.images[0], ax=ax2, label='dB')
            
            plt.tight_layout()
            plt.savefig(output_dir / f"map_{sample_id}.png", dpi=100)
            plt.close()
        
        builder.cleanup()
        return sample_id
        
    except Exception as e:
        logger.error(f"Error: {e}")
        if builder:
            builder.cleanup()
        return None


def main():
    import subprocess
    import sys
    
    # Check if running as worker
    if len(sys.argv) >= 5 and sys.argv[1] == "--worker":
        worker_id = int(sys.argv[2])
        total_workers = int(sys.argv[3])
        output_dir_arg = Path(sys.argv[4])
        # GPU assignment passed as env var
        run_worker(worker_id, total_workers, output_dir_arg)
        return
    
    # Main coordinator
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Detect available GPUs
    try:
        result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
        num_gpus = len([l for l in result.stdout.strip().split('\n') if l.startswith('GPU')])
    except Exception:
        num_gpus = 1
    
    actual_workers = num_gpus * WORKERS_PER_GPU
    
    print("=" * 60)
    print("Boulder Coverage Map Generator (Parallel)")
    print("=" * 60)
    print(f"Samples: {NUM_SAMPLES_TO_GENERATE}")
    print(f"Workers: {actual_workers} ({WORKERS_PER_GPU} per GPU x {num_gpus} GPUs)")
    print(f"GPU memory fraction per worker: {GPU_MEMORY_FRACTION:.0%}")
    print(f"Radius: {RADIUS_RANGE[0]}-{RADIUS_RANGE[1]}m")
    print(f"Frequency: {FREQUENCY/1e9:.1f} GHz, Depth: {MAX_DEPTH}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)
    
    # Launch worker subprocesses
    processes = []
    for i in range(actual_workers):
        # Assign GPU to worker (round-robin across GPUs)
        gpu_id = i % num_gpus if num_gpus > 0 else 0
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        # Limit GPU memory per worker
        env['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        env['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
        env['GPU_MEMORY_FRACTION'] = str(GPU_MEMORY_FRACTION)
        
        cmd = [sys.executable, __file__, "--worker", str(i), str(actual_workers), str(OUTPUT_DIR)]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        processes.append((i, p))
        print(f"Started worker {i} on GPU {gpu_id}")
    
    start = time.time()
    
    # Monitor progress
    try:
        while any(p.poll() is None for _, p in processes):
            # Count completed samples
            h5_files = list(OUTPUT_DIR.glob("data_*.h5"))
            count = len(h5_files)
            elapsed = time.time() - start
            rate = count / elapsed if elapsed > 0 else 0
            eta = (NUM_SAMPLES_TO_GENERATE - count) / rate / 3600 if rate > 0 else float('inf')
            print(f"\r✓ {count}/{NUM_SAMPLES_TO_GENERATE} (rate: {rate:.2f}/s) ETA: {eta:.2f}h", end="", flush=True)
            
            if count >= NUM_SAMPLES_TO_GENERATE:
                break
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nInterrupted - terminating workers...")
    
    # Terminate any remaining workers
    for i, p in processes:
        if p.poll() is None:
            p.terminate()
    
    # Wait for all to finish
    for i, p in processes:
        p.wait()
    
    # Final count
    h5_files = list(OUTPUT_DIR.glob("data_*.h5"))
    count = len(h5_files)
    
    # Cleanup cache directories
    for cache_d in OUTPUT_DIR.glob("cache_*"):
        shutil.rmtree(cache_d, ignore_errors=True)
    
    print()
    print("=" * 60)
    print(f"Done: {count} samples in {(time.time()-start)/60:.1f}min")
    print("=" * 60)


def run_worker(worker_id, total_workers, output_dir):
    """Worker process - generates samples assigned to this worker."""
    # Limit GPU memory for this worker
    memory_fraction = float(os.environ.get('GPU_MEMORY_FRACTION', 0.20))
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                # Set memory limit as fraction of total
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(
                        memory_limit=int(memory_fraction * 80000)  # ~80GB per GPU, adjust as needed
                    )]
                )
    except Exception as e:
        logger.warning(f"Could not set TensorFlow GPU memory limit: {e}")
    
    # Also limit Dr.Jit/Mitsuba memory usage
    try:
        import drjit as dr
        # Flush caches frequently to reduce memory footprint
        dr.set_flag(dr.JitFlag.KernelHistory, False)  # Disable kernel history to save memory
    except Exception as e:
        logger.warning(f"Could not configure Dr.Jit: {e}")
    
    cache_dir = output_dir / f"cache_{worker_id}"
    cache_dir.mkdir(exist_ok=True)
    
    samples_per_worker = (NUM_SAMPLES_TO_GENERATE + total_workers - 1) // total_workers
    start_idx = worker_id * samples_per_worker
    end_idx = min(start_idx + samples_per_worker, NUM_SAMPLES_TO_GENERATE)
    
    base_seed = int(time.time() * 1000) % (2**31) + worker_id * 10000
    
    count = 0
    max_attempts = samples_per_worker * 3
    
    for attempt in range(max_attempts):
        if count >= (end_idx - start_idx):
            break
        
        idx = start_idx + attempt
        seed = base_seed + attempt
        args = (idx, output_dir, cache_dir, seed)
        
        try:
            result = generate_sample(args)
            if result:
                count += 1
                logger.info(f"Worker {worker_id}: {count}/{end_idx - start_idx}")
        except Exception as e:
            logger.error(f"Worker {worker_id} error: {e}")
        
        # Aggressive memory cleanup after each sample
        gc.collect()
        try:
            dr.flush_malloc_cache()
            dr.flush_kernel_cache()
        except Exception:
            pass
    
    shutil.rmtree(cache_dir, ignore_errors=True)
    logger.info(f"Worker {worker_id} finished: {count} samples")


if __name__ == "__main__":
    main()
