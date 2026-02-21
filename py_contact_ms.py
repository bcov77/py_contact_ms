#!/usr/bin/env python
import math

import sys
sys.path.append('/home/bcov/sc/random/npose')
import npose_util as nu
import numpy as np
from scipy.spatial.distance import cdist

class Vec3:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x_ = float(x)
        self.y_ = float(y)
        self.z_ = float(z)

    def x(self, value=None):
        if value is None:
            return self.x_
        self.x_ = float(value)

    def y(self, value=None):
        if value is None:
            return self.y_
        self.y_ = float(value)

    def z(self, value=None):
        if value is None:
            return self.z_
        self.z_ = float(value)

    def __add__(self, other):
        return Vec3(self.x_ + other.x_, self.y_ + other.y_, self.z_ + other.z_)

    def __sub__(self, other):
        return Vec3(self.x_ - other.x_, self.y_ - other.y_, self.z_ - other.z_)

    def __mul__(self, scalar):
        return Vec3(self.x_ * scalar, self.y_ * scalar, self.z_ * scalar)

    def __neg__(self):
        return Vec3(-self.x_, -self.y_, -self.z_)

    def __truediv__(self, scalar):
        return Vec3(self.x_ / scalar, self.y_ / scalar, self.z_ / scalar)

    def dot(self, other):
        return self.x_ * other.x_ + self.y_ * other.y_ + self.z_ * other.z_

    def cross(self, other):
        return Vec3(
            self.y_ * other.z_ - self.z_ * other.y_,
            self.z_ * other.x_ - self.x_ * other.z_,
            self.x_ * other.y_ - self.y_ * other.x_
        )

    def magnitude_squared(self):
        return self.dot(self)

    def magnitude(self):
        return math.sqrt(self.magnitude_squared())

    def normalize(self):
        mag = self.magnitude()
        if mag > 0:
            self.x_ /= mag
            self.y_ /= mag
            self.z_ /= mag

    def distance(self, other):
        return (self - other).magnitude()

    def distance_squared(self, other):
        return (self - other).magnitude_squared()


class Atom(Vec3):

    def __init__(self):
        super().__init__(0.0, 0.0, 0.0)
        self.natom = 0
        self.nresidue = 0
        self.atom = ""
        self.residue = ""
        self.molecule = 0
        self.radius = 0.0
        self.density = 0.0
        self.atten = 0
        self.access = 0
        self.neighbors = []
        self.buried = []

    def __eq__(self, other):
        return self is other

    def __le__(self, other):
        return self.natom <= other.natom


class ResultsSurface:
    def __init__(self):
        self.d_mean = 0.0
        self.d_median = 0.0
        self.s_mean = 0.0
        self.s_median = 0.0
        self.nAtoms = 0
        self.nBuriedAtoms = 0
        self.nBlockedAtoms = 0
        self.nAllDots = 0
        self.nTrimmedDots = 0
        self.nBuriedDots = 0
        self.nAccessibleDots = 0
        self.trimmedArea = 0.0


class ResultsDots:
    def __init__(self):
        self.convex = 0
        self.concave = 0
        self.toroidal = 0


class RESULTS:
    def __init__(self):
        self.sc = 0.0
        self.area = 0.0
        self.distance = 0.0
        self.perimeter = 0.0
        self.nAtoms = 0
        self.surface = [ResultsSurface(), ResultsSurface(), ResultsSurface()]
        self.dots = ResultsDots()
        self.valid = 0


class DOT:
    def __init__(self, coor=None, area=None, atom=None, dot_type=None):
        self.coor = coor
        self.outnml = Vec3()
        self.area = area
        self.buried = 0
        self.type = dot_type
        self.atom = atom


class PROBE:
    def __init__(self):
        self.pAtoms = [None, None, None]
        self.height = 0.0
        self.point = Vec3()
        self.alt = Vec3()


from pyrosetta import rosetta
from pyrosetta import *
from pyrosetta.rosetta import *
init('-mute all')


# ── Struct-of-Arrays containers ───────────────────────────────────────────────
#
# AtomArray / AtomView  – drop-in replacement for list[Atom]
# DotArray              – replaces list[DOT]  (no view class needed)
# ProbeArray / ProbeView – replaces list[PROBE]
#
# Design goals
#   • All numeric data lives in pre-allocated numpy arrays (SoA layout).
#   • AtomView / ProbeView are *cached* proxy objects: atoms[i] always returns
#     the same Python object, so "atom1 is atom2" identity checks remain valid.
#   • The three-loop surface algorithm continues to work unmodified because
#     AtomView exposes the same attribute / method interface as the old Atom
#     class (x_, y_, z_, atten, neighbors, …).
#   • Vectorised callers (assign_attention_numbers, calc_contact_molecular_
#     surface) operate directly on the numpy arrays for O(N²) → numpy speed.

class AtomView:
    """
    Proxy to a single row in an AtomArray.

    Scalar numeric fields (x_, y_, z_, radius, …) are backed by numpy arrays
    inside the parent AtomArray.  Per-atom variable-length lists (neighbors,
    buried) are stored as ordinary Python lists on the view object itself.
    """

    # These live on the view instance, not in the numpy arrays.
    _LOCAL = frozenset({'_arr', '_idx', 'neighbors', 'buried'})

    def __init__(self, arr, idx):
        object.__setattr__(self, '_arr',      arr)
        object.__setattr__(self, '_idx',      idx)
        object.__setattr__(self, 'neighbors', [])
        object.__setattr__(self, 'buried',    [])

    # ── numpy-backed scalar properties ────────────────────────────────────

    @property
    def x_(self):            return float(self._arr.x[self._idx])
    @x_.setter
    def x_(self, v):         self._arr.x[self._idx] = v

    @property
    def y_(self):            return float(self._arr.y[self._idx])
    @y_.setter
    def y_(self, v):         self._arr.y[self._idx] = v

    @property
    def z_(self):            return float(self._arr.z[self._idx])
    @z_.setter
    def z_(self, v):         self._arr.z[self._idx] = v

    @property
    def natom(self):         return int(self._arr.natom[self._idx])
    @natom.setter
    def natom(self, v):      self._arr.natom[self._idx] = v

    @property
    def nresidue(self):      return int(self._arr.nresidue[self._idx])
    @nresidue.setter
    def nresidue(self, v):   self._arr.nresidue[self._idx] = v

    @property
    def molecule(self):      return int(self._arr.molecule[self._idx])
    @molecule.setter
    def molecule(self, v):   self._arr.molecule[self._idx] = v

    @property
    def radius(self):        return float(self._arr.radius[self._idx])
    @radius.setter
    def radius(self, v):     self._arr.radius[self._idx] = v

    @property
    def density(self):       return float(self._arr.density[self._idx])
    @density.setter
    def density(self, v):    self._arr.density[self._idx] = v

    @property
    def atten(self):         return int(self._arr.atten[self._idx])
    @atten.setter
    def atten(self, v):      self._arr.atten[self._idx] = v

    @property
    def access(self):        return int(self._arr.access[self._idx])
    @access.setter
    def access(self, v):     self._arr.access[self._idx] = v

    # String fields stored in Python lists inside AtomArray
    @property
    def atom(self):          return self._arr.atom_name[self._idx]
    @atom.setter
    def atom(self, v):       self._arr.atom_name[self._idx] = v

    @property
    def residue(self):       return self._arr.residue_name[self._idx]
    @residue.setter
    def residue(self, v):    self._arr.residue_name[self._idx] = v

    # ── Vec3-compatible interface ──────────────────────────────────────────

    def x(self, value=None):
        if value is None: return self.x_
        self.x_ = float(value)

    def y(self, value=None):
        if value is None: return self.y_
        self.y_ = float(value)

    def z(self, value=None):
        if value is None: return self.z_
        self.z_ = float(value)

    def __add__(self, other):
        return Vec3(self.x_ + other.x_, self.y_ + other.y_, self.z_ + other.z_)

    def __sub__(self, other):
        return Vec3(self.x_ - other.x_, self.y_ - other.y_, self.z_ - other.z_)

    def __mul__(self, scalar):
        return Vec3(self.x_ * scalar, self.y_ * scalar, self.z_ * scalar)

    def __truediv__(self, scalar):
        return Vec3(self.x_ / scalar, self.y_ / scalar, self.z_ / scalar)

    def __neg__(self):
        return Vec3(-self.x_, -self.y_, -self.z_)

    def dot(self, other):
        return self.x_*other.x_ + self.y_*other.y_ + self.z_*other.z_

    def cross(self, other):
        return Vec3(
            self.y_*other.z_ - self.z_*other.y_,
            self.z_*other.x_ - self.x_*other.z_,
            self.x_*other.y_ - self.y_*other.x_,
        )

    def magnitude_squared(self):
        return self.x_*self.x_ + self.y_*self.y_ + self.z_*self.z_

    def magnitude(self):
        return math.sqrt(self.magnitude_squared())

    def normalize(self):
        mag = self.magnitude()
        if mag > 0:
            self.x_ /= mag
            self.y_ /= mag
            self.z_ /= mag

    def distance(self, other):
        dx = self.x_ - other.x_; dy = self.y_ - other.y_; dz = self.z_ - other.z_
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def distance_squared(self, other):
        dx = self.x_ - other.x_; dy = self.y_ - other.y_; dz = self.z_ - other.z_
        return dx*dx + dy*dy + dz*dz

    # ── identity / ordering (matching Atom semantics) ─────────────────────

    def __eq__(self, other):  return self is other
    def __le__(self, other):  return self.natom <= other.natom
    def __hash__(self):       return id(self)

    def __repr__(self):
        return f"AtomView({self._idx}: {self.residue}:{self.atom} mol={self.molecule})"


class AtomArray:
    """
    Struct-of-arrays container for Atom data.

    Grows dynamically via append(); call finalize() to trim arrays to their
    true size before running vectorised operations.

    Iterating or indexing always returns the *same cached AtomView* object for
    a given index, preserving Python identity so that "atom1 is atom2" checks
    inside the surface-generation algorithm remain correct.
    """

    _INITIAL_CAP = 256
    _FLOAT_FIELDS = ('x', 'y', 'z', 'radius', 'density')
    _INT8_FIELDS  = ('molecule', 'atten', 'access')
    _INT32_FIELDS = ('natom', 'nresidue')

    def __init__(self):
        cap = self._INITIAL_CAP
        self._n   = 0
        self._cap = cap

        for f in self._FLOAT_FIELDS:
            setattr(self, f, np.zeros(cap, dtype=np.float64))
        for f in self._INT8_FIELDS:
            setattr(self, f, np.zeros(cap, dtype=np.int8))
        for f in self._INT32_FIELDS:
            setattr(self, f, np.zeros(cap, dtype=np.int32))

        self.atom_name    = [''] * cap
        self.residue_name = [''] * cap

        self._views: dict = {}   # index → AtomView cache

    def _grow(self):
        new_cap = self._cap * 2
        for f in self._FLOAT_FIELDS + self._INT8_FIELDS + self._INT32_FIELDS:
            old = getattr(self, f)
            new = np.zeros(new_cap, dtype=old.dtype)
            new[:self._n] = old[:self._n]
            setattr(self, f, new)
        self.atom_name    = self.atom_name    + [''] * self._cap
        self.residue_name = self.residue_name + [''] * self._cap
        self._cap = new_cap

    def append(self, atom):
        """Copy data from an Atom (or AtomView) into the array; return a cached AtomView."""
        if self._n >= self._cap:
            self._grow()
        i = self._n
        self.x[i]           = atom.x_
        self.y[i]           = atom.y_
        self.z[i]           = atom.z_
        self.radius[i]      = atom.radius
        self.density[i]     = atom.density
        self.natom[i]       = atom.natom
        self.nresidue[i]    = atom.nresidue
        self.molecule[i]    = atom.molecule
        self.atten[i]       = atom.atten
        self.access[i]      = atom.access
        self.atom_name[i]   = atom.atom
        self.residue_name[i]= atom.residue
        self._n += 1
        view = AtomView(self, i)
        self._views[i] = view
        return view

    def __len__(self):   return self._n
    def __bool__(self):  return self._n > 0

    def __getitem__(self, idx):
        if idx < 0:
            idx = self._n + idx
        if idx not in self._views:
            self._views[idx] = AtomView(self, idx)
        return self._views[idx]

    def __iter__(self):
        for i in range(self._n):
            yield self[i]

    def finalize(self):
        """Trim all arrays to [0:n].  Call once after all appends are done."""
        for f in self._FLOAT_FIELDS + self._INT8_FIELDS + self._INT32_FIELDS:
            setattr(self, f, getattr(self, f)[:self._n].copy())
        self.atom_name    = self.atom_name[:self._n]
        self.residue_name = self.residue_name[:self._n]


class DotArray:
    """
    Struct-of-arrays container for DOT data.

    No view class: the caller writes individual field values via append() and
    reads via the numpy arrays directly (enabling vectorised operations).
    """

    _INITIAL_CAP = 4096

    def __init__(self):
        cap = self._INITIAL_CAP
        self._n   = 0
        self._cap = cap

        for f in ('coor_x', 'coor_y', 'coor_z',
                  'outnml_x', 'outnml_y', 'outnml_z', 'area'):
            setattr(self, f, np.zeros(cap, dtype=np.float64))
        for f in ('buried', 'type_'):
            setattr(self, f, np.zeros(cap, dtype=np.int8))
        self.atom_idx = np.zeros(cap, dtype=np.int32)

    def _grow(self):
        new_cap = self._cap * 2
        for f in ('coor_x', 'coor_y', 'coor_z',
                  'outnml_x', 'outnml_y', 'outnml_z', 'area',
                  'buried', 'type_', 'atom_idx'):
            old = getattr(self, f)
            new = np.zeros(new_cap, dtype=old.dtype)
            new[:self._n] = old[:self._n]
            setattr(self, f, new)
        self._cap = new_cap

    def append(self, coor_x, coor_y, coor_z,
               outnml_x, outnml_y, outnml_z,
               area, buried, type_, atom_idx):
        if self._n >= self._cap:
            self._grow()
        i = self._n
        self.coor_x[i]   = coor_x;  self.coor_y[i]   = coor_y;  self.coor_z[i]   = coor_z
        self.outnml_x[i] = outnml_x; self.outnml_y[i] = outnml_y; self.outnml_z[i] = outnml_z
        self.area[i]     = area
        self.buried[i]   = buried
        self.type_[i]    = type_
        self.atom_idx[i] = atom_idx
        self._n += 1

    def __len__(self):   return self._n
    def __bool__(self):  return self._n > 0

    def finalize(self):
        for f in ('coor_x', 'coor_y', 'coor_z',
                  'outnml_x', 'outnml_y', 'outnml_z', 'area',
                  'buried', 'type_', 'atom_idx'):
            setattr(self, f, getattr(self, f)[:self._n].copy())


class ProbeView:
    """Proxy to a single row in a ProbeArray."""

    def __init__(self, arr, atom_arr, idx):
        object.__setattr__(self, '_arr',      arr)
        object.__setattr__(self, '_atom_arr', atom_arr)
        object.__setattr__(self, '_idx',      idx)

    @property
    def height(self):
        return float(self._arr.height[self._idx])

    @property
    def point(self):
        i = self._idx
        return Vec3(float(self._arr.point_x[i]),
                    float(self._arr.point_y[i]),
                    float(self._arr.point_z[i]))

    @property
    def alt(self):
        i = self._idx
        return Vec3(float(self._arr.alt_x[i]),
                    float(self._arr.alt_y[i]),
                    float(self._arr.alt_z[i]))

    @property
    def pAtoms(self):
        i = self._idx
        return [self._atom_arr[int(self._arr.atom_idx_0[i])],
                self._atom_arr[int(self._arr.atom_idx_1[i])],
                self._atom_arr[int(self._arr.atom_idx_2[i])]]

    def __eq__(self, other): return self is other
    def __hash__(self):      return id(self)


class ProbeArray:
    """
    Struct-of-arrays container for PROBE data.

    Like AtomArray, indexing returns cached ProbeView objects so that
    "probe is lprobe" identity comparisons in generate_concave_surface work.
    """

    _INITIAL_CAP = 1024

    def __init__(self, atom_arr):
        cap = self._INITIAL_CAP
        self._n        = 0
        self._cap      = cap
        self._atom_arr = atom_arr

        for f in ('height',
                  'point_x', 'point_y', 'point_z',
                  'alt_x',   'alt_y',   'alt_z'):
            setattr(self, f, np.zeros(cap, dtype=np.float64))
        for f in ('atom_idx_0', 'atom_idx_1', 'atom_idx_2'):
            setattr(self, f, np.zeros(cap, dtype=np.int32))

        self._views: dict = {}

    def _grow(self):
        new_cap = self._cap * 2
        for f in ('height',
                  'point_x', 'point_y', 'point_z',
                  'alt_x',   'alt_y',   'alt_z',
                  'atom_idx_0', 'atom_idx_1', 'atom_idx_2'):
            old = getattr(self, f)
            new = np.zeros(new_cap, dtype=old.dtype)
            new[:self._n] = old[:self._n]
            setattr(self, f, new)
        self._cap = new_cap

    def append(self, atom_idx_0, atom_idx_1, atom_idx_2, height, point, alt):
        """point and alt are Vec3 (or anything with .x_, .y_, .z_)."""
        if self._n >= self._cap:
            self._grow()
        i = self._n
        self.atom_idx_0[i] = atom_idx_0
        self.atom_idx_1[i] = atom_idx_1
        self.atom_idx_2[i] = atom_idx_2
        self.height[i]     = height
        self.point_x[i]    = point.x_; self.point_y[i] = point.y_; self.point_z[i] = point.z_
        self.alt_x[i]      = alt.x_;   self.alt_y[i]   = alt.y_;   self.alt_z[i]   = alt.z_
        self._n += 1
        view = ProbeView(self, self._atom_arr, i)
        self._views[i] = view
        return view

    def __len__(self):   return self._n
    def __bool__(self):  return self._n > 0

    def __getitem__(self, idx):
        if idx not in self._views:
            self._views[idx] = ProbeView(self, self._atom_arr, idx)
        return self._views[idx]

    def __iter__(self):
        for i in range(self._n):
            yield self[i]

    def finalize(self):
        for f in ('height',
                  'point_x', 'point_y', 'point_z',
                  'alt_x',   'alt_y',   'alt_z',
                  'atom_idx_0', 'atom_idx_1', 'atom_idx_2'):
            setattr(self, f, getattr(self, f)[:self._n].copy())


ATTEN_BLOCKER = 1
ATTEN_2 = 2
ATTEN_BURIED_FLAGGED = 5
ATTEN_6 = 6

MAX_SUBDIV = 100
PI = math.pi


class MolecularSurfaceCalculator:

    radii = []

    def __init__(self):

        self.settings = type("Settings", (), {})()
        self.settings.rp = 1.7
        self.settings.density = 15.0
        self.settings.band = 1.5
        self.settings.sep = 8.0
        self.settings.weight = 0.5
        self.settings.binwidth_dist = 0.02
        self.settings.binwidth_norm = 0.02

        self.reset()

        self.read_sc_radii()

    def reset(self):
        self.run = type("Run", (), {})()
        self.run.radmax     = 0.0
        self.run.results    = RESULTS()
        self.run.atoms      = AtomArray()
        self.run.dots       = [DotArray(), DotArray()]
        self.run.trimmed_dots = [DotArray(), DotArray()]
        self.run.probes     = ProbeArray(self.run.atoms)
        self.run.prevp      = Vec3()
        self.run.prevburied = 0


    def calc(self, pose, jump_id=1):

        if jump_id > pose.num_jump():
            raise ValueError("Jump ID out of bounds")

        is_upstream = rosetta.utility.vector1_bool(pose.size())
        # is_upstream.resize(pose.size())

        if jump_id > 0:
            is_upstream = pose.fold_tree().partition_by_jump(jump_id)
        else:
            for i in range(1, pose.size()+1):
                is_upstream[i] = True

        for i in range(1, pose.size()+1):


            residue = pose.residue(i)

            if residue.type().name() == "VRT":
                continue

            if residue.type().is_metal():
                continue

            mol = 0 if is_upstream[i] else 1
            self.add_residue(mol, residue)

        return self.CalcLoaded()

    def CalcLoaded(self):
        self.run.results.valid = 0
        assert len(self.run.atoms) > 0

        # Trim atom arrays to true size before vectorised attention assignment
        self.run.atoms.finalize()

        self.assign_attention_numbers(self.run.atoms)

        self.generate_molecular_surfaces()

        # Trim dot / probe arrays after surface generation
        self.run.dots[0].finalize()
        self.run.dots[1].finalize()
        self.run.probes.finalize()

        cms_return = self.calc_contact_molecular_surface(self.run.dots[0], self.run.dots[1])

        return cms_return

    def generate_molecular_surfaces(self):

        assert len(self.run.atoms) > 0

        self.calc_dots_for_all_atoms(self.run.atoms)


    def add_residue(self, molecule, residue, apolar_only=False):

        scatoms = []

        for i in range(1, residue.nheavyatoms()+1):

            if residue.is_virtual(i):
                continue

            if residue.type().is_metal():
                continue

            if apolar_only:
                if residue.atom_type(i).is_acceptor() or residue.atom_type(i).is_donor():
                    continue

            atom = Atom()
            xyz = residue.xyz(i)

            atom.x(xyz.x)
            atom.y(xyz.y)
            atom.z(xyz.z)

            atom.nresidue = residue.seqpos()
            atom.residue = residue.name3()
            atom.atom = residue.atom_name(i).strip()

            if not self.assign_atom_radius(atom):
                return 0

            scatoms.append(atom)

        for atom in scatoms:
            self.add_atom(molecule, atom)

        return len(scatoms)

    def read_sc_radii(self, filename="sc_radii.lib"):
        """
        Read side-chain radii definitions.

        Returns:
            1 if radii were successfully read (non-empty)
            0 otherwise
        """

        # If already populated, mimic MULTI_THREADED early exit behavior
        if getattr(self, "multi_threaded", False):
            if self.radii_:
                return 1

        try:
            with open(filename, "r") as f:
                lines = f.readlines()
        except OSError:
            if getattr(self, "trace_error", True):
                print(f"Failed to read {filename}")
            return 0

        self.radii_ = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 3:
                continue

            residue, atom, radius_value = parts[0], parts[1], parts[2]

            try:
                radius_float = float(radius_value)
            except ValueError:
                continue

            if residue and atom and radius_float > 0:
                if getattr(self, "trace_visible", False):
                    print(f"Atom Radius: {residue}:{atom} = {radius_float}")

                # Create ATOM_RADIUS-like object
                radius_obj = type("ATOM_RADIUS", (), {})()
                radius_obj.residue = residue
                radius_obj.atom = atom
                radius_obj.radius = radius_float

                self.radii_.append(radius_obj)

        if getattr(self, "trace_visible", False):
            print(f"Atom radii read: {len(self.radii_)}")

        return 1 if self.radii_ else 0


    def add_atom(self, molecule, atom):

        if atom.radius <= 0:
            self.assign_atom_radius(atom)

        if atom.radius > 0:
            atom.density  = self.settings.density
            atom.molecule = 1 if molecule == 1 else 0
            atom.natom    = len(self.run.atoms) + 1
            atom.access   = 0

            self.run.atoms.append(atom)  # copies into AtomArray, returns AtomView (unused here)
            self.run.results.surface[atom.molecule].nAtoms += 1
            self.run.results.nAtoms += 1

            return 1

        return 0



    def assign_atom_radius(self, atom):
        """
        Assign atom radius using wildcard matching.
        Returns 1 if assigned, 0 otherwise.
        """

        for radius in self.radii_:
            if not self.wildcard_match(atom.residue, radius.residue, len(atom.residue)+2):
                continue
            if not self.wildcard_match(atom.atom, radius.atom, len(atom.atom)+2):
                continue

            atom.radius = radius.radius
            return 1

        return 0

    def wildcard_match(self, query: str, pattern: str, l: int):
        """
        Inline residue and atom name matching.
        Mirrors C++ logic exactly.
        """

        qi = 0
        pi = 0

        while True:
            l -= 1
            if l <= 0:
                break

            q = query[qi] if qi < len(query) else '\0'
            p = pattern[pi] if pi < len(pattern) else '\0'

            match = (
                (q == p) or
                (q != '\0' and p == '*') or
                (q == ' ' and p == '\0')
            )

            if not match:
                return 0

            # Allow anything following a * in pattern
            if p == '*' and (pi + 1 >= len(pattern) or pattern[pi + 1] == '\0'):
                return 1

            if q != '\0':
                qi += 1
            if p != '\0':
                pi += 1

        return 1

    def calc_contact_molecular_surface(self, dots0, dots1):
        """
        Compute the contact molecular surface (vectorised).

        Replaces the O(K²) Python nested loop with a single cdist call.
        For each buried dot in molecule 0 the nearest buried dot in molecule 1
        is found in one shot; the weighted area sum is then a numpy reduction.
        """
        if len(dots0) == 0:
            return 0.0

        buried0 = dots0.buried.astype(bool)
        buried1 = dots1.buried.astype(bool)

        if not buried0.any() or not buried1.any():
            return 0.0

        xyz0_b  = np.column_stack([dots0.coor_x[buried0],
                                   dots0.coor_y[buried0],
                                   dots0.coor_z[buried0]])   # (K0, 3)
        xyz1_b  = np.column_stack([dots1.coor_x[buried1],
                                   dots1.coor_y[buried1],
                                   dots1.coor_z[buried1]])   # (K1, 3)
        area0_b = dots0.area[buried0]                        # (K0,)

        dist_sq      = cdist(xyz0_b, xyz1_b, metric='sqeuclidean')  # (K0, K1)
        min_dist_sq  = dist_sq.min(axis=1)                           # (K0,)

        return float((area0_b * np.exp(-min_dist_sq * self.settings.weight)).sum())


    def assign_attention_numbers(self, atoms, all_atoms=False):
        """
        Assign attention values to all atoms (vectorised).

        Replaces the O(N²) Python double-loop with a single scipy cdist call
        to compute all inter-molecule pairwise distances at once.
        """
        n   = len(atoms)
        mol = atoms.molecule[:n]

        if all_atoms:
            atoms.atten[:n] = ATTEN_BURIED_FLAGGED
            for m in range(2):
                self.run.results.surface[m].nBuriedAtoms += int((mol == m).sum())
            return 1

        xyz    = np.column_stack([atoms.x[:n], atoms.y[:n], atoms.z[:n]])
        mask0  = mol == 0
        mask1  = mol == 1
        xyz0   = xyz[mask0]    # (N0, 3)
        xyz1   = xyz[mask1]    # (N1, 3)

        if len(xyz0) > 0 and len(xyz1) > 0:
            d      = cdist(xyz0, xyz1)   # (N0, N1)
            min0   = d.min(axis=1)       # min distance for each mol-0 atom
            min1   = d.min(axis=0)       # min distance for each mol-1 atom
        else:
            min0 = np.full(mask0.sum(), 99999.0)
            min1 = np.full(mask1.sum(), 99999.0)

        idx0      = np.where(mask0)[0]
        idx1      = np.where(mask1)[0]
        blocker0  = min0 >= self.settings.sep
        blocker1  = min1 >= self.settings.sep

        atoms.atten[idx0[blocker0]]  = ATTEN_BLOCKER
        atoms.atten[idx0[~blocker0]] = ATTEN_BURIED_FLAGGED
        atoms.atten[idx1[blocker1]]  = ATTEN_BLOCKER
        atoms.atten[idx1[~blocker1]] = ATTEN_BURIED_FLAGGED

        self.run.results.surface[0].nBlockedAtoms += int(blocker0.sum())
        self.run.results.surface[0].nBuriedAtoms  += int((~blocker0).sum())
        self.run.results.surface[1].nBlockedAtoms += int(blocker1.sum())
        self.run.results.surface[1].nBuriedAtoms  += int((~blocker1).sum())

        return 1

    def calc_dots_for_all_atoms(self, _atoms_unused):
        """
        Main surface generation loop.
        """

        # Compute maximum atom radius
        self.run.radmax = 0.0
        for atom in self.run.atoms:
            if atom.radius > self.run.radmax:
                self.run.radmax = atom.radius

        # Generate convex surfaces
        for iatom, atom1 in enumerate(self.run.atoms):
            debug = iatom == 0

            if atom1.atten <= 0:
                assert not debug
                continue

            if not self.find_neighbors_and_buried_atoms(atom1):
                assert not debug
                continue

            if not atom1.access:
                assert not debug
                continue

            if atom1.atten <= ATTEN_BLOCKER:
                assert not debug
                continue

            if atom1.atten == ATTEN_6 and not atom1.buried:
                assert not debug
                continue

            self.generate_convex_surface(atom1)

        # Concave surface
        if self.settings.rp > 0:
            self.generate_concave_surface()

        return 1

    def _distance_key(self, ref_atom):
        def key(atom):
            return ref_atom.distance(atom)
        return key

    def find_neighbors_and_buried_atoms(self, atom1):

        if not self.find_neighbors_for_atom(atom1):
            return 0

        # sort neighbors by distance
        atom1.neighbors.sort(key=lambda a: atom1.distance(a))

        self.second_loop(atom1)

        return len(atom1.neighbors)

    import math

    def find_neighbors_for_atom(self, atom1):

        neighbors = atom1.neighbors
        if len(neighbors) != 0:
            raise RuntimeError("atom1.neighbors not empty before neighbor search")

        bb2 = (4 * self.run.radmax + 4 * self.settings.rp) ** 2
        nbb = 0

        for atom2 in self.run.atoms:

            if atom1 is atom2 or atom2.atten <= 0:
                continue

            if atom1.molecule == atom2.molecule:

                d2 = atom1.distance_squared(atom2)

                if d2 <= 0.0001:
                    raise RuntimeError(
                        f"Coincident atoms: "
                        f"{atom1.natom}:{atom1.residue}:{atom1.atom} == "
                        f"{atom2.natom}:{atom2.residue}:{atom2.atom}"
                    )

                bridge = atom1.radius + atom2.radius + 2 * self.settings.rp

                if d2 >= bridge * bridge:
                    continue

                neighbors.append(atom2)

            else:

                if atom2.atten < ATTEN_BURIED_FLAGGED:
                    continue

                d2 = atom1.distance_squared(atom2)

                if d2 < bb2:
                    nbb += 1

                bridge = atom1.radius + atom2.radius + 2 * self.settings.rp

                if d2 >= bridge * bridge:
                    continue

                atom1.buried.append(atom2)

        if atom1.atten == ATTEN_6 and not nbb:
            return 0

        if not neighbors:
            atom1.access = 1
            return 0

        return len(neighbors)

    import math

    def second_loop(self, atom1):

        neighbors = atom1.neighbors

        eri = atom1.radius + self.settings.rp

        for atom2 in neighbors:

            if atom2 <= atom1:
                continue

            erj = atom2.radius + self.settings.rp
            dij = atom1.distance(atom2)

            uij = (atom2 - atom1) / dij
            asymm = (eri * eri - erj * erj) / dij
            between = abs(asymm) < dij

            tij = ((atom1 + atom2) * 0.5) + (uij * (asymm * 0.5))

            _far_ = (eri + erj) ** 2 - dij * dij
            if _far_ <= 0.0:
                continue

            _far_ = math.sqrt(_far_)

            contain = dij * dij - (atom1.radius - atom2.radius) ** 2
            if contain <= 0.0:
                continue

            contain = math.sqrt(contain)
            rij = 0.5 * _far_ * contain / dij

            if len(neighbors) <= 1:
                atom1.access = 1
                atom2.access = 1
                break

            self.third_loop(atom1, atom2, uij, tij, rij)

            if (
                atom1.atten > ATTEN_BLOCKER or
                (atom2.atten > ATTEN_BLOCKER and self.settings.rp > 0.0)
            ):
                self.generate_toroidal_surface(atom1, atom2, uij, tij, rij, between)

        return 1


    def third_loop(self, atom1, atom2, uij, tij, rij):

        neighbors = atom1.neighbors

        eri = atom1.radius + self.settings.rp
        erj = atom2.radius + self.settings.rp

        for atom3 in neighbors:

            if atom3 <= atom2:
                continue

            erk = atom3.radius + self.settings.rp

            djk = atom2.distance(atom3)
            if djk >= erj + erk:
                continue

            dik = atom1.distance(atom3)
            if dik >= eri + erk:
                continue

            if (
                atom1.atten <= ATTEN_BLOCKER and
                atom2.atten <= ATTEN_BLOCKER and
                atom3.atten <= ATTEN_BLOCKER
            ):
                continue

            uik = (atom3 - atom1) / dik
            dt = uij.dot(uik)
            wijk = math.acos(dt)
            swijk = math.sin(wijk)

            if dt >= 1.0 or dt <= -1.0 or wijk <= 0.0 or swijk <= 0.0:

                dtijk2 = tij.distance(atom3)
                rkp2 = erk * erk - rij * rij

                if dtijk2 < rkp2:
                    return 0
                continue

            uijk = uij.cross(uik) / swijk
            utb = uijk.cross(uij)

            asymm = (eri * eri - erk * erk) / dik
            tik = (atom1 + atom3) * 0.5 + uik * asymm * 0.5

            tv = (tik - tij)
            tv = Vec3(
                uik.x() * tv.x(),
                uik.y() * tv.y(),
                uik.z() * tv.z()
            )

            dt = tv.x() + tv.y() + tv.z()
            bijk = tij + utb * (dt / swijk)

            hijk = eri * eri - bijk.distance_squared(atom1)
            if hijk <= 0.0:
                continue

            hijk = math.sqrt(hijk)

            for is0 in (1, 2):

                isign = 3 - 2 * is0
                pijk = bijk + uijk * (hijk * isign)

                if self.check_atom_collision2(pijk, atom2, atom3, neighbors):
                    continue

                if isign > 0:
                    a0, a1, a2 = atom1, atom2, atom3
                else:
                    a0, a1, a2 = atom2, atom1, atom3

                self.run.probes.append(
                    a0._idx, a1._idx, a2._idx,
                    hijk, pijk, uijk * isign,
                )

                atom1.access = 1
                atom2.access = 1
                atom3.access = 1

        return 1

    def check_atom_collision2(self, pijk, atom1, atom2, atoms):

        for neighbor in atoms:

            if neighbor is atom1 or neighbor is atom2:
                continue

            if (
                pijk.distance_squared(neighbor)
                <= (neighbor.radius + self.settings.rp) ** 2
            ):
                return 1

        return 0


    def generate_convex_surface(self, atom1):

        neighbors = atom1.neighbors

        north = Vec3(0, 0, 1)
        south = Vec3(0, 0, -1)
        eqvec = Vec3(1, 0, 0)

        ri = atom1.radius
        eri = atom1.radius + self.settings.rp

        if neighbors:

            neighbor = neighbors[0]

            north = atom1 - neighbor
            north.normalize()

            vtemp = Vec3(
                north.y()*north.y() + north.z()*north.z(),
                north.x()*north.x() + north.z()*north.z(),
                north.x()*north.x() + north.y()*north.y()
            )
            vtemp.normalize()

            dt = vtemp.dot(north)
            if abs(dt) > 0.99:
                vtemp = Vec3(1, 0, 0)

            eqvec = north.cross(vtemp)
            eqvec.normalize()

            vql = eqvec.cross(north)

            rj = neighbor.radius
            erj = neighbor.radius + self.settings.rp

            dij = atom1.distance(neighbor)
            uij = (neighbor - atom1) / dij

            asymm = (eri*eri - erj*erj) / dij
            tij = ((atom1 + neighbor) * 0.5) + (uij * (asymm * 0.5))

            _far_ = (eri + erj)**2 - dij*dij
            if _far_ <= 0.0:
                raise RuntimeError("Imaginary _far_")

            _far_ = math.sqrt(_far_)

            contain = dij*dij - (ri - rj)**2
            if contain <= 0.0:
                raise RuntimeError("Imaginary contain")

            contain = math.sqrt(contain)
            rij = 0.5 * _far_ * contain / dij

            pij = tij + (vql * rij)
            south = (pij - atom1) / eri

            if north.cross(south).dot(eqvec) <= 0.0:
                raise RuntimeError("Non-positive frame")

        lats = []
        o = Vec3(0, 0, 0)

        cs = self.sub_arc(o, ri, eqvec, atom1.density, north, south, lats)

        if not lats:
            return 0

        for ilat in lats:

            dt = ilat.dot(north)
            cen = atom1 + (north * dt)

            rad = ri*ri - dt*dt
            if rad <= 0.0:
                continue

            rad = math.sqrt(rad)

            points = []
            ps = self.sub_cir(cen, rad, north, atom1.density, points)

            if not points:
                continue

            area = ps * cs

            for point in points:

                pcen = atom1 + ((point - atom1) * (eri/ri))

                if self.check_point_collision(pcen, neighbors):
                    continue

                self.run.results.dots.convex += 1

                self.add_dot(
                    atom1.molecule,
                    1,
                    point,
                    area,
                    pcen,
                    atom1
                )

        return 1

    def check_point_collision(self, pcen, atoms):

        # skip first neighbor (matches C++ begin()+1)
        for neighbor in atoms[1:]:
            if pcen.distance(neighbor) <= (neighbor.radius + self.settings.rp):
                return 1

        return 0


    import math

    def generate_toroidal_surface(
        self,
        atom1,
        atom2,
        uij,
        tij,
        rij,
        between
    ):

        neighbors = atom1.neighbors

        density = (atom1.density + atom2.density) / 2.0

        eri = atom1.radius + self.settings.rp
        erj = atom2.radius + self.settings.rp

        rci = rij * atom1.radius / eri
        rcj = rij * atom2.radius / erj
        rb = rij - self.settings.rp
        if rb <= 0.0:
            rb = 0.0

        rs = (rci + 2 * rb + rcj) / 4.0
        e = rs / rij
        edens = e * e * density

        subs = []
        ts = self.sub_cir(tij, rij, uij, edens, subs)
        if not subs:
            return 0

        for sub in subs:

            # collision check
            tooclose = False
            for neighbor in neighbors:
                if neighbor is atom2:
                    continue
                erl = neighbor.radius + self.settings.rp
                if sub.distance_squared(neighbor) < erl * erl:
                    tooclose = True
                    break

            if tooclose:
                continue

            pij = sub
            atom1.access = 1
            atom2.access = 1

            if (
                atom1.atten == ATTEN_6 and
                atom2.atten == ATTEN_6 and
                not atom1.buried
            ):
                continue

            pi = (atom1 - pij) / eri
            pj = (atom2 - pij) / erj

            axis = pi.cross(pj)
            axis.normalize()

            dtq = self.settings.rp ** 2 - rij ** 2
            pcusp = dtq > 0 and between

            if pcusp:
                dtq = math.sqrt(dtq)
                qij = tij - uij * dtq
                qjk = tij + uij * dtq
                pqi = (qij - pij) / self.settings.rp
                pqj = Vec3(0.0, 0.0, 0.0)
            else:
                pqi = pi + pj
                pqi.normalize()
                pqj = pqi

            dt = pqi.dot(pi)
            if dt >= 1.0 or dt <= -1.0:
                return 0

            dt = pqj.dot(pj)
            if dt >= 1.0 or dt <= -1.0:
                return 0

            # Arc for atom1
            if atom1.atten >= ATTEN_2:
                points = []
                ps = self.sub_arc(pij, self.settings.rp, axis,
                                  density, pi, pqi, points)

                for point in points:
                    area = ps * ts * \
                           self.distance_point_to_line(tij, uij, point) / rij

                    self.run.results.dots.toroidal += 1
                    self.add_dot(atom1.molecule, 2,
                                 point, area, pij, atom1)

            # Arc for atom2
            if atom2.atten >= ATTEN_2:
                points = []
                ps = self.sub_arc(pij, self.settings.rp, axis,
                                  density, pqj, pj, points)

                for point in points:
                    area = ps * ts * \
                           self.distance_point_to_line(tij, uij, point) / rij

                    self.run.results.dots.toroidal += 1
                    self.add_dot(atom1.molecule, 2,
                                 point, area, pij, atom2)

        return 1

    def generate_concave_surface(self):

        lowprobs = []
        nears = []

        # collect low probes
        for probe in self.run.probes:
            if probe.height < self.settings.rp:
                lowprobs.append(probe)

        for probe in self.run.probes:

            if (
                probe.pAtoms[0].atten == ATTEN_6 and
                probe.pAtoms[1].atten == ATTEN_6 and
                probe.pAtoms[2].atten == ATTEN_6
            ):
                continue

            pijk = probe.point
            uijk = probe.alt
            hijk = probe.height

            density = (
                probe.pAtoms[0].density +
                probe.pAtoms[1].density +
                probe.pAtoms[2].density
            ) / 3.0

            # gather nearby low probes
            nears.clear()
            for lprobe in lowprobs:
                if probe is lprobe:
                    continue
                d2 = pijk.distance_squared(lprobe.point)
                if d2 <= 4 * (self.settings.rp ** 2):
                    nears.append(lprobe)

            # vectors to atoms
            vp = []
            for i in range(3):
                v = probe.pAtoms[i] - pijk
                v.normalize()
                vp.append(v)

            vectors = [
                vp[0].cross(vp[1]),
                vp[1].cross(vp[2]),
                vp[2].cross(vp[0])
            ]
            for v in vectors:
                v.normalize()

            # highest vertex
            dm = -1.0
            mm = 0
            for i in range(3):
                dt = uijk.dot(vp[i])
                if dt > dm:
                    dm = dt
                    mm = i

            south = -uijk
            axis = vp[mm].cross(south)
            axis.normalize()

            lats = []
            o = Vec3(0, 0, 0)

            cs = self.sub_arc(o, self.settings.rp,
                              axis, density,
                              vp[mm], south, lats)

            if not lats:
                continue

            for ilat in lats:

                dt = ilat.dot(south)
                cen = south * dt

                rad = self.settings.rp**2 - dt**2
                if rad <= 0.0:
                    continue
                rad = math.sqrt(rad)

                points = []
                ps = self.sub_cir(cen, rad, south,
                                  density, points)

                if not points:
                    continue

                area = ps * cs

                for point in points:

                    bail = False
                    for vector in vectors:
                        if point.dot(vector) >= 0.0:
                            bail = True
                            break
                    if bail:
                        continue

                    point = point + pijk

                    if (
                        hijk < self.settings.rp and
                        nears and
                        self.check_probe_collision(
                            point, nears,
                            self.settings.rp**2
                        )
                    ):
                        continue

                    # find closest atom
                    mc = 0
                    dmin = 2 * self.settings.rp
                    for i in range(3):
                        d = (
                            point.distance(probe.pAtoms[i]) -
                            probe.pAtoms[i].radius
                        )
                        if d < dmin:
                            dmin = d
                            mc = i

                    self.run.results.dots.concave += 1
                    self.add_dot(
                        probe.pAtoms[mc].molecule,
                        3,
                        point,
                        area,
                        pijk,
                        probe.pAtoms[mc]
                    )

        return 1

    def check_probe_collision(self, point, nears, r2):

        for near in nears:
            if point.distance_squared(near.point) < r2:
                return 1

        return 0

    def add_dot(
        self,
        molecule,
        type_,
        coor,
        area,
        pcen,
        atom
    ):
        pradius = self.settings.rp

        # outward normal
        if pradius <= 0:
            outnml = coor - atom   # Vec3 - AtomView → Vec3
        else:
            outnml = (pcen - coor) / pradius

        # buried determination
        if pcen.distance_squared(self.run.prevp) <= 0.0:
            buried = self.run.prevburied
        else:
            buried = 0
            for neighbor in atom.buried:
                erl = neighbor.radius + pradius
                d = pcen.distance_squared(neighbor)
                if d <= erl * erl:
                    buried = 1
                    break

            self.run.prevp      = pcen
            self.run.prevburied = buried

        self.run.dots[molecule].append(
            coor.x_,    coor.y_,    coor.z_,
            outnml.x_,  outnml.y_,  outnml.z_,
            area, buried, type_,
            atom._idx,   # AtomView carries its index into AtomArray
        )


    def distance_point_to_line(self, cen, axis, pnt):

        vec = pnt - cen
        dt = vec.dot(axis)
        d2 = vec.magnitude_squared() - dt * dt

        if d2 < 0.0:
            return 0.0

        return math.sqrt(d2)

    def sub_arc(
        self,
        cen,
        rad,
        axis,
        density,
        x,
        v,
        points
    ):

        y = axis.cross(x)

        dt1 = v.dot(x)
        dt2 = v.dot(y)

        angle = math.atan2(dt2, dt1)

        if angle < 0.0:
            angle += 2 * math.pi

        return self.sub_div(
            cen,
            rad,
            x,
            y,
            angle,
            density,
            points
        )

    def sub_div(self, cen, rad, x, y, angle, density, points):
        """
        Subdivide a circular arc and generate surface points.
        Returns arc length per subdivision (angular step).
        """

        delta = 1.0 / (math.sqrt(density) * rad)
        a = - delta / 2
        for i in range(MAX_SUBDIV):
            a = a + delta
            if a > angle:
                break
            c = rad * math.cos(a)
            s = rad * math.sin(a)
            points.append(cen + x*c + y*s)

        if len(points) > 0:
            ps = rad * angle / len(points)
        else:
            ps = 0

        return ps

    def sub_cir(self, cen, rad, axis, density, points):
        """
        Generate points on a full circle perpendicular to 'north'.
        Returns angular subdivision step.
        """

        v1 = Vec3(axis.y_**2 + axis.z_**2, axis.x_**2 + axis.z_**2, axis.x_**2 + axis.y_**2)
        v1.normalize()
        dt = v1.dot(axis)

        if abs(dt) > 0.99:
            v1 = Vec3(1.0, 0.0, 0.0)
        v2 = axis.cross(v1)
        v2.normalize()
        x = axis.cross(v2)
        x.normalize()
        y = axis.cross(x)

        return self.sub_div(cen, rad, x, y, 2*PI, density, points)



if __name__ == '__main__':

    import sys

    pdb = sys.argv[1]

    pose = pose_from_file(pdb)

    calc = MolecularSurfaceCalculator()
    cms = calc.calc(pose)

    print(cms)



    d0    = calc.run.dots[0]
    dots0 = np.column_stack([d0.coor_x, d0.coor_y, d0.coor_z])

    d1    = calc.run.dots[1]
    dots1 = np.column_stack([d1.coor_x, d1.coor_y, d1.coor_z])

    pr    = calc.run.probes
    probes = np.column_stack([pr.point_x, pr.point_y, pr.point_z]) if len(pr) else np.empty((0, 3))


