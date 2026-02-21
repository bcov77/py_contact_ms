#!/usr/bin/env python
import math

import sys
sys.path.append('/home/bcov/sc/random/npose')
import npose_util as nu
import numpy as np

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
        self.run.radmax = 0.0
        self.run.results = RESULTS()
        self.run.atoms = []
        self.run.dots = [[], []]
        self.run.trimmed_dots = [[], []]
        self.run.probes = []
        self.run.prevp = Vec3()
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

        self.assign_attention_numbers(self.run.atoms)

        self.generate_molecular_surfaces()

        cms_return = self.calc_contact_molecular_surface( self.run.dots[0], self.run.dots[1] )

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
            atom.density = self.settings.density
            atom.molecule = 1 if molecule == 1 else 0
            atom.natom = len(self.run.atoms) + 1
            atom.access = 0

            self.run.atoms.append(atom)
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

    def calc_contact_molecular_surface(self, my_dots, their_dots):

        if len(my_dots) == 0:
            return 0

        buried_their_dots = []
        for dot in their_dots:
            if dot.buried:
                buried_their_dots.append(dot)

        areas = np.zeros(len(my_dots))
        for idot, dot in enumerate(my_dots):
            if not dot.buried:
                continue
            neighbor = self.calc_neighbor_distance_find_closest_neighbor(dot, buried_their_dots)
            if not neighbor:
                continue
            distmin = neighbor.coor.distance_squared(dot.coor)
            areas[idot] = dot.area * np.exp( -distmin * self.settings.weight )

        return areas.sum()

    def calc_neighbor_distance_find_closest_neighbor(self, dot1, their_dots):
        distmin = 9999999
        neighbor = None
        for dot2 in their_dots:
            if not dot2.buried:
                continue
            d = dot2.coor.distance_squared(dot1.coor)
            if d < distmin:
                distmin = d
                neighbor = dot2

        return neighbor


    def assign_attention_numbers(self, atoms, all_atoms=False):
        """
        Assign default attention values to all atoms.
        """

        if all_atoms:
            for atom in atoms:
                atom.atten = ATTEN_BURIED_FLAGGED
                self.run.results.surface[atom.molecule].nBuriedAtoms += 1
        else:
            for atom1 in atoms:
                dist_min = 99999.0
                for atom2 in atoms:
                    if atom1.molecule == atom2.molecule:
                        continue
                    r = atom1.distance(atom2)
                    if r < dist_min:
                        dist_min = r

                if dist_min >= self.settings.sep:
                    atom1.atten = ATTEN_BLOCKER
                    self.run.results.surface[atom1.molecule].nBlockedAtoms += 1
                else:
                    atom1.atten = ATTEN_BURIED_FLAGGED
                    self.run.results.surface[atom1.molecule].nBuriedAtoms += 1

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

                probe = PROBE()

                if isign > 0:
                    probe.pAtoms = [atom1, atom2, atom3]
                else:
                    probe.pAtoms = [atom2, atom1, atom3]

                probe.height = hijk
                probe.point = pijk
                probe.alt = uijk * isign

                self.run.probes.append(probe)

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

        dot = DOT()
        dot.coor = coor
        dot.outnml = Vec3()
        dot.area = area
        dot.buried = 0
        dot.type = type_
        dot.atom = atom

        pradius = self.settings.rp

        # outward normal
        if pradius <= 0:
            dot.outnml = coor - atom
        else:
            dot.outnml = (pcen - coor) / pradius

        # buried determination
        if pcen.distance_squared(self.run.prevp) <= 0.0:
            dot.buried = self.run.prevburied
        else:
            dot.buried = 0
            for neighbor in atom.buried:
                erl = neighbor.radius + pradius
                d = pcen.distance_squared(neighbor)
                if d <= erl * erl:
                    dot.buried = 1
                    break

            self.run.prevp = pcen
            self.run.prevburied = dot.buried

        self.run.dots[molecule].append(dot)


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



    dots0 = []
    for dot in calc.run.dots[0]:
        dots0.append(np.array([dot.coor.x_, dot.coor.y_, dot.coor.z_]))

    dots0 = np.array(dots0)

    dots1 = []
    for dot in calc.run.dots[1]:
        dots1.append(np.array([dot.coor.x_, dot.coor.y_, dot.coor.z_]))
    dots1 = np.array(dots1)


    probes = []
    for probe in calc.run.probes:
        probes.append(np.array([probe.point.x_, probe.point.y_, probe.point.z_]))
    probes = np.array(probes)


