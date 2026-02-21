#!/usr/bin/env python
"""
Regression test for MolecularSurfaceCalculator.

Usage:
    Run from the project root so that sc_radii.lib is on the working directory:
        python test/cms_test.py

    To save the golden file:
        Set save_golden = True below and run once, then set it back to False.
"""
import os
import sys
import pickle

# Run from project root so sc_radii.lib can be found
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_TEST_DIR)
os.chdir(_ROOT_DIR)
sys.path.insert(0, _ROOT_DIR)

# Importing py_contact_ms initialises PyRosetta at module level
from py_contact_ms import MolecularSurfaceCalculator
from pyrosetta import pose_from_file

# ── Configuration ─────────────────────────────────────────────────────────────

save_golden = False
TOLERANCE   = 0.01          # 1 %

TEST_PDB    = os.path.join(_TEST_DIR, 'sc_example.pdb')
GOLDEN_PKL  = os.path.join(_TEST_DIR, 'golden.pkl')


# ── TestCalculatorRun ─────────────────────────────────────────────────────────

class TestCalculatorRun:
    """
    Serialisable snapshot of all self.run variables from a
    MolecularSurfaceCalculator, ready for pickle round-trips.

    Complex object references (Atom.neighbors, DOT.atom, etc.) are resolved
    to integer indices so the snapshot has no circular references.
    """

    def __init__(self, calc, cms_value):
        run = calc.run

        # ── top-level scalars ──────────────────────────────────────────────
        self.cms        = cms_value
        self.radmax     = run.radmax
        self.prevburied = run.prevburied

        # ── results ───────────────────────────────────────────────────────
        r = run.results
        self.results = {
            'sc':           r.sc,
            'area':         r.area,
            'distance':     r.distance,
            'perimeter':    r.perimeter,
            'nAtoms':       r.nAtoms,
            'valid':        r.valid,
            'dots_convex':  r.dots.convex,
            'dots_toroidal':r.dots.toroidal,
            'dots_concave': r.dots.concave,
        }
        for i, surf in enumerate(r.surface):
            p = f'surface_{i}'
            self.results.update({
                f'{p}_nAtoms':          surf.nAtoms,
                f'{p}_nBuriedAtoms':    surf.nBuriedAtoms,
                f'{p}_nBlockedAtoms':   surf.nBlockedAtoms,
                f'{p}_nAllDots':        surf.nAllDots,
                f'{p}_nTrimmedDots':    surf.nTrimmedDots,
                f'{p}_nBuriedDots':     surf.nBuriedDots,
                f'{p}_nAccessibleDots': surf.nAccessibleDots,
                f'{p}_trimmedArea':     surf.trimmedArea,
                f'{p}_d_mean':          surf.d_mean,
                f'{p}_d_median':        surf.d_median,
                f'{p}_s_mean':          surf.s_mean,
                f'{p}_s_median':        surf.s_median,
            })

        # ── atoms ─────────────────────────────────────────────────────────
        # Build id-to-index map first so dots/probes can reference atoms by index
        atom_to_idx = {id(a): i for i, a in enumerate(run.atoms)}

        self.atoms = []
        for atom in run.atoms:
            self.atoms.append({
                'natom':       atom.natom,
                'nresidue':    atom.nresidue,
                'atom':        atom.atom,
                'residue':     atom.residue,
                'molecule':    atom.molecule,
                'radius':      atom.radius,
                'density':     atom.density,
                'atten':       atom.atten,
                'access':      atom.access,
                'x':           atom.x_,
                'y':           atom.y_,
                'z':           atom.z_,
                'n_neighbors': len(atom.neighbors),
                'n_buried':    len(atom.buried),
            })

        # ── dots ──────────────────────────────────────────────────────────
        self.dots = [[], []]
        for mol in range(2):
            for dot in run.dots[mol]:
                self.dots[mol].append({
                    'coor_x':   dot.coor.x_,
                    'coor_y':   dot.coor.y_,
                    'coor_z':   dot.coor.z_,
                    'area':     dot.area,
                    'buried':   dot.buried,
                    'type':     dot.type,
                    'atom_idx': atom_to_idx.get(id(dot.atom), -1),
                })

        # ── trimmed_dots ──────────────────────────────────────────────────
        self.trimmed_dots = [[], []]
        for mol in range(2):
            for dot in run.trimmed_dots[mol]:
                self.trimmed_dots[mol].append({
                    'coor_x':   dot.coor.x_,
                    'coor_y':   dot.coor.y_,
                    'coor_z':   dot.coor.z_,
                    'area':     dot.area,
                    'buried':   dot.buried,
                    'type':     dot.type,
                    'atom_idx': atom_to_idx.get(id(dot.atom), -1),
                })

        # ── probes ────────────────────────────────────────────────────────
        self.probes = []
        for probe in run.probes:
            self.probes.append({
                'atom_indices': [atom_to_idx.get(id(a), -1) for a in probe.pAtoms],
                'height':  probe.height,
                'point_x': probe.point.x_,
                'point_y': probe.point.y_,
                'point_z': probe.point.z_,
                'alt_x':   probe.alt.x_,
                'alt_y':   probe.alt.y_,
                'alt_z':   probe.alt.z_,
            })


# ── Comparison helpers ────────────────────────────────────────────────────────

def _rel_diff(a, b):
    denom = max(abs(a), abs(b), 1e-10)
    return abs(a - b) / denom


def _check_scalar(name, cur, gold, tol, out):
    if isinstance(cur, str) or isinstance(gold, str):
        if cur != gold:
            out.append(f"  {name}: {gold!r} -> {cur!r}")
        return
    rd = _rel_diff(cur, gold)
    if rd > tol:
        sign = '+' if cur > gold else ''
        pct  = 100 * (cur - gold) / max(abs(gold), 1e-10)
        out.append(f"  {name}: {gold:.6g} -> {cur:.6g} ({sign}{pct:.2f}%)")


def _check_dict_list(name, cur_list, gold_list, numeric_keys, tol, out):
    """
    Compare two parallel lists of dicts.  Reports count mismatches and, for
    each numeric key, how many elements exceed the tolerance and the worst
    relative difference seen.
    """
    if len(cur_list) != len(gold_list):
        out.append(f"  {name} count: {len(gold_list)} -> {len(cur_list)}")
        return

    n = len(cur_list)
    for key in numeric_keys:
        n_diff  = 0
        max_rd  = 0.0
        for c, g in zip(cur_list, gold_list):
            if key not in c or key not in g:
                continue
            rd = _rel_diff(c[key], g[key])
            if rd > tol:
                n_diff += 1
                max_rd  = max(max_rd, rd)
        if n_diff:
            out.append(
                f"  {name}[*].{key}: "
                f"{n_diff}/{n} elements differ (max {100*max_rd:.2f}%)"
            )


def compare_runs(current, golden, tol=TOLERANCE):
    """
    Return a list of human-readable difference strings.
    An empty list means the two runs agree within *tol*.
    """
    diffs = []

    # top-level scalars
    _check_scalar('cms',        current.cms,        golden.cms,        tol, diffs)
    _check_scalar('radmax',     current.radmax,     golden.radmax,     tol, diffs)
    _check_scalar('prevburied', current.prevburied, golden.prevburied, tol, diffs)

    # results dict
    all_keys = sorted(set(golden.results) | set(current.results))
    for key in all_keys:
        if key not in current.results:
            diffs.append(f"  results.{key}: missing in current run")
        elif key not in golden.results:
            diffs.append(f"  results.{key}: missing in golden")
        else:
            _check_scalar(f'results.{key}',
                          current.results[key], golden.results[key], tol, diffs)

    # atoms
    _check_dict_list(
        'atoms', current.atoms, golden.atoms,
        ['natom', 'nresidue', 'molecule', 'radius', 'density',
         'atten', 'access', 'x', 'y', 'z', 'n_neighbors', 'n_buried'],
        tol, diffs,
    )

    # dots
    dot_keys = ['coor_x', 'coor_y', 'coor_z', 'area', 'buried', 'type', 'atom_idx']
    for mol in range(2):
        _check_dict_list(f'dots[{mol}]',
                         current.dots[mol], golden.dots[mol],
                         dot_keys, tol, diffs)

    # trimmed_dots
    for mol in range(2):
        _check_dict_list(f'trimmed_dots[{mol}]',
                         current.trimmed_dots[mol], golden.trimmed_dots[mol],
                         dot_keys, tol, diffs)

    # probes
    _check_dict_list(
        'probes', current.probes, golden.probes,
        ['height', 'point_x', 'point_y', 'point_z', 'alt_x', 'alt_y', 'alt_z'],
        tol, diffs,
    )

    return diffs


# ── Main ──────────────────────────────────────────────────────────────────────

def run_test():
    # 1. Load PDB
    print(f"Loading {TEST_PDB} ...")
    pose = pose_from_file(TEST_PDB)

    # 2. Run calculator
    calc = MolecularSurfaceCalculator()
    cms  = calc.calc(pose)
    print(f"CMS value: {cms:.6f}")

    # 3. Snapshot run state
    current_run = TestCalculatorRun(calc, cms)

    # 6. Optionally save golden and exit
    if save_golden:
        with open(GOLDEN_PKL, 'wb') as f:
            pickle.dump(current_run, f)
        print(f"Golden data saved to {GOLDEN_PKL}")
        return

    # 4. Load golden
    if not os.path.exists(GOLDEN_PKL):
        print(
            f"No golden file found at {GOLDEN_PKL}.\n"
            f"Set save_golden = True in {__file__} and run once to create it."
        )
        return

    with open(GOLDEN_PKL, 'rb') as f:
        golden_run = pickle.load(f)

    # 5. Compare
    diffs = compare_runs(current_run, golden_run)

    if diffs:
        print(f"REGRESSION: {len(diffs)} variable(s) differ by more than "
              f"{100*TOLERANCE:.0f}%:")
        for d in diffs:
            print(d)
    else:
        print(f"PASS: all variables match within {100*TOLERANCE:.0f}% tolerance.")


if __name__ == '__main__':
    run_test()
