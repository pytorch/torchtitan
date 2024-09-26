import itertools
import re
from collections import Counter
from contextlib import suppress
from typing import Callable, List, Optional, Union

import datamol as dm
import numpy as np
from rdkit import Chem
from rdkit.Chem import BRICS

class SAFEDecodeError(Exception):
    """Raised when a string cannot be decoded with the given encoding."""
    pass

class SAFEEncodeError(Exception):
    """Raised when a molecule cannot be encoded using SAFE."""
    pass


class SAFEFragmentationError(Exception):
    """Raised when a the slicing algorithm return empty bonds."""
    pass


class SAFEConverter:
    """Molecule line notation conversion from SMILES to SAFE

    A SAFE representation is a string based representation of a molecule decomposition into fragment components,
    separated by a dot ('.'). Note that each component (fragment) might not be a valid molecule by themselves,
    unless explicitely correct to add missing hydrogens.

    !!! note "Slicing algorithms"

        By default SAFE strings are generated using `BRICS`, however, the following alternative are supported:

        * [Hussain-Rea (`hr`)](https://pubs.acs.org/doi/10.1021/ci900450m)
        * [RECAP (`recap`)](https://pubmed.ncbi.nlm.nih.gov/9611787/)
        * [RDKit's MMPA (`mmpa`)](https://www.rdkit.org/docs/source/rdkit.Chem.rdMMPA.html)
        * Any possible attachment points (`attach`)

        Furthermore, you can also provide your own slicing algorithm, which should return a pair of atoms
        corresponding to the bonds to break.

    """

    SUPPORTED_SLICERS = ["hr", "rotatable", "recap", "mmpa", "attach", "brics"]
    __SLICE_SMARTS = {
        "hr": ["[*]!@-[*]"],  # any non ring single bond
        "recap": [
            "[$([C;!$(C([#7])[#7])](=!@[O]))]!@[$([#7;+0;!D1])]",
            "[$(C=!@O)]!@[$([O;+0])]",
            "[$([N;!D1;+0;!$(N-C=[#7,#8,#15,#16])](-!@[*]))]-!@[$([*])]",
            "[$(C(=!@O)([#7;+0;D2,D3])!@[#7;+0;D2,D3])]!@[$([#7;+0;D2,D3])]",
            "[$([O;+0](-!@[#6!$(C=O)])-!@[#6!$(C=O)])]-!@[$([#6!$(C=O)])]",
            "C=!@C",
            "[N;+1;D4]!@[#6]",
            "[$([n;+0])]-!@C",
            "[$([O]=[C]-@[N;+0])]-!@[$([C])]",
            "c-!@c",
            "[$([#7;+0;D2,D3])]-!@[$([S](=[O])=[O])]",
        ],
        "mmpa": ["[#6+0;!$(*=,#[!#6])]!@!=!#[*]"],  # classical mmpa slicing smarts
        "attach": ["[*]!@[*]"],  # any potential attachment point, including hydrogens when explicit
        "rotatable": ["[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]"],
    }

    def __init__(
        self,
        slicer: Optional[Union[str, List[str], Callable]] = "brics",
        require_hs: Optional[bool] = None,
        use_original_opener_for_attach: bool = True,
        ignore_stereo: bool = False,
    ):
        """Constructor for the SAFE converter

        Args:
            slicer: slicer algorithm to use for encoding.
                Can either be one of the supported slicing algorithm (SUPPORTED_SLICERS)
                or a custom callable that returns the bond ids that can be sliced.
            require_hs: whether the slicing algorithm require the molecule to have hydrogen explictly added.
                `attach` slicer requires adding hydrogens.
            use_original_opener_for_attach: whether to use the original branch opener digit when adding back
                mapping number to attachment points, or use simple enumeration.
            ignore_stereo: RDKIT does not support some particular SAFE subset when stereochemistry is defined.

        """
        self.slicer = slicer
        if isinstance(slicer, str) and slicer.lower() in self.SUPPORTED_SLICERS:
            self.slicer = self.__SLICE_SMARTS.get(slicer.lower(), slicer)
        if self.slicer != "brics" and isinstance(self.slicer, str):
            self.slicer = [self.slicer]
        if isinstance(self.slicer, (list, tuple)):
            self.slicer = [dm.from_smarts(x) for x in self.slicer]
            if any(x is None for x in self.slicer):
                raise ValueError(f"Slicer: {slicer} cannot be valid")
        self.require_hs = require_hs or (slicer == "attach")
        self.use_original_opener_for_attach = use_original_opener_for_attach
        self.ignore_stereo = ignore_stereo

    @staticmethod
    def randomize(mol: dm.Mol, rng: Optional[int] = None):
        """Randomize the position of the atoms in a mol.

        Args:
            mol: molecules to randomize
            rng: optional seed to use
        """
        if isinstance(rng, int):
            rng = np.random.default_rng(rng)
        if mol.GetNumAtoms() == 0:
            return mol
        atom_indices = list(range(mol.GetNumAtoms()))
        atom_indices = rng.permutation(atom_indices).tolist()
        return Chem.RenumberAtoms(mol, atom_indices)

    @classmethod
    def _find_branch_number(cls, inp: str):
        """Find the branch number and ring closure in the SMILES representation using regexp

        Args:
            inp: input smiles
        """
        inp = re.sub(r"\[.*?\]", "", inp)  # noqa
        matching_groups = re.findall(r"((?<=%)\d{2})|((?<!%)\d+)(?![^\[]*\])", inp)
        # first match is for multiple connection as multiple digits
        # second match is for single connections requiring 2 digits
        # SMILES does not support triple digits
        branch_numbers = []
        for m in matching_groups:
            if m[0] == "":
                branch_numbers.extend(int(mm) for mm in m[1])
            elif m[1] == "":
                branch_numbers.append(int(m[0].replace("%", "")))
        return branch_numbers

    def _fix_safe(self, inp: str):
        """Ensure that the input SAFE string is valid by fixing the missing attachment points

        Args:
            inp: input SAFE string

        """
        missing_tokens = [inp]
        branch_numbers = self._find_branch_number(inp)
        # only use the set that have exactly 1 element
        # any branch number that is not pairwise should receive a dummy atom to complete the attachment point
        branch_numbers = Counter(branch_numbers)
        for i, (bnum, bcount) in enumerate(branch_numbers.items()):
            if bcount % 2 != 0:
                bnum_str = str(bnum) if bnum < 10 else f"%{bnum}"
                _tk = f"[*:{i+1}]{bnum_str}"
                if self.use_original_opener_for_attach:
                    bnum_digit = bnum_str.strip("%")  # strip out the % sign
                    _tk = f"[*:{bnum_digit}]{bnum_str}"
                missing_tokens.append(_tk)
        return ".".join(missing_tokens)

    def decoder(
        self,
        inp: str,
        as_mol: bool = False,
        canonical: bool = False,
        fix: bool = True,
        remove_dummies: bool = True,
        remove_added_hs: bool = True,
    ):
        """Convert input SAFE representation to smiles

        Args:
            inp: input SAFE representation to decode as a valid molecule or smiles
            as_mol: whether to return a molecule object or a smiles string
            canonical: whether to return a canonical
            fix: whether to fix the SAFE representation to take into account non-connected attachment points
            remove_dummies: whether to remove dummy atoms from the SAFE representation. Note that removing_dummies is incompatible with
            remove_added_hs: whether to remove all the added hydrogen atoms after applying dummy removal for recovery
        """

        if fix:
            inp = self._fix_safe(inp)
        mol = dm.to_mol(inp)
        if remove_dummies:
            with suppress(Exception):
                du = dm.from_smarts("[$([#0]!-!:*);$([#0;D1])]")
                out = Chem.ReplaceSubstructs(mol, du, dm.to_mol("C"), True)[0]
                mol = dm.remove_dummies(out)
        if as_mol:
            if remove_added_hs:
                mol = dm.remove_hs(mol, update_explicit_count=True)
            if canonical:
                mol = dm.standardize_mol(mol)
                mol = dm.canonical_tautomer(mol)
            return mol
        out = dm.to_smiles(mol, canonical=canonical, explicit_hs=(not remove_added_hs))
        if canonical:
            out = dm.standardize_smiles(out)
        return out

    def _fragment(self, mol: dm.Mol, allow_empty: bool = False):
        """
        Perform bond cutting in place for the input molecule, given the slicing algorithm

        Args:
            mol: input molecule to split
            allow_empty: whether to allow the slicing algorithm to return empty bonds
        Raises:
            SAFEFragmentationError: if the slicing algorithm return empty bonds
        """

        if self.slicer is None:
            matching_bonds = []

        elif callable(self.slicer):
            matching_bonds = self.slicer(mol)
            matching_bonds = list(matching_bonds)

        elif self.slicer == "brics":
            matching_bonds = BRICS.FindBRICSBonds(mol)
            matching_bonds = [brics_match[0] for brics_match in matching_bonds]

        else:
            matches = set()
            for smarts in self.slicer:
                matches |= {
                    tuple(sorted(match)) for match in mol.GetSubstructMatches(smarts, uniquify=True)
                }
            matching_bonds = list(matches)

        if matching_bonds is None or len(matching_bonds) == 0 and not allow_empty:
            raise SAFEFragmentationError(
                "Slicing algorithms did not return any bonds that can be cut !"
            )
        return matching_bonds or []

    def encoder(
        self,
        inp: Union[str, dm.Mol],
        canonical: bool = True,
        randomize: Optional[bool] = False,
        seed: Optional[int] = None,
        constraints: Optional[List[dm.Mol]] = None,
        allow_empty: bool = False,
        rdkit_safe: bool = True,
    ):
        """Convert input smiles to SAFE representation

        Args:
            inp: input smiles
            canonical: whether to return canonical smiles string. Defaults to True
            randomize: whether to randomize the safe string encoding. Will be ignored if canonical is provided
            seed: optional seed to use when allowing randomization of the SAFE encoding.
                Randomization happens at two steps:
                1. at the original smiles representation by randomization the atoms.
                2. at the SAFE conversion by randomizing fragment orders
            constraints: List of molecules or pattern to preserve during the SAFE construction. Any bond slicing would
                happen outside of a substructure matching one of the patterns.
            allow_empty: whether to allow the slicing algorithm to return empty bonds
            rdkit_safe: whether to apply rdkit-safe digit standardization to the output SAFE string.
        """
        rng = None
        if randomize:
            rng = np.random.default_rng(seed)
            if not canonical:
                inp = dm.to_mol(inp, remove_hs=False)
                inp = self.randomize(inp, rng)

        if isinstance(inp, dm.Mol):
            inp = dm.to_smiles(inp, canonical=canonical, randomize=False, ordered=False)

        # EN: we first normalize the attachment if the molecule is a query:
        # inp = dm.reactions.convert_attach_to_isotope(inp, as_smiles=True)

        # TODO(maclandrol): RDKit supports some extended form of ring closure, up to 5 digits
        # https://www.rdkit.org/docs/RDKit_Book.html#ring-closures and I should try to include them
        branch_numbers = self._find_branch_number(inp)

        mol = dm.to_mol(inp, remove_hs=False)
        potential_stereos = Chem.FindPotentialStereo(mol)
        has_stereo_bonds = any(x.type == Chem.StereoType.Bond_Double for x in potential_stereos)
        if self.ignore_stereo:
            mol = dm.remove_stereochemistry(mol)

        bond_map_id = 1
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                atom.SetAtomMapNum(0)
                atom.SetIsotope(bond_map_id)
                bond_map_id += 1

        if self.require_hs:
            mol = dm.add_hs(mol)
        matching_bonds = self._fragment(mol, allow_empty=allow_empty)
        substructed_ignored = []
        if constraints is not None:
            substructed_ignored = list(
                itertools.chain(
                    *[
                        mol.GetSubstructMatches(constraint, uniquify=True)
                        for constraint in constraints
                    ]
                )
            )

        bonds = []
        for i_a, i_b in matching_bonds:
            # if both atoms of the bond are found in a disallowed substructure, we cannot consider them
            # on the other end, a bond between two substructure to preserved independently is perfectly fine
            if any((i_a in ignore_x and i_b in ignore_x) for ignore_x in substructed_ignored):
                continue
            obond = mol.GetBondBetweenAtoms(i_a, i_b)
            bonds.append(obond.GetIdx())

        if len(bonds) > 0:
            mol = Chem.FragmentOnBonds(
                mol,
                bonds,
                dummyLabels=[(i + bond_map_id, i + bond_map_id) for i in range(len(bonds))],
            )
        # here we need to be clever and disable rooted atom as the atom with mapping

        frags = list(Chem.GetMolFrags(mol, asMols=True))
        if randomize:
            frags = rng.permutation(frags).tolist()
        elif canonical:
            frags = sorted(
                frags,
                key=lambda x: x.GetNumAtoms(),
                reverse=True,
            )

        frags_str = []
        for frag in frags:
            non_map_atom_idxs = [
                atom.GetIdx() for atom in frag.GetAtoms() if atom.GetAtomicNum() != 0
            ]
            frags_str.append(
                Chem.MolToSmiles(
                    frag,
                    isomericSmiles=True,
                    canonical=True,  # needs to always be true
                    rootedAtAtom=non_map_atom_idxs[0],
                )
            )

        scaffold_str = ".".join(frags_str)
        # EN: fix for https://github.com/datamol-io/safe/issues/37
        # we were using the wrong branch number count which did not take into account
        # possible change in digit utilization after bond slicing
        scf_branch_num = self._find_branch_number(scaffold_str) + branch_numbers

        # don't capture atom mapping in the scaffold
        attach_pos = set(re.findall(r"(\[\d+\*\]|!\[[^:]*:\d+\])", scaffold_str))
        if canonical:
            attach_pos = sorted(attach_pos)
        starting_num = 1 if len(scf_branch_num) == 0 else max(scf_branch_num) + 1
        for attach in attach_pos:
            val = str(starting_num) if starting_num < 10 else f"%{starting_num}"
            # we cannot have anything of the form "\([@=-#-$/\]*\d+\)"
            attach_regexp = re.compile(r"(" + re.escape(attach) + r")")
            scaffold_str = attach_regexp.sub(val, scaffold_str)
            starting_num += 1

        # now we need to remove all the parenthesis around digit only number
        wrong_attach = re.compile(r"\(([\%\d]*)\)")
        scaffold_str = wrong_attach.sub(r"\g<1>", scaffold_str)
        # furthermore, we autoapply rdkit-compatible digit standardization.
        if rdkit_safe:
            pattern = r"\(([=-@#\/\\]{0,2})(%?\d{1,2})\)"
            replacement = r"\g<1>\g<2>"
            scaffold_str = re.sub(pattern, replacement, scaffold_str)
        if not self.ignore_stereo and has_stereo_bonds and not dm.same_mol(scaffold_str, inp):
            print(
                "Warning: Ignoring stereo is disabled, but molecule has stereochemistry interferring with SAFE representation"
            )
        return scaffold_str


def encode(
    inp: Union[str, dm.Mol],
    canonical: bool = True,
    randomize: Optional[bool] = False,
    seed: Optional[int] = None,
    slicer: Optional[Union[List[str], str, Callable]] = None,
    require_hs: Optional[bool] = None,
    constraints: Optional[List[dm.Mol]] = None,
    ignore_stereo: Optional[bool] = False,
):
    """
    Convert input smiles to SAFE representation

    Args:
        inp: input smiles
        canonical: whether to return canonical SAFE string. Defaults to True
        randomize: whether to randomize the safe string encoding. Will be ignored if canonical is provided
        seed: optional seed to use when allowing randomization of the SAFE encoding.
        slicer: slicer algorithm to use for encoding. Defaults to "brics".
        require_hs: whether the slicing algorithm require the molecule to have hydrogen explictly added.
        constraints: List of molecules or pattern to preserve during the SAFE construction.
        ignore_stereo: RDKIT does not support some particular SAFE subset when stereochemistry is defined.
    """
    if slicer is None:
        slicer = "brics"
    with dm.without_rdkit_log():
        safe_obj = SAFEConverter(slicer=slicer, require_hs=require_hs, ignore_stereo=ignore_stereo)
        try:
            encoded = safe_obj.encoder(
                inp,
                canonical=canonical,
                randomize=randomize,
                constraints=constraints,
                seed=seed,
            )
        except SAFEFragmentationError as e:
            raise e
        except Exception as e:
            raise SAFEEncodeError(f"Failed to encode {inp} with {slicer}") from e
        return encoded


def decode(
    safe_str: str,
    as_mol: bool = False,
    canonical: bool = False,
    fix: bool = True,
    remove_added_hs: bool = True,
    remove_dummies: bool = True,
    ignore_errors: bool = False,
):
    """Convert input SAFE representation to smiles
    Args:
        safe_str: input SAFE representation to decode as a valid molecule or smiles
        as_mol: whether to return a molecule object or a smiles string
        canonical: whether to return a canonical smiles or a randomized smiles
        fix: whether to fix the SAFE representation to take into account non-connected attachment points
        remove_added_hs: whether to remove the hydrogen atoms that have been added to fix the string.
        remove_dummies: whether to remove dummy atoms from the SAFE representation
        ignore_errors: whether to ignore error and return None on decoding failure or raise an error

    """
    with dm.without_rdkit_log():
        safe_obj = SAFEConverter()
        try:
            decoded = safe_obj.decoder(
                safe_str,
                as_mol=as_mol,
                canonical=canonical,
                fix=fix,
                remove_dummies=remove_dummies,
                remove_added_hs=remove_added_hs,
            )

        except Exception as e:
            if ignore_errors:
                return None
            raise SAFEDecodeError(f"Failed to decode {safe_str}") from e
        return decoded

def main():
    smiles = "O=C(C#CCN1CCCCC1)Nc1ccc2ncnc(Nc3cccc(Br)c3)c2c1"
    safe_string = encode(smiles)
    print("SAFE representation:", safe_string)
    print("SMILES representation:", decode(safe_string))

if __name__ == "main":
    main()