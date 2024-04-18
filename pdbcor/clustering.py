from typing import Tuple, List

import Bio
import numpy as np
from sklearn.mixture import GaussianMixture
from Bio.PDB import is_aa
from Bio.PDB.Structure import Structure
from Bio.PDB.Residue import Residue
from tqdm import tqdm


class DistanceCor:
    """Distance-based clustering"""

    def __init__(
        self,
        structure: Structure,
        mode: str,
        nstates: int,
        clust_model: GaussianMixture,
        pose_estimator: GaussianMixture,
        therm_fluct: float,
        loop_start: int,
        loop_end: int,
    ):
        # HYPERVARIABLES
        self.nstates = nstates  # number of states
        self.clust_model = clust_model
        self.pose_estimator = pose_estimator
        self.therm_fluct = therm_fluct
        self.loop_start = loop_start
        self.loop_end = loop_end

        # IMPORT THE STRUCTURE
        self.structure = structure
        self.mode = mode
        self.backboneAtoms = ["N", "H", "CA", "HA", "HA2", "HA3", "C", "O"]
        self.banres = []
        self.resid = []
        self.coord_matrix = np.empty((0, 0))

    def _residue_center(self, res: Residue) -> np.ndarray:
        """Calculate the center of mass of a given residue"""
        coord = []
        atom_number = 0
        for atom in res.get_atoms():
            if self.mode == "backbone":
                if atom.id in self.backboneAtoms:
                    atom_number += 1
                    coord.extend(list(atom.get_coord()))
            elif self.mode == "sidechain":
                if atom.id not in self.backboneAtoms:
                    atom_number += 1
                    coord.extend(list(atom.get_coord()))
            else:
                atom_number += 1
                coord.extend(list(atom.get_coord()))
        coord = np.array(coord).reshape(-1, 3)

        # average coordinates of all atoms
        coord = np.mean(coord, axis=0)

        # add thermal noise to the residue center
        coord += np.random.normal(0, self.therm_fluct / np.sqrt(atom_number), 3)
        return coord

    def _residue_coords(self, chain_id: str) -> np.ndarray:
        """Get coordinates of all residues"""
        coord_list = []
        for model in self.structure.get_models():
            chain = model[chain_id]
            model_coord = []
            for res in chain.get_residues():
                if is_aa(res, standard=True):
                    if not (
                        self.mode == "sidechain" and res.get_resname() == "GLY"
                    ) and not (self.loop_end >= res.id[1] >= self.loop_start):
                        model_coord.append(self._residue_center(res))
                    else:
                        self.banres.append(res.id[1])
                        model_coord.append([0, 0, 0])
            coord_list.append(model_coord)
        return np.array(coord_list).reshape(len(self.structure), len(self.resid), 3)

    def _clust_aa(self, ind: int) -> List[int]:
        """Cluster conformers according to the distances between the ind residue with other residues"""
        features = []
        for model in range(len(self.structure)):
            for i in range(len(self.resid)):
                if i != ind:
                    # calculate Euclidean distance between residue centers
                    dist = np.sqrt(
                        np.sum(
                            (
                                self.coord_matrix[model][ind]
                                - self.coord_matrix[model][i]
                            )
                            ** 2
                        )
                    )

                    # save the distance
                    features.append(dist)

        # scale features down to the unit norm and rearange into the feature matrix
        features = np.array(features) / np.linalg.norm(np.array(features))
        features = features.reshape(len(self.structure), -1)

        # clustering
        return list(self.clust_model.fit_predict(features))

    def clust_cor(self, chain: str, resid: List[int]) -> Tuple[np.ndarray, List[int]]:
        """Get clustering matrix"""
        self.resid = resid
        self.coord_matrix = self._residue_coords(chain)
        self.resid = [i for i in self.resid if i not in self.banres]
        clusters = []
        print("DISTANCE CLUSTERING PROCESS:")
        for i in tqdm(range(len(self.resid))):
            if self.resid[i] in self.banres:
                clusters.append(self.resid[i])
                clusters.extend(list(np.zeros(len(self.structure))))
            else:
                clusters.append(self.resid[i])
                clusters.extend(self._clust_aa(i))
        return np.array(clusters).reshape(-1, len(self.structure) + 1), self.banres


# angle correlations estimator
class AngleCor:
    """Angle-based clustering"""

    def __init__(
        self,
        structure: Bio.PDB.Structure,
        mode: str,
        nstates: int,
        clust_model: GaussianMixture,
    ):
        # HYPERVARIABLES
        self.nstates = nstates  # number of states
        self.clustModel = clust_model
        self.structure = structure
        self.nConf = len(self.structure)  # number of PDB models
        self.structure.atom_to_internal_coordinates()
        allowed_angles = {
            "backbone": ["phi", "psi"],
            "sidechain": ["chi1", "chi2", "chi3", "chi4", "chi5"],
            "combined": ["phi", "psi", "chi1", "chi2", "chi3", "chi4", "chi5"],
        }
        banned_res_dict = {"backbone": [], "sidechain": ["GLY", "ALA"], "combined": []}
        self.bannedRes = banned_res_dict[mode]
        self.angleDict = allowed_angles[mode]
        self.banres = []
        self.resid = []
        self.angle_data = np.empty((0, 0))

    def _all_residues_angles(self, chainID: str) -> np.ndarray:
        """Collect angle data"""
        angles = []
        for model in self.structure.get_models():
            chain = model[chainID]
            for res in chain.get_residues():
                if is_aa(res, standard=True):
                    if res.internal_coord and res.get_resname() not in self.bannedRes:
                        entry = [res.get_full_id()[1], res.id[1]]
                        for angle in self.angleDict:
                            entry.append(res.internal_coord.get_angle(angle))
                        entry = [0 if v is None else v for v in entry]
                        angles.extend(entry)
        return np.array(angles).reshape(-1, 2 + len(self.angleDict))

    def _single_residue_angles(self, aa_id: int) -> np.ndarray:
        """Gather angles from one amino acid"""
        return self.angle_data[self.angle_data[:, 1] == aa_id, 2:]

    @staticmethod
    def correct_cyclic_angle(ang: np.ndarray) -> np.ndarray:
        """Shift angles to avoid clusters spreading over the (-180,180) cyclic coordinate closure"""
        ang_sort = np.sort(ang)
        ang_cycled = np.append(ang_sort, ang_sort[0] + 360)
        ang_max_id = np.argmax(np.diff(ang_cycled))
        ang_max = ang_sort[ang_max_id]
        ang_shift = ang + 360 - ang_max
        ang_shift = np.array([v - 360 if v > 360 else v for v in ang_shift])
        return ang_shift - np.mean(ang_shift)

    def _clust_aa(self, aa_id: int) -> np.ndarray:
        """Execute clustering of single residue"""
        aa_data = self._single_residue_angles(aa_id)
        if aa_data.shape == (0, 5):
            self.banres.append(aa_id)
            return np.zeros(self.nConf)
        # correct the cyclic angle coordinates
        for i in range(len(self.angleDict)):
            aa_data[:, i] = self.correct_cyclic_angle(aa_data[:, i])
        # CLUSTERING
        return self.clustModel.fit_predict(aa_data)

    def clust_cor(self, chain: str, resid: List[int]) -> Tuple[np.ndarray, List[int]]:
        """Get clustering matrix"""
        self.angle_data = self._all_residues_angles(chain)
        self.resid = resid
        # collect all clusterings
        clusters = []
        print("ANGLE CLUSTERING PROCESS:")
        for i in tqdm(range(len(self.resid))):
            clusters.append(self.resid[i])
            clusters.extend(list(self._clust_aa(self.resid[i])))
        return np.array(clusters).reshape(-1, self.nConf + 1), self.banres