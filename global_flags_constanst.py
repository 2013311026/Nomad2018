import logging


LABELS = {}
LABELS["id"] = 0
LABELS["spacegroup"] = 1
LABELS["number_of_total_atoms"] = 2
LABELS["percent_atom_al"] = 3
LABELS["percent_atom_ga"] = 4
LABELS["percent_atom_in"] = 5
LABELS["lattice_vector_1_ang"] = 6
LABELS["lattice_vector_2_ang"] = 7
LABELS["lattice_vector_3_ang"] = 8
LABELS["lattice_angle_alpha_degree"] = 9
LABELS["lattice_angle_beta_degree"] = 10
LABELS["lattice_angle_gamma_degree"] = 11
LABELS["formation_energy_ev_natom"] = 12
LABELS["bandgap_energy_ev"] = 13


ID = 0
NUMBER_OF_TOTAL_ATOMS = 2
NUMBER_OF_TOTAL_ATOMS_LIST = [10, 20, 30, 40, 60, 80]

LOGGING_LEVEL = logging.INFO
SPACE_GROUP_PROPERTIES = {12: 4,
                          33: 4,
                          167: 12,
                          194: 24,
                          206: 24,
                          227: 48}