class Atom:

    # Equality epsilon
    ee = 1e-6

    def __init__(self,
                 x=0.0,
                 y=0.0,
                 z=0.0,
                 t="",
                 c=0):

        self.x = x
        self.y = y
        self.z = z
        self.t = t
        self.c = c

    def __eq__(self, other):
        if (abs(self.x - other.x) < Atom.ee and
            abs(self.y - other.y) < Atom.ee and
            abs(self.z - other.z) < Atom.ee and
            self.t == other.t):

            return True
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)


    def __hash__(self):
        s = str(self.x) + str(self.y) + str(self.z) + str(self.t)
        return hash(s)


    def __str__(self):

        s = "x: " + str(self.x) + \
            " y: " + str(self.y) + \
            " z: " + str(self.z) + \
            " t: " + str(self.t)

        return s

    def __repr__(self):
        return self.__str__()


class UCAtoms(list):

    def _count_atoms_of_type_t(self, t):
        n_atoms = 0
        for i in range(len(self)):
            if self[i].t == t:
                n_atoms = n_atoms + 1

        return n_atoms

    def __init__(self, *args):
        super().__init__(self, *args)

        self.n_ga_atoms = self._count_atoms_of_type_t("Ga")
        self.n_al_atoms = self._count_atoms_of_type_t("Al")
        self.n_in_atoms = self._count_atoms_of_type_t("In")
        self.n_o_atoms = self._count_atoms_of_type_t("O")
