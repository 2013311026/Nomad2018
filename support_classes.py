class Atom:

    # Equality epsilon
    ee = 1e-6

    def __init__(self,
                 x=0.0,
                 y=0.0,
                 z=0.0,
                 t=""):

        self.x = x
        self.y = y
        self.z = z
        self.t = t

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