class Atom:

    def __init__(self,
                 x=0.0,
                 y=0.0,
                 z=0.0,
                 t=""):

        self.x = x
        self.y = y
        self.z = z
        self.t = t


    def __str__(self):

        s = "x: " + str(self.x) + \
            " y: " + str(self.y) + \
            " z: " + str(self.z) + \
            " t: " + str(self.t)

        return s

    def __repr__(self):
        return self.__str__()