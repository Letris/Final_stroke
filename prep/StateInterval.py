class StateInterval(object):
    state = ""
    begin = ""
    end = ""

    def matches(self, other, rel):
        return (self.is_before(other) and rel=='b') or (self.co_occurs(other) and rel=='c')

    def matches_loosely(self, other):
        return self.is_before(other) or self.co_occurs(other)

    def is_before(self, other):
        return self.end < other.begin

    def co_occurs(self, other):
        return self.begin <= other.begin and other.begin <= self.end

    def __init__(self, s, b, e):
        self.state = s
        self.begin = b
        self.end = e

    def __str__(self):
        return '; '.join([self.state, str(self.begin), str(self.end)])

    def __repr__(self): # for printing with lists
        return self.__str__()

def new_SI(s, b, e):
    return StateInterval(s, b, e)
