import numpy as np
from operator import attrgetter
from scipy.stats import binom

class Pattern(object):
    states = None
    relations = None
    support = None
    confidence = None

    subpatterns = None
    id_list = None
    pid_list = None

    in_candidates = False

    is_frequent = False

    # mptp stuff
    mcs = None
    mc = None

    def __init__(self, s, r, sup=None, conf=None, subpatterns=None, id_list=None, in_candidates=False):
        self.states = s
        self.relations = r
        self.flat_relations = r.flatten()
        self.support = sup
        self.confidence = conf
        self.subpatterns = subpatterns
        self.id_list = id_list
        self.in_candidates = in_candidates

    def __str__(self):
        return ','.join(np.append(self.states, self.flat_relations))
        s = 'States: ' + ';'.join(self.states) + ' | '
        s = s + 'Relations: ' + self.relations.tostring() + ' | '
        # s = s + 'Support: ' + str(self.support) + ' | '
        # s = s + 'Confidence: ' + str(self.confidence) + ' | '
        s = s + 'pid list: ' + str(self.pid_list) + ' | '
        s = s + 'id list: ' + str(self.id_list) + ' | '
        s = s + 'support: ' + str(self.support)
        return '\n' + s

    def __repr__(self): # for printing with lists
        return self.__str__()

    def __hash__(self):
        return int(len(self.states) + np.sum('c' == self.flat_relations) )

    def __eq__(self, other):
        return np.array_equal(self.states, other.states) and np.array_equal(self.relations, other.relations)

    def get_subpatterns(self):
        if self.subpatterns != None:
            return self.subpatterns
        
        result = []
        for i, s in enumerate(self.states):
            # remove state from list
            new_states=np.delete(self.states,i)

            # remove relations from matrix
            new_rels = np.delete(self.relations, i, 0)
            new_rels = np.delete(new_rels, i, 1)
            # try:
            pattern = Pattern(new_states, new_rels)
            result.append(pattern)
            # result = result + pattern.get_subpatterns(Fk)
            # except:
            #     print "encountered a subpattern not in the dict. Should be fine."
        self.subpatterns = set(result)
        return self.subpatterns

    def generate_pid_list(self, Fk, negated=False):
        if not negated:
            id_lists = [pattern.id_list for pattern in Fk]
            self.pid_list = set(set(id_lists[0]).intersection(*id_lists))
            return self.pid_list
        else:
            id_lists = [pattern.id_list_negated for pattern in Fk]
            self.pid_list_negated = set(set(id_lists[0]).intersection(*id_lists))
            return self.pid_list_negated

    def calc_max_conf_subpatterns(self, Fk):
        self.mcs = 0
        for p in Fk: 
            if p.mc > self.mcs:
                self.mcs = p.mc
        # print self.mcs
        # self.mcs = max(Fk, key=attrgetter('mc'))
        # print self.mcs
        return self.mcs

    def generate_id_list(self, dct, negated=False):
        if self.pid_list == None:
            if not negated:
                self.id_list = [key for key in dct if dct[key].covered_by(self)]
            else:
                self.id_list_negated = [key for key in dct if dct[key].covered_by(self)]
        else:
            if not negated:
                self.id_list = [key for key in dct if key in self.pid_list and dct[key].covered_by(self)]
            else:
                self.id_list_negated = [key for key in dct if key in self.pid_list_negated and dct[key].covered_by(self)]

    def calculate_support(self, min_sup, num_data_points):
        self.support = len(self.id_list)/float(num_data_points)
        return self.support

    def calc_max_conf(self):
        self.mc = max(self.confidence, self.mcs)
        return self.mc

    def is_significant(self, alpha=0.05):
        Ny = len(self.id_list)
        N = Ny + len(self.id_list_negated)
        p_value = binom.sf(Ny, N, self.mcs)
        # print Ny, N, self.mcs, p_value
        return p_value < alpha

    def get_p_value(self):
        Ny = len(self.id_list)
        N = Ny + len(self.id_list_negated)
        p_value = binom.sf(Ny, N, self.mcs)
        return p_value

def enrich_pattern(P, S, new_rels, data_dict):
    states = np.insert(P.states, 0, S.states[0]) # insert singleton state to the rest
    row, col = P.relations.shape # original shape
    relations = np.empty((row+1, col+1), dtype='S1') # create a slightly larger matrix
    relations.fill(' ')
    # print relations
    relations[1:,1:] = P.relations # add old values to matrix
    # print relations
    relations[0,:] = new_rels # add new relation values to first row of matrix
    result = Pattern(states, relations) # new pattern
    # print result.relations
    return result