import numpy as np
from collections import defaultdict
from itertools import product
import time

class StateSequence(object):
    sequence = [] # sequence of state intervals
    patient = None
    stroke = 2

    def __init__(self, SIs, patient, gender=-1, birthyear=1850, stroke=2):
        self.sequence = SIs
        self.patient = patient
        self.stroke = stroke
        self.gender = gender
        self.birthyear = birthyear

    def add(self, SI):
        self.sequence.append(SI)

    # returns all states in all state intervals of the sequence
    def get_states(self):
        return [SI.state for SI in self.sequence]

    # get all SI indices with the specified 'state' string
    def get_interval_indices_with_state(self, s):
        return [idx for idx, SI in enumerate(self.sequence) if SI.state == s]

    # from memory_profiler import profile
    # @profile
    def covered_by(self, pattern, strict_matching=True):
        potential_matches = self.generate_potential_matches(pattern.states)
        for p_match in potential_matches:
            if self.satisfies_relations(p_match, pattern.relations, strict_matching): return True
        return False


    def generate_potential_matches(self, pattern_states):
        potential_state_intervals = []

        previous_SIs = [0]
        for state in pattern_states:
            SIs = self.get_interval_indices_with_state(state)
            if len(SIs) == 0: return [] # if even 1 state in the pattern does not occur in the sequence, the pattern is certain not to cover the sequence
            
            SIs = SIs[np.searchsorted(SIs, previous_SIs[0]):]
            if len(SIs) == 0: return [] # if even 1 state in the pattern does not occur in the sequence, the pattern is certain not to cover the sequence

            potential_state_intervals.append(SIs)
            previous_SIs = SIs

        potential_matches = [comb for comb in product(*potential_state_intervals) if sorted(comb) == list(comb) and len(set(comb)) == len(comb)]
        return potential_matches

    def satisfies_relations(self, match_indices, rels, strict_matching):
        for i, row in enumerate(match_indices):
            for j, col in enumerate(match_indices):
                if i >= j: continue # only visit upper triangular matrix
                rel = rels[i,j]
                state_interval1 = self.sequence[row]
                state_interval2 = self.sequence[col]
                if strict_matching:
                    if not state_interval1.matches(state_interval2, rel):
                        return False
                else:
                    if not state_interval1.matches_loosely(state_interval2):
                        return False
        return True

    def __str__(self):
        sequence = [SI.__str__() for SI in self.sequence]
        return ','.join([self.patient] + sequence)
        # return '|'.join([si.__str__() for si in self.sequence])

    def __repr__(self): # for printing with lists
        return self.__str__()

    def __len__(self):
        return len(self.sequence)
