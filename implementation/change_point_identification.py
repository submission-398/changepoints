#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import itertools

import z3
from scipy.special import comb
from typing import Dict, Tuple, Sequence

from multiprocessing import Pool

import random

import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import uniform as sp_rand

import sklearn.linear_model as lm
import sklearn.tree as tree
import sklearn.ensemble as ensemble

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import RandomizedSearchCV

from abc import ABC, abstractmethod
from bisect import bisect_left


class DataProvider(ABC):

    @abstractmethod
    def oracle(config, commit):
        pass


class Synthesizer(DataProvider):
    """
    Generator for performance data for configurable software systems
    """

    def __init__(
            self,
            n_commits,
            n_options,
            n_cps,
            p_interaction,
            p_interaction_degree,
            cov,
            seed
    ):
        '''
        Parameters
        ----------
        n_commits : int
            number of commits
        n_options : int
            number of binary options.
        n_cps : int
            number of change points for option/interaction influence.
        p_interaction : float
            percentage of additional interactions (default: 0.1, adds 10% * n_options interactions)
        p_interaction_degree : float
            parameter for interaction degrees, the interactions
            follow an beta-distribution (default: 0.8, higher -> less higher-order interactions,
            lower -> more higher-order interactions).
        cov : float
            coefficient of variation for measurements (standard deviation normalized to mean/location).
        seed : int
            seed parameter for model for reproducible results.

        Returns
        -------
        None.

        '''
        self.n_commits = n_commits
        self.n_options = n_options
        self.n_cps = n_cps
        self.p_interaction = p_interaction
        self.p_interaction_degree = p_interaction_degree
        self.cov = cov
        self.seed = seed

        # apply seeds
        np.random.seed(seed)
        random.seed(seed)

        # pre-compute influences
        self.__compute_influences()


    def __compute_influences(self):
        '''
        Compute influences of different options/interactions used for synthesis

        Returns
        -------
        None.

        '''

        # list of configuration options
        options = list(map(lambda x: x, np.arange(self.n_options)))

        # number of interactions
        n_interactions = int(self.p_interaction * self.n_options)

        # sampled distribution of interaction weights
        degrees = np.random.geometric(p=self.p_interaction_degree, size=n_interactions)
        unique, counts = np.unique(degrees, return_counts=True)
        degrees = dict(zip(unique, counts))

        # Compose terms of the performance-influence model
        # term = one (single option) or more configuration options (interaction)
        terms = list(map(lambda x: tuple([x]), options))
        for degree in degrees:
            available_terms = list(itertools.combinations(options, degree + 1))
            if len(available_terms) > 1:
                term_indexes = np.random.choice(np.arange(len(available_terms)), size=degrees[degree])
                terms_ = [available_terms[f] for f in term_indexes]
                terms += terms_
        terms.append(())

        # Compose a matrix (#terms, #options) encoding all terms
        self.term_matrix = pd.DataFrame([[1 if j in term else 0 for j in range(self.n_options)] for term in terms])
        self.term_matrix.columns = ["opt" + str(i) for i in range(self.n_options)]

        # create mapping from terms to options
        self.terms = {t: [o for o in terms[t]] for t in range(len(terms))}

        # Create matrix with initial performance-influences
        base = np.random.random(size=self.term_matrix.shape[0])
        self.influences = pd.DataFrame(np.tile(base, (self.n_commits, 1)))
        self.influences.columns = [i for i in range(len(terms))]  # term

        # Create the change points and adjust influences
        changepoints = np.random.choice(np.arange(self.n_commits), size=self.n_cps)
        self.changepoints = []
        for loc in changepoints:
            # determine term to which apply change
            term = np.random.choice(self.influences.columns, size=1)[0]

            # current influence
            current = self.influences[term].iloc[loc]

            # determine relative size of change point
            shift = np.random.choice([-1, 1]) * np.random.choice([0.5, 0.9, 0.99, 1.01, 1.1, 2])

            # adjust influences for the term
            self.influences[term].iloc[loc:] = current + (shift * current)

            # record change points
            cp = dict()
            cp["loc"] = loc
            cp["shift"] = shift
            cp["term"] = term
            self.changepoints.append(cp)
        self.changepoints = pd.DataFrame(self.changepoints)

    def oracle(self, config, commit):
        '''
        Compute/Predict performance or performance history for a given configurations

        Parameters
        ----------
        config : np.ndarray
            configuration vector (bit vector).
        commit : int
            number of the commit to predict performance for or -1 (yields entire performance history).

        Returns
        -------
        float or numpy.ndarray
            performance value (scalar) or performance history (vector).

        '''

        # translate configuration over options to configuration over performance-influence model terms
        term_config = 1 * np.equal(np.dot(config, self.term_matrix.T), np.sum(self.term_matrix, axis=1))

        # compute performance history
        perf_hist = pd.DataFrame(np.dot(term_config, self.influences.T))

        # entire performance history with noise
        if commit == -1:
            noise_vector = np.random.normal(0, np.abs(self.cov * perf_hist.iloc[commit]), size=perf_hist.shape[0])
            perf_hist = perf_hist.values + noise_vector
            return perf_hist  # .values

        # single performance value with noise
        noise = np.random.normal(0, np.abs(self.cov * perf_hist.iloc[commit]))
        perf_hist_single = perf_hist.iloc[commit].values[0] + noise
        return perf_hist_single

def sample_commits(
        n_configs,
        n_commits,
        commit_srate
):
    '''
    Select a sample of commits per configurations

    Parameters
    ----------
    n_configs : int
    number of configurations

    n_commits : int
    number of commits of the software system

    commit_srate : float
    percentage of commits to sample per configuration

    Returns
    -------

    Vector (#configs, #commits:per:sample) with configurations
    '''

    commits_to_sample = max(2, int(n_commits * commit_srate))
    commit_matrix = []  # np.zeros(shape=(n_configs, commits_to_sample), dtype=np.int8)

    for c in range(n_configs):
        sample = [0] + np.random.choice(np.arange(n_commits), size=commits_to_sample).tolist() + [n_commits - 1]
        commit_matrix.append(sample)
    commit_matrix = np.vstack(commit_matrix)
    return commit_matrix  # configs


def construct_sample(
        configs,
        commits
):
    assert configs.shape[0] == len(commits)
    sample = []
    for i, conf in enumerate(configs):
        sample.append({"config": conf, "commit": commits[i]})

    return sample


def estimate_change_point_likelihood(
        perf,
        threshold,
        n_commits,
        absolute_threshold=False
):
    '''
    Estimates the change point likelihood for a given set of performance measurements.
    Parameters
    ----------
    perf : dict
    Loader for performance measurements or synthesizer
    commits
    threshold : float
    Threshold that defines a substantial performance change. Can either be a relative or absolute threshold.
    A performance change is substantial if the change exceeds this threshold.

    n_commits : int
    number of commits

    absolute_threshold : bool
    Enables absolute threshold, performance changes are substantial if they exceed this absolute value.

    Returns
    -------

    Vector with change point likelihoods

    '''
    cpl = np.zeros(n_commits)
    for a, b in itertools.combinations(perf.keys(), 2):
        diff = np.abs(perf[a][0] - perf[b][0])
        threshold_a = np.abs(threshold * perf[a][0])
        threshold_b = np.abs(threshold * perf[b][0])

        if absolute_threshold:
            if diff > threshold:
                cpl[min(a, b): max(a, b)] += 1 / np.abs(a - b) ** 2
        if 1:
            if any([diff > threshold_a, diff > threshold_b]):
                cpl[min(a, b): max(a, b)] += 1 / np.abs(a - b) ** 2

    # if no change point was detected return a uniform distribution + normalize
    if np.sum(cpl) == 0.0:
        cpl[:] = 1.0 / n_commits
    else:
        cpl = cpl / np.nansum(cpl)

    return cpl  # scl.fit_transform(cpl).reshape((n_commits,))


def take_closest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return after
    else:
        return before


def hamming(V1, V2, target):
    h = V1 ^ V2
    s = max(target.bit_length(), V1.size().bit_length())
    return z3.Sum([z3.ZeroExt(s, z3.Extract(i, i, h)) for i in range(V1.size())])


def query_configurations(
        path_to_dimacs: str,
        configs: Sequence[Sequence[int]],
        n_options: int,
        max_solutions: int = 10,
        ignore_vm: bool = False,
        set_vary: Sequence[int] = []
):
    '''
    This function allows sampling a configuration over binary options
    based on two sources of constraints:

        a) a variability/feature model specified in DIMACS format
        b) existing configurations to which one can specify a minimum Hamming distance

    Parameters:
        path_to_dimacs: path to variability model
        configs: list of lists of integers (0s and 1s)
        n_options: number of options in the variabiltiy model
        max_solutions: maximum number of configurations to sample
        ignore_vm: ignores the variability model (if no constraints are used)
    '''

    # Parse variability model in DIMACS format and add constraints
    if not ignore_vm:
        dimacs = list()
        dimacs.append(list())
        with open(path_to_dimacs) as mfile:
            lines = list(mfile)
            N = int(lines[0].split(" ")[2])

            assert N == n_options

            # parse lines 
            for line in lines:
                tokens = line.split()
                if len(tokens) != 0 and tokens[0] not in ("p", "c"):
                    for tok in tokens:
                        lit = int(tok)
                        if lit == 0:
                            dimacs.append(list())
                        else:
                            dimacs[-1].append(lit)
        assert len(dimacs[-1]) == 0
        dimacs.pop()

        # add clauses of variability model 
        for clause in dimacs:
            c = []
            for opt in clause:
                opt_sign = 1 if opt >= 0 else 0
                optid = N - abs(opt)
                c.append(z3.Extract(optid, optid, target) == opt_sign)
            solver.add(z3.Or(c))

    solutions = []
    solution_vectors = []
    iterations_count = 0

    # create matrix of configurations
    if len(configs) == 0:
        config = [0 for i in range(n_options)]
        configs.append(config)

    while len(solution_vectors) < max_solutions and iterations_count < 1000:
        #print("query")

        solver = z3.Solver()
        target = z3.BitVec('target', n_options)

        for sol_vector in solution_vectors:
            solver.add(target != sol_vector)

        ''' Add configuration-based constraints - do not repeat previous configuration'''
        config_strings = ["".join(list(map(str, conf))) for conf in configs]
        for config in config_strings:
            vec = z3.BitVecVal(config, n_options)
            solver.add(target != vec)

        if len(set_vary) == 0: # Add distance constraint
            origin = z3.BitVecVal("0" * n_options, n_options)
            distance = int(np.random.choice(np.arange(n_options)[1:-1]))
            solver.add(hamming(origin, target, 1) == distance)

        else: # Force specific options to be de-selected
            set_deselected = set(list(range(n_options))) - set(set_vary)
            set_deselected = np.random.choice(list(set_deselected), size=min(1, len(set_deselected) // 2), replace=False)
            for selected_opt in set_deselected:
                option = int(n_options - selected_opt)
                try:
                    solver.add(z3.Extract(option, option, target) == 0)
                except:
                    pass

        if solver.check() == z3.sat:
            mod = bin(solver.model()[target].as_long())
            solutions.append(list(map(int, list('0' * (n_options - (len(mod) - 2)) + str(mod)[2:]) )))

            solution_vectors.append(solver.model()[target])

        iterations_count += 1

    return solutions

class ChangePointLearner:

    def __init__(self, synthesizer, m_measurements_per_iteration: int):
        '''
        Active Learner vor Change Points.

        Parameters
        ----------
        synthesizer : Synthesizer
            Loader for performance measurements or synthesizer

        Returns
        -------
        None.
        '''

        self.synth = synthesizer

        self.candidate_solutions = []

        self.m_measurements_per_iteration = m_measurements_per_iteration

        ''' parameters for acquisition step'''
        self.r_commit_explore = 0.5
        self.r_commits_to_configs = 0.5
        # self.r_config_explore = 0.1

        self.s_commit_explore = 0.9
        self.s_commits_to_configs = 0.95
        self.s_config_explore = 1.0 # 1.0 = does not change

        self.n_configs_for_exploration = 75

        self.cached_solutions = dict()
        self.cached_solutions_keys = []

    def build_likelihoods(self,
                          sample,
                          threshold=0.01,
                          absolute_threshold=False
                          ):

        self.sample = sample  # self.n_configs_for_exploration = 20# shortcuts
        n_commits = self.synth.n_commits
        n_options = self.synth.n_options

        change_point_likelihoods = []

        heights = []
        for i, s in enumerate(sample):
            # obtain or estimate performance values
            perf = {v: self.synth.oracle(s["config"], v) for v in s["commit"]}
            likelihood = estimate_change_point_likelihood(perf, threshold, n_commits)
            change_point_likelihoods.append(likelihood)
            heights += find_peaks(likelihood, height=(None, None))[1]["peak_heights"].tolist()
        median = np.nanmedian(heights)
        change_point_likelihoods = np.array(change_point_likelihoods)

        self.change_point_likelihoods = change_point_likelihoods

    def get_ground_truth(self):
        ground_truth = self.synth.changepoints
        ground_truth_peaks = []
        for i in range(len(ground_truth["loc"])):
            term = ground_truth["term"][i]
            if len(self.synth.terms[term]) > 0:
                opts = self.synth.terms[term]
            else:
                opts = [-1]

            loc = ground_truth["loc"][i]
            for opt in opts:
                ground_truth_peaks.append((loc, opt))

        return ground_truth_peaks

    def calc_candidate_solution(self, N=5):

        # shortcut
        n_options = self.synth.n_options
        n_commits = self.synth.n_commits

        ys = []
        xs = []
        for a in range(len(self.sample)):
            threshold = N * np.nanstd(self.change_point_likelihoods[a])
            y = self.change_point_likelihoods[a] > (1.0 / n_commits) + threshold
            x = self.sample[a]["config"]
            xs.append(x)
            ys.append(y)
        xs = np.vstack(xs)
        ys = np.vstack(ys)

        # Estimate change point locations
        #cp_locations = find_peaks(np.sum(ys, axis=0))[0]
        pxs = []
        for a in range(len(self.sample)):
            cpl = self.change_point_likelihoods[a]
            peaks = find_peaks(cpl, ((1.0 / n_commits) + N * np.nanstd(self.change_point_likelihoods[a])))[0]
            for peak in peaks:
                pxs.append(peak)

        bandwidths = 10 ** np.linspace(-1, 1, 100)
        grid = GridSearchCV(KernelDensity(kernel='exponential'), {'bandwidth': bandwidths})
        pxs = np.array(pxs)
        grid.fit(pxs.reshape(-1, 1))
        kde = grid.best_estimator_
        estimated_density = np.exp(kde.score_samples(np.arange(n_commits).reshape(-1, 1)))

        cp_locations = find_peaks(estimated_density)[0]

        cp_estimations = []

        for i, v in enumerate(cp_locations):
            try:
                estimator = lm.LassoCV(positive=True)
                estimator.fit(xs, self.change_point_likelihoods[:, v])
                weight = np.abs(estimator.coef_)
                weight = weight / np.nansum(weight)
            except:
                weight = np.full(n_options, 1.0 / n_options)

            options = np.argwhere(weight > 1.0 / n_options + np.nanstd(weight)).reshape(1, -1)  # [0]
            for opt in options[0]:
                entry = {"commit": v, "option": opt}
                cp_estimations.append(entry)

        self.candidate_solutions.append(cp_estimations)
        return cp_estimations

    def score(
            self,
            ground_truth=None,
            estimated=[],
            return_f1=False,
            tolerance=5,
    ):

        ground_truth = self.synth.changepoints
        ground_truth_peaks = []
        for i in range(len(ground_truth["loc"])):
            term = ground_truth["term"][i]
            if len(self.synth.terms[term]) > 0:
                opts = self.synth.terms[term]
            else:
                opts = [-1]

            loc = ground_truth["loc"][i]
            for opt in opts:
                ground_truth_peaks.append((loc, opt))

        """ 
        Compute true and false positives. We count an estimated change point as
        a true positive if it is within 5 commits of an actual change point.
        """
        true_positives = 0
        false_positives = 0
        for loc, opt in ground_truth_peaks:
            tolerances = [(loc + i, opt) for i in range(tolerance)]
            tolerances += [(loc - i, opt) for i in range(1, tolerance)]

            clauses = [(tol in estimated) for tol in tolerances]
            if len(clauses) > 0:
                if any(clauses):
                    true_positives += 1
                else:
                    false_positives += 1
            else:
                false_positives += 1

        """ Compute precision and recall from true and false positives """
        result = dict()
        result["precision"] = true_positives / max(1, len(estimated))
        result["recall"] = true_positives / max(1, len(ground_truth_peaks))
        if return_f1:
            if result["precision"] == 0 and result["recall"] == 0:
                result["f1_score"] = 0.0
            else:
                result["f1_score"] = 2 * (result["precision"] * result["recall"]) / (
                            result["precision"] + result["recall"])

        count = 0
        for s in self.sample:
            count += len(s["commit"])
        result["configs"] = len(self.sample)
        result["measurements"] = count

        print(ground_truth_peaks)
        return result

    def explore_commits(self):
        # shortcut
        n_options = self.synth.n_options
        n_commits = self.synth.n_commits

        indices_list = []
        for s in self.sample:
            commits = sorted(s["commit"])
            min_distance = np.zeros(n_commits)
            for c in range(n_commits):
                if c not in commits:
                    dist = np.abs(c - take_closest(commits, c))
                    min_distance[c] = dist
            indices = sorted(range(len(min_distance)), key=lambda k: min_distance[k], reverse=True)
            indices_list.append(indices[:10])

        return indices_list

    def exploit_commits(self, N=5):

        # shortcut
        n_options = self.synth.n_options
        n_commits = self.synth.n_commits

        indices_list = []
        for a in range(len(self.sample)):
            threshold = N * np.nanstd(self.change_point_likelihoods[a])
            y1 = self.change_point_likelihoods[a] < (1.0 / n_commits) + threshold
            y2 = self.change_point_likelihoods[a] > (1.0 / n_commits)
            y = np.logical_and(y1, y2)
            indices = np.argwhere(y == True).reshape(1, -1).tolist()[0]
            indices_list.append(indices)

        return indices_list

    def acquire_configurations(self, max_explore=3):
        '''
        Explore configurations
        '''
        # shortcut
        n_options = self.synth.n_options

        existing_configs = [s["config"] for s in self.sample]
        explore_configs = query_configurations("", existing_configs, n_options, ignore_vm=True,
                                               max_solutions=max_explore)

        ''' Exploit commits:

        '''
        existing_configs2 = existing_configs + explore_configs

        exploit_configs = []  # container
        candidates = dict()
        for candidate in self.candidate_solutions[-1]:
            if candidate["commit"] not in candidates:
                candidates[candidate["commit"]] = []
            candidates[candidate["commit"]].append(candidate["option"])

        for key in candidates:
            options = candidates[key]  # options = candidate[]
            exploit_configs += query_configurations("", existing_configs2, n_options, ignore_vm=True, max_solutions=3, set_vary=options)
            inverse_options = list(filter(lambda o: o not in options, np.arange(n_options)))
            exploit_configs += query_configurations("", existing_configs2, n_options, ignore_vm=True, max_solutions=3, set_vary=inverse_options)

        return explore_configs, exploit_configs

    def next_sample(self):

        # shortcut
        n_commits = self.synth.n_commits

        ''' Add exploration and exploitation commits to existing sample '''
        sample = self.sample

        n_commits_for_exploration = int(
            self.m_measurements_per_iteration * self.r_commits_to_configs * self.r_commit_explore)
        n_commits_for_exploitation = int(
            self.m_measurements_per_iteration * self.r_commits_to_configs * (1 - self.r_commit_explore))

        # add exploration commits:
        commits_per_config = int(n_commits_for_exploration / len(sample))
        for config_id, s in enumerate(sample):
            not_measured = list(set(np.arange(n_commits)) - set(sample[config_id]["commit"]))
            to_add = np.random.choice(not_measured, size=commits_per_config)
            sample[config_id]["commit"] = np.append(sample[config_id]["commit"], to_add)

        # add exploitation commits:
        commits_per_config = int(n_commits_for_exploitation / len(sample))
        exploit_indices_list = self.exploit_commits()
        for config_id, s in enumerate(sample):
            to_add = np.array(exploit_indices_list[config_id][:commits_per_config]).astype(int)
            sample[config_id]["commit"] = np.append(sample[config_id]["commit"], to_add)

        explore_configs, exploit_configs = self.acquire_configurations(max(self.n_configs_for_exploration, 1))

        n_explore_configs = len(explore_configs)
        n_exploit_configs = len(exploit_configs)

        n_new_configs = n_explore_configs + n_exploit_configs
        measurements_left = self.m_measurements_per_iteration - n_commits_for_exploration - n_commits_for_exploitation

        commits_per_config = int(measurements_left / max(1, n_new_configs))
        for config in explore_configs:
            sample.append({"config": config, "commit": np.random.choice(np.arange(n_commits), size=commits_per_config)})
        for config in exploit_configs:
            sample.append({"config": config, "commit": np.random.choice(np.arange(n_commits), size=commits_per_config)})

        self.r_commit_explore *= self.s_commit_explore
        self.r_commits_to_configs *= self.s_commits_to_configs
        self.n_configs_for_exploration *= self.s_config_explore

        return sample

    def caching(self, evapo_rate=0.3, n_rounds=3):

        current_candidate_solution = self.candidate_solutions[-1]

        # evaporate
        for cs in self.cached_solutions:
            self.cached_solutions[cs] *= evapo_rate

        # increase
        for cs in current_candidate_solution:

            cands = [(cs["commit"] + tol, cs["option"]) for tol in np.arange(-5, 6)]
            existing = [cand in self.cached_solutions for cand in cands]

            # case 1) solution appears fdor the first time
            if not any(existing):
                self.cached_solutions[(cs["commit"], cs["option"])] = 1.0

            # case 2) solution already exists (with 5 commits tolerance)
            else:
                index = [i for i, x in enumerate(existing) if x][0]
                # print(index)
                self.cached_solutions[cands[index]] += 1.0

        # remove solutions with weights smaller than 0.125
        keys_to_delete = list(filter(lambda key: self.cached_solutions[key] < evapo_rate ** n_rounds,
                                     self.cached_solutions.keys()))  # [key if  for key in self.cached_solution.keys()]
        for key_to_delete in keys_to_delete:
            print(">> dropped ", key_to_delete)
            del self.cached_solutions[key_to_delete]

        self.cached_solutions_keys.append(list(self.cached_solutions.keys()))

    def stop(self):
        if len(self.cached_solutions_keys) < 3:
            return False
        else:  # >= 3
            # all weights > 1
            greater_than_one = [self.cached_solutions[key] > 1.0 for key in self.cached_solutions_keys[-1]]
            if len(self.cached_solutions) > 0 and self.cached_solutions_keys[-1] == self.cached_solutions_keys[-2] and \
                    self.cached_solutions_keys[-2] == self.cached_solutions_keys[-3] and all(greater_than_one):
                return True
            else:
                return False

# Example run
if __name__ == "__main__":
    synth = Synthesizer(2000, 16, 3, 0.25, 0.95, 0.00, 298)

    confs = np.vstack(list(map(lambda x: np.array(x), query_configurations("", [], 16, 25, ignore_vm=True, set_vary=[]))))

    revs = sample_commits(len(confs), 2000, 0.05)
    sample = construct_sample(confs, revs)
    learner = ChangePointLearner(synth, m_measurements_per_iteration=5000)

    precisions = []
    recalls = []
    f1s = []
    for i in range(30):
        if not learner.stop():

            learner.build_likelihoods(sample)

            learner.calc_candidate_solution()

            learner.acquire_configurations()

            sample = learner.next_sample()

            learner.caching()

            print(learner.stop())

            scores = learner.score(estimated=learner.cached_solutions.keys(), return_f1=True)
            precisions.append(scores["precision"])
            f1s.append(scores["f1_score"])
            recalls.append(scores["recall"])

            print(scores)

    print(synth.changepoints)
    print(synth.term_matrix)
