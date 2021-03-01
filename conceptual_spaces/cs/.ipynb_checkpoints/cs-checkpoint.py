# -*- coding: utf-8 -*-
"""
The overall conceptual space.

Also provides some utility functions.

Created on Tue Jun  6 12:15:30 2017

@author: lbechberger
"""

from math import sqrt, isinf
from . import concept as con
from . import weights as wghts
import sys
import builtins

this = sys.modules[__name__]

this._n_dim = None
this._domains = None
this._dim_names = None
this._concepts = None
this._concept_colors = None
this._no_weights = None
this._precision_digits = 15
this._epsilon = 1e-10
this._z=lambda x:x
this.dim_to_dom = {}

def init(n_dim, domains, dim_names = None):
    """Initializes a conceptual space with the given numer of dimensions and the given set of domains.
    
    'n_dim' is an integer >= 1 and 'domains' is a dictionary from domain ids to sets of dimensions.
    The optional argument 'dim_names' contains names for the individual dimensions. Its length must be identical to 'n_dim'.
    If it is not given, numbered dimension names are generated."""

    if n_dim < 1:
        raise Exception("Need at least one dimension")
    
    if not _check_domain_structure(domains, n_dim):
        raise Exception("Invalid domain structure")
        
    this._n_dim = n_dim
    this._domains = domains
    this._domain_list = domains.keys()
    this._concepts = {}
    this._concept_colors = {}
    # take care of dimension names
    if dim_names != None:
        if len(dim_names) != n_dim:
            raise Exception("Invalid number of dimension names")
        else:
            this._dim_names = dim_names
    else:
        this._dim_names = ["dim_{0}".format(i) for i in range(n_dim)]
    
    # construct default weights
    dim_weights = {}
    dom_weights = {}
    for (dom, dims) in list(domains.items()):
        dom_weights[dom] = 1.0
        local_dim_weights = {}
        for dim in dims:
            local_dim_weights[dim] = 1
        dim_weights[dom] = local_dim_weights
    for domain in domains:
        for dim in domains[domain]:
            this.dim_to_dom[str(dim)]=domain
    this._no_weights = wghts.Weights(dom_weights, dim_weights)
    this._def_dim_weights = dim_weights
    this._def_dom_weights = dom_weights
    
def _check_domain_structure(domains, n_dim):
    """Checks whether the domain structure is valid."""

    vals = [val for domain in list(domains.values()) for val in domain] # flatten values
   
    # each dimension must appear in exactly one domain
    for i in range(n_dim):
        if vals.count(i) != 1:
            return False
    
    # we need the correct number of dimensions in total
    if len(vals) != n_dim:
        return False
    
    # there are no empty domains allowed
    for (k,v) in list(domains.items()):
        if v == []:
            return False
    
    return True

def distance(x, y, weights):
    """Computes the combined metric d_C(x,y,W) between the two points x and y using the weights in 'weights'. If all weights are zero, float('inf') is returned"""
    
    if len(x) != this._n_dim or len(y) != this._n_dim:
        raise Exception("Points have wrong dimensionality")
    if not(len(weights._domain_weights)):
        return(float('inf'))
        
    '''print(x,'\nzu')
    print(y,'\nwith')
    print(weights,'\n')'''
    
    distance = 0.0
    for domain in list(this._domains.keys()):
        inner_distance = 0.0
        if not domain in weights._domain_weights:    # don't take into account domains w/o weights
            continue
        for dimension in this._domains[domain]:
            inner_distance += weights._dimension_weights[domain][dimension] * (x[dimension] - y[dimension])**2
        distance += weights._domain_weights[domain] * sqrt(inner_distance)
    return distance

def add_concept(key, concept, color = None):
    """Adds a concept to the internal storage under the given key."""
    
    if not isinstance(concept, con.Concept):
        raise Exception("Not a valid concept")
    this._concepts[key] = concept
    
    if color != None:
        this._concept_colors[key] = color

def delete_concept(key):
    """Deletes the concept with the given key form the internal storage."""
    
    if key in this._concepts:
        del this._concepts[key]
    if key in this._concept_colors:
        del this._concept_colors[key]

def between(first, middle, second, weights=None, method="crisp"):
    """Computes the betweenness relation between the three given points.
    
    Right now only uses the crisp definition of betweenness (returns either 1.0 or 0.0)."""
    
    if weights == None:
        weights = this._no_weights
    
    if method == "crisp":
        if (distance(first, middle, this._no_weights) + distance(middle, second, this._no_weights) - distance(first, second, this._no_weights)) < 0.00001:
            return 1.0
        else:
            return 0.0

    elif method == "soft":
        d1 = distance(first, middle, weights)
        d2 = distance(middle, second, weights)
        d3 = distance(first, second, weights)
        return d3 / (d1 + d2) if d1 + d2 > 0 else 1.0
    
    else:
        raise Exception("Unknown method")

def round(x):
    """Rounds the given number to a globally constant precision."""
    return builtins.round(x, this._precision_digits)

def equal(x, y):
    """Checks whether two floating point numbers are considered to be equal under the globally set precision."""
    return abs(x - y) < this._epsilon or (isinf(x) and isinf(y) and (x>0) == (y>0))

def export_concept_samples(num_samples = 100, path_to_file = './samples.csv', labeling_threshold = 0.9):
    """Samples 'num_samples' points each concept, labels it with the concept with highest membership as well
    as all concepts with a membership above the labeling_threshold (relative to the highest membership concept)
    and stores them as csv file under the given 'path_to_file'."""
    
    # collect data points
    samples = []
    for concept in list(this._concepts.values()):
        samples += concept.sample(num_samples)
    
    # collect labels
    labeled_samples = []
    for sample in samples:
        memberships = []
        for name, concept in this._concepts.items():
            memberships.append((name, concept.membership_of(sample)))
        memberships.sort(key = lambda x: x[1], reverse = True)
        l_sample = list(sample)
        l_sample.append(memberships[0][0])
        idx = 1
        threshold = labeling_threshold * memberships[0][1]
        while memberships[idx][1] > threshold:
            l_sample.append(memberships[idx][0])
            idx += 1
        labeled_samples.append(l_sample)
     
    with open(path_to_file, 'w') as f:
        for l_sample in labeled_samples:
            f.write("{0}\n".format(",".join(map(str, l_sample))))