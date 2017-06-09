# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 10:51:06 2017

@author: lbechberger
"""

import unittest
import sys
sys.path.append("..")
from cs.core import Core
from cs.cuboid import Cuboid
import cs.cs

class TestCore(unittest.TestCase):
    
    # constructor
    def test_constructor_no_arg(self):
        with self.assertRaises(Exception):
            Core([])
    
    def test_constructor_correct_arg(self):
        cs.cs.ConceptualSpace(3, {0:[0,1,2]})
        c1 = Cuboid([1,2,3],[4,5,6], {0:[0,1,2]})
        c2 = Cuboid([2,3,4],[5,6,7], {0:[0,1,2]})
        l = [c1, c2]
        s = Core(l, {0:[0,1,2]})
        self.assertEqual(s._cuboids, l)
    
    def test_constructor_no_list(self):
        with self.assertRaises(Exception):
            Core(42, {0:[0,1,2]})
    
    def test_constructor_no_cuboid_list(self):
        cs.cs.ConceptualSpace(2, {0:[0,1]})
        with self.assertRaises(Exception):
            Core([Cuboid([1,2],[3,4], {0:[0,1]}), 42, "test"], {0:[0,1]})
    
    def test_constructor_nonintersecting(self):
        cs.cs.ConceptualSpace(3, {0:[0,1,2]})
        c1 = Cuboid([1,2,3],[4,5,6], {0:[0,1,2]})
        c2 = Cuboid([0,0,0],[1,1,1], {0:[0,1,2]})
        l = [c1, c2]
        with self.assertRaises(Exception):
            Core(l, {0:[0,1,2]})

    def test_constructor_different_relevant_dimensions(self):
        cs.cs.ConceptualSpace(3, {0:[0], 1:[1], 2:[2]})
        c1 = Cuboid([float("-inf"),2,3],[float("inf"),5,6], {1:[1], 2:[2]})
        c2 = Cuboid([2,float("-inf"),4],[5,float("inf"),7], {0:[0], 2:[2]})
        with self.assertRaises(Exception):
            Core([c1, c2], {0:[0], 1:[1], 2:[2]})

    def test_constructor_same_relevant_dimensions(self):
        cs.cs.ConceptualSpace(3, {0:[0], 1:[1,2]})
        c1 = Cuboid([float("-inf"),2,3],[float("inf"),5,6], {1:[1,2]})
        c2 = Cuboid([float("-inf"),3,4],[float("inf"),6,7], {1:[1,2]})
        s = Core([c1, c2], {1:[1,2]})
        self.assertEquals(s._cuboids, [c1, c2])

    # _check
    def test_check_true(self):
        cs.cs.ConceptualSpace(3, {0:[0,1,2]})
        c1 = Cuboid([1,2,3],[4,5,6], {0:[0,1,2]})
        c2 = Cuboid([2,3,4],[5,6,7], {0:[0,1,2]})
        c3 = Cuboid([2,2,2],[12.4,12.5,12.6], {0:[0,1,2]})
        l = [c1, c2, c3]
        s = Core(l, {0:[0,1,2]})
        self.assertTrue(s._check())
    
    def test_check_false(self):
        cs.cs.ConceptualSpace(3, {0:[0,1,2]})
        c1 = Cuboid([1,2,3],[4,5,6], {0:[0,1,2]})
        c2 = Cuboid([0,0,0],[1,1,1], {0:[0,1,2]})
        c3 = Cuboid([1,1,1],[2,3,4], {0:[0,1,2]})
        l = [c1, c2, c3]
        s = Core([c1], {0:[0,1,2]})
        self.assertFalse(s._check(l))
    
    # add_cuboid
    def test_add_cuboid_true(self):
        cs.cs.ConceptualSpace(3, {0:[0,1,2]})
        c1 = Cuboid([1,2,3],[4,5,6], {0:[0,1,2]})
        c2 = Cuboid([2,3,4],[5,6,7], {0:[0,1,2]})
        c3 = Cuboid([2,2,2],[12.4,12.5,12.6], {0:[0,1,2]})
        l = [c1]
        s = Core(l, {0:[0,1,2]})
        self.assertTrue(s.add_cuboid(c2))
        self.assertEqual(s._cuboids, [c1, c2])
        self.assertTrue(s.add_cuboid(c3))
        self.assertEqual(s._cuboids, [c1, c2, c3])

    def test_add_cuboid_false(self):
        cs.cs.ConceptualSpace(3, {0:[0,1,2]})
        c1 = Cuboid([1,2,3],[4,5,6], {0:[0,1,2]})
        c2 = Cuboid([0,0,0],[1,1,1], {0:[0,1,2]})
        c3 = Cuboid([1,1,1],[2,3,4], {0:[0,1,2]})
        l = [c1]
        s = Core(l, {0:[0,1,2]})
        self.assertFalse(s.add_cuboid(c2))
        self.assertEqual(s._cuboids, [c1])
        self.assertTrue(s.add_cuboid(c3))
        self.assertEqual(s._cuboids, [c1, c3])

    def test_add_cuboid_no_cuboid(self):
        cs.cs.ConceptualSpace(3, {0:[0,1,2]})
        c1 = Cuboid([1,2,3],[4,5,6], {0:[0,1,2]})
        l = [c1]
        s = Core(l, {0:[0,1,2]})
        with self.assertRaises(Exception):
            s.add_cuboid(42)
        self.assertEqual(s._cuboids, [c1])
    
    def test_add_cuboid_different_relevant_dimensions(self):
        cs.cs.ConceptualSpace(3, {0:[0], 1:[1], 2:[2]})
        c1 = Cuboid([float("-inf"),2,3],[float("inf"),5,6], {1:[1], 2:[2]})
        c2 = Cuboid([2,float("-inf"),4],[5,float("inf"),7], {0:[0], 2:[2]})
        s1 = Core([c1], {1:[1], 2:[2]})
        s2 = Core([c2], {0:[0], 2:[2]})
        self.assertFalse(s1.add_cuboid(c2))
        self.assertFalse(s2.add_cuboid(c1))

    def test_add_cuboid_same_relevant_dimensions(self):
        cs.cs.ConceptualSpace(3, {0:[0], 1:[1,2]})
        c1 = Cuboid([float("-inf"),2,3],[float("inf"),5,6], {1:[1,2]})
        c2 = Cuboid([float("-inf"),3,4],[float("inf"),6,7], {1:[1,2]})
        s1 = Core([c1], {1:[1,2]})
        s2 = Core([c2], {1:[1,2]})
        self.assertTrue(s1.add_cuboid(c2))
        self.assertTrue(s2.add_cuboid(c1))
        self.assertEqual(s1, s2)
 
    # find_closest_point_candidates
    def test_find_closest_point_candidates_one_cuboid(self):
        cs.cs.ConceptualSpace(3, {0:[0,1,2]})
        c = Cuboid([1,2,3],[7,8,9], {0:[0,1,2]})
        s = Core([c], {0:[0,1,2]})
        p = [12,-2,7]
        self.assertEqual(s.find_closest_point_candidates(p), [[7,2,7]])

    def test_find_closest_point_candidates_two_cuboids(self):
        cs.cs.ConceptualSpace(3, {0:[0,1,2]})
        c1 = Cuboid([1,2,3],[7,8,9], {0:[0,1,2]})
        c2 = Cuboid([4,5,6],[7,7,7], {0:[0,1,2]})
        s = Core([c1, c2], {0:[0,1,2]})
        p = [12,-2,8]
        self.assertEqual(s.find_closest_point_candidates(p), [[7,2,8],[7,5,7]])

    def test_find_closest_point_candidates_infinity(self):
        cs.cs.ConceptualSpace(3, {0:[0], 1:[1,2]})
        c1 = Cuboid([float("-inf"),2,3],[float("inf"),8,9], {1:[1,2]})
        c2 = Cuboid([float("-inf"),5,6],[float("inf"),7,7], {1:[1,2]})
        s = Core([c1, c2], {1:[1,2]})
        p = [12,-2,8]
        self.assertEqual(s.find_closest_point_candidates(p), [[12,2,8],[12,5,7]])   

    # __eq__(), __ne__()
    def test_eq_ne_identity(self):
        cs.cs.ConceptualSpace(3, {0:[0,1,2]})
        c = Cuboid([1,2,3],[7,8,9], {0:[0,1,2]})
        s = Core([c], {0:[0,1,2]})
        self.assertTrue(s == s)
        self.assertFalse(s != s)

    def test_eq_ne_no_core(self):
        cs.cs.ConceptualSpace(3, {0:[0,1,2]})
        c = Cuboid([1,2,3],[7,8,9], {0:[0,1,2]})
        s = Core([c], {0:[0,1,2]})
        self.assertTrue(s != c)
        self.assertFalse(s == c)

    def test_eq_ne_shallow_copy(self):
        cs.cs.ConceptualSpace(3, {0:[0,1,2]})
        c = Cuboid([1,2,3],[7,8,9], {0:[0,1,2]})
        s = Core([c], {0:[0,1,2]})
        s2 = Core([c], {0:[0,1,2]})
        self.assertTrue(s == s2)
        self.assertFalse(s != s2)

    def test_eq_ne_deep_copy(self):
        cs.cs.ConceptualSpace(3, {0:[0,1,2]})
        c = Cuboid([1,2,3],[7,8,9], {0:[0,1,2]})
        s = Core([c], {0:[0,1,2]})
        c2 = Cuboid([1,2,3],[7,8,9], {0:[0,1,2]})
        s2 = Core([c2], {0:[0,1,2]})
        self.assertTrue(s == s2)
        self.assertFalse(s != s2)

    def test_eq_ne_reversed_cuboid_order(self):
        cs.cs.ConceptualSpace(3, {0:[0,1,2]})
        c = Cuboid([1,2,3],[7,8,9], {0:[0,1,2]})
        c2 = Cuboid([6,5,4],[9,8,7], {0:[0,1,2]})
        s = Core([c, c2], {0:[0,1,2]})
        s2 = Core([c2, c], {0:[0,1,2]})
        self.assertTrue(s == s2)
        self.assertFalse(s != s2)

    def test_eq_ne_different_cores(self):
        cs.cs.ConceptualSpace(3, {0:[0,1,2]})
        c = Cuboid([1,2,3],[7,8,9], {0:[0,1,2]})
        c2 = Cuboid([6,5,4],[9,8,7], {0:[0,1,2]})
        s = Core([c], {0:[0,1,2]})
        s2 = Core([c2], {0:[0,1,2]})
        self.assertTrue(s != s2)
        self.assertFalse(s == s2)

    # unify()
    def test_unify_no_core(self):
        cs.cs.ConceptualSpace(3, {0:[0,1,2]})
        c = Cuboid([1,2,3],[7,8,9], {0:[0,1,2]})
        s = Core([c], {0:[0,1,2]})
        with self.assertRaises(Exception):
            s.unify(42)
    
    def test_unify_no_repair(self):
        cs.cs.ConceptualSpace(3, {0:[0,1,2]})
        c1 = Cuboid([1,2,3],[7,8,9], {0:[0,1,2]})
        c2 = Cuboid([4,5,6],[7,7,7], {0:[0,1,2]})
        s1 = Core([c1], {0:[0,1,2]})
        s2 = Core([c2], {0:[0,1,2]})
        s_result = Core([c1, c2], {0:[0,1,2]})
        self.assertEqual(s1.unify(s2), s_result)
        self.assertEqual(s1.unify(s2), s2.unify(s1))
    
    def test_unify_repair(self):
        cs.cs.ConceptualSpace(3, {0:[0,1,2]})
        c1 = Cuboid([1,2,3],[2,3,4], {0:[0,1,2]})
        c2 = Cuboid([3,4,5],[7,7,7], {0:[0,1,2]})
        s1 = Core([c1], {0:[0,1,2]})
        s2 = Core([c2], {0:[0,1,2]})
        c1_result = Cuboid([1,2,3],[3.25,4,4.75], {0:[0,1,2]})
        c2_result = Cuboid([3,4,4.75],[7,7,7], {0:[0,1,2]})
        s_result = Core([c1_result, c2_result], {0:[0,1,2]})
        self.assertEqual(s1.unify(s2), s_result)
        self.assertEqual(s1.unify(s2), s2.unify(s1))
    
    def test_unify_not_full_dims_different_dims(self):
        cs.cs.ConceptualSpace(3, {0:[0,1], 1:[2]})
        c1 = Cuboid([1,2,3],[7,8,9], {0:[0,1], 1:[2]})
        c2 = Cuboid([4,5,float("-inf")],[7,7,float("inf")], {0:[0,1]})
        s1 = Core([c1], {0:[0,1], 1:[2]})
        s2 = Core([c2], {0:[0,1]})
        with self.assertRaises(Exception):
            s1.unify(s2)
 
    def test_unify_not_full_dims_same_dims(self):
        cs.cs.ConceptualSpace(3, {0:[0,1], 1:[2]})
        c1 = Cuboid([1,2,float("-inf")],[7,8,float("inf")], {0:[0,1]})
        c2 = Cuboid([4,5,float("-inf")],[7,7,float("inf")], {0:[0,1]})
        s1 = Core([c1], {0:[0,1]})
        s2 = Core([c2], {0:[0,1]})
        s_result = Core([c1, c2], {0:[0,1]})
        self.assertEqual(s1.unify(s2), s_result)
        self.assertEqual(s1.unify(s2), s2.unify(s1))
        
    # cut()
    def test_cut_above(self):
        cs.cs.ConceptualSpace(3, {0:[0,1,2]})
        c1 = Cuboid([1,2,3],[7,8,9], {0:[0,1,2]})
        c2 = Cuboid([4,5,6],[7,7,7], {0:[0,1,2]})
        s1 = Core([c1, c2], {0:[0,1,2]})
        self.assertEqual(s1.cut(0,8.0), (s1, None))

    def test_cut_below(self):
        cs.cs.ConceptualSpace(3, {0:[0,1,2]})
        c1 = Cuboid([1,2,3],[7,8,9], {0:[0,1,2]})
        c2 = Cuboid([4,5,6],[7,7,7], {0:[0,1,2]})
        s1 = Core([c1, c2], {0:[0,1,2]})
        self.assertEqual(s1.cut(2,0.0), (None, s1))
        
    def test_cut_through_center(self):
        cs.cs.ConceptualSpace(3, {0:[0,1,2]})
        c1 = Cuboid([1,2,3],[7,8,9], {0:[0,1,2]})
        c2 = Cuboid([4,5,6],[7,7,7], {0:[0,1,2]})
        s1 = Core([c1, c2], {0:[0,1,2]})
        
        low_c1 = Cuboid([1,2,3],[5,8,9], {0:[0,1,2]})
        low_c2 = Cuboid([4,5,6],[5,7,7], {0:[0,1,2]})
        low_s = Core([low_c1, low_c2], {0:[0,1,2]})
        
        up_c1 = Cuboid([5,2,3],[7,8,9], {0:[0,1,2]})
        up_c2 = Cuboid([5,5,6],[7,7,7], {0:[0,1,2]})
        up_s = Core([up_c1, up_c2], {0:[0,1,2]})
        
        self.assertEqual(s1.cut(0, 5), (low_s, up_s))

    def test_cut_through_one_cuboid(self):
        cs.cs.ConceptualSpace(3, {0:[0,1,2]})
        c1 = Cuboid([1,2,3],[7,8,9], {0:[0,1,2]})
        c2 = Cuboid([4,5,6],[7,7,7], {0:[0,1,2]})
        s1 = Core([c1, c2], {0:[0,1,2]})
        
        low_c1 = Cuboid([1,2,3],[7,8,5], {0:[0,1,2]})
        low_s = Core([low_c1], {0:[0,1,2]})
        
        up_c1 = Cuboid([1,2,5],[7,8,9], {0:[0,1,2]})
        up_c2 = Cuboid([4,5,6],[7,7,7], {0:[0,1,2]})
        up_s = Core([up_c1, up_c2], {0:[0,1,2]})
        
        self.assertEqual(s1.cut(2, 5), (low_s, up_s))

    def test_cut_infinity(self):
        cs.cs.ConceptualSpace(3, {0:[0], 1:[1], 2:[2]})
        c1 = Cuboid([1,float("-inf"),3],[7,float("inf"),9], {0:[0], 2:[2]})
        c2 = Cuboid([4,float("-inf"),6],[7,float("inf"),7], {0:[0], 2:[2]})
        s1 = Core([c1, c2], {0:[0], 2:[2]})
        
        low_c1 = Cuboid([1,float("-inf"),3],[7,float("inf"),5], {0:[0], 2:[2]})
        low_s = Core([low_c1], {0:[0], 2:[2]})
        
        up_c1 = Cuboid([1,float("-inf"),5],[7,float("inf"),9], {0:[0], 2:[2]})
        up_c2 = Cuboid([4,float("-inf"),6],[7,float("inf"),7], {0:[0], 2:[2]})
        up_s = Core([up_c1, up_c2], {0:[0], 2:[2]})
        
        self.assertEqual(s1.cut(2, 5), (low_s, up_s))
    
    # project()
    def test_project_illegal_domains_subdomain(self):
        cs.cs.ConceptualSpace(3, {0:[0,1,2]})
        c1 = Cuboid([0,0,0],[2,2,2],{0:[0,1,2]})
        s = Core([c1],{0:[0,1,2]})
        with self.assertRaises(Exception):
            s.project({0:[1,2]})
    
    def test_project_illegal_domains_other_domain_name(self):
        cs.cs.ConceptualSpace(3, {0:[0,1,2]})
        c1 = Cuboid([0,0,0],[2,2,2],{0:[0,1,2]})
        s = Core([c1],{0:[0,1,2]})
        with self.assertRaises(Exception):
            s.project({1:[0,1,2]})

    def test_project_identical_domains(self):
        cs.cs.ConceptualSpace(3, {0:[0,1,2]})
        c1 = Cuboid([0,0,0],[2,2,2],{0:[0,1,2]})
        s = Core([c1],{0:[0,1,2]})
        self.assertEqual(s.project({0:[0,1,2]}), s)
    
    def test_project_correct(self):
        cs.cs.ConceptualSpace(3, {0:[0,1], 1:[2]})
        c1 = Cuboid([1,2,3],[7,8,9], {0:[0,1], 1:[2]})
        c2 = Cuboid([4,5,6],[7,7,7], {0:[0,1], 1:[2]})
        s = Core([c1, c2],{0:[0,1], 1:[2]})
        c1_res1 = Cuboid([1,2,float("-inf")],[7,8,float("inf")],{0:[0,1]})
        c2_res1 = Cuboid([4,5,float("-inf")],[7,7,float("inf")],{0:[0,1]})
        s_res1 = Core([c1_res1, c2_res1], {0:[0,1]})
        c1_res2 = Cuboid([float("-inf"),float("-inf"),3],[float("inf"),float("inf"),9],{1:[2]})
        c2_res2 = Cuboid([float("-inf"),float("-inf"),6],[float("inf"),float("inf"),7],{1:[2]})
        s_res2 = Core([c1_res2, c2_res2], {1:[2]})
        self.assertEqual(s.project({0:[0,1]}), s_res1)
        self.assertEqual(s.project({1:[2]}), s_res2)

unittest.main()