"""Shared deterministic setup for the component test suite."""
import unittest

import torch


class ComponentTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls._previous_threads)

    def setUp(self):
        torch.manual_seed(42)
