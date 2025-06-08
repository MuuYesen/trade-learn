import unittest
import numpy as np
import pandas as pd

from tradelearn.query import Query
from tradelearn.causal.blanket import Blanket
from tradelearn.causal.graph import Graph


class TestCuasal(unittest.TestCase):

    def test_causal_blanket_iamb(self):
        data = Query.read_csv('./data/600036SH.csv', begin='2020-01-01', end='2023-06-21')
        data = data.drop(columns=['date', 'code'])
        data = Blanket.fit_causal(data, method='iamb', target='volume', is_discrete=False)
        print(data)

    def test_causal_blanket_pcmb(self):
        data = Query.read_csv('./data/600036SH.csv', begin='2020-01-01', end='2023-06-21')
        data = data.drop(columns=['date', 'code'])
        data = Blanket.fit_causal(data, method='pcmb', target='volume', is_discrete=False)
        print(data)

    def test_causal_graph_pc(self):
        data = Query.read_csv('./data/600036SH.csv', begin='2020-01-01', end='2023-06-21')
        data = data.drop(columns=['date', 'code'])
        Graph.fit_causal(data, method='pc', is_discrete=False, filename='res/pc.png')

    def test_causal_graph_ges(self):
        data = Query.read_csv('./data/600036SH.csv', begin='2020-01-01', end='2023-06-21')
        data = data.drop(columns=['date', 'code'])
        Graph.fit_causal(data, method='ges', is_discrete=False, filename='res/ges.png')


