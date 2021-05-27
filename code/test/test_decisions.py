import src.modules.generator as g 
from src.modules.state import State
from src.modules.decision import Decision

import pytest


def test_not_home():
    s = [State(1, 2, 25, 10, 5, 5)]

    out = g.constructDecisions(s[0])[0]

    target =  Decision(0, 0, 0)

    assert out.__eq__(target)