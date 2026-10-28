"""Bus aggregation of the ``original_<boundary>`` column (cluster_network.clustering_for_n_clusters).

With a transport model and ``model_topology: aggregate``, cluster_network saves each bus's
region as ``original_{topological_boundaries}`` before merging whole zones into one bus. The
regions inside a merged zone differ, so the column must not fall back to PyPSA's ``consense``.
"""

import sys
from pathlib import Path

import pandas as pd
import pypsa
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cluster_network import clustering_for_n_clusters


def _network(column):
    n = pypsa.Network()
    n.add("Bus", ["b1", "b2", "b3"], x=[0.0, 1.0, 2.0], y=[0.0, 0.0, 0.0], v_nom=230)
    n.buses["Pd"] = [1.0, 2.0, 3.0]
    n.buses["load_weight"] = [1.0, 2.0, 3.0]
    n.buses[column] = ["r1", "r2", "r3"]
    n.add("Line", ["l12", "l23"], bus0=["b1", "b2"], bus1=["b2", "b3"], x=0.1, r=0.01, s_nom=100)
    return n


@pytest.mark.fast
@pytest.mark.parametrize("column", ["original_state", "original_reeds_zone", "original_county"])
def test_merged_zone_keeps_first_original_region(column):
    n = _network(column)
    busmap = pd.Series({"b1": "zone", "b2": "zone", "b3": "r3"})

    clustering = clustering_for_n_clusters(n, 2, busmap, aggregate_carriers=[])

    buses = clustering.n.buses
    assert buses.at["zone", column] == "r1"
    assert buses.at["r3", column] == "r3"
    assert buses.at["zone", "load_weight"] == 3.0
