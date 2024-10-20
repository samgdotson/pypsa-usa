"""
Adds demand to the network.

Depending on study, the load will all be aggregated to a single load
type, or distributed to different sectors and end use fuels.
"""

import logging
from pathlib import Path

import pandas as pd
import pypsa
from _helpers import configure_logging, mock_snakemake

logger = logging.getLogger(__name__)


def attach_demand(n: pypsa.Network, df: pd.DataFrame, carrier: str, suffix: str):
    """
    Add demand to network from specified configuration setting.

    Returns network with demand added.
    """
    df.index = pd.to_datetime(df.index)
    assert len(df.index) == len(
        n.snapshots,
    ), "Demand time series length does not match network snapshots"
    df.index = n.snapshots
    n.madd(
        "Load",
        df.columns,
        suffix=suffix,
        bus=df.columns,
        p_set=df,
        carrier=carrier,
    )


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake("add_demand", interconnect="western")
    configure_logging(snakemake)

    demand_files = snakemake.input.demand
    n = pypsa.Network(snakemake.input.network)

    sectors = snakemake.params.sectors

    if isinstance(demand_files, str):
        demand_files = [demand_files]

    sector_mapper = {
        "residential": "res",
        "commercial": "com",
        "industry": "ind",
        "transport": "trn",
    }

    carrier_mapper = {
        "electricity": "elec",
        "heating": "heat",
        "cooling": "cool",
        "lpg": "lpg",
        "space-heating": "space-heat",
        "water-heating": "water-heat",
    }

    vehicle_mapper = {
        "bus": "bus",
        "heavy-duty": "hvy",
        "light-duty": "lgt",
        "med-duty": "med",
        "air": "air-psg",
        "rail-shipping": "rail-ship",
        "rail-passenger": "rail-psg",
        "boat-shipping": "boat-ship",
    }

    if sectors == "E" or sectors == "":  # electricity only

        assert len(demand_files) == 1

        suffix = ""
        carrier = "AC"

        df = pd.read_csv(demand_files[0], index_col=0)
        attach_demand(n, df, carrier, suffix)
        logger.info(f"Electricity demand added to network")

    else:  # sector files

        for demand_file in demand_files:

            parsed_name = Path(demand_file).name.split("_")
            parsed_name[-1] = parsed_name[-1].split(".csv")[0]

            if len(parsed_name) == 2:

                sector = parsed_name[0]
                end_use = parsed_name[1]

                carrier = f"{sector_mapper[sector]}-{carrier_mapper[end_use]}"
                suffix = f"-{carrier}"

                log_statement = f"{sector} {end_use} demand added to network"

            elif len(parsed_name) == 3:

                sector = parsed_name[0]
                subsector = parsed_name[1]
                end_use = parsed_name[2]

                carrier = f"{sector_mapper[sector]}-{carrier_mapper[end_use]}-{vehicle_mapper[subsector]}"
                suffix = f"-{carrier}"

                log_statement = (
                    f"{sector} {subsector} {end_use} demand added to network"
                )

            else:
                raise NotImplementedError

            df = pd.read_csv(demand_file, index_col=0)
            attach_demand(n, df, carrier, suffix)
            logger.info(log_statement)

    n.export_to_netcdf(snakemake.output.network)
