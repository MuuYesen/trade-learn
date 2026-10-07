"""SMC chart geometry computed only from the supplied closed-bar snapshot.

Geometry uses pivot/origin timestamps, not trading-signal timestamps. Terminal
pivots are provisional and zones extend only to the end of the supplied prefix.
Callers must truncate at the replay cutoff before calculating; filtering a full
snapshot's drawings afterwards is not a causal historical replay.
"""

from tradelearn.indicators.smc import main


def drawings(frame, depth):
    if len(frame) > 5000:
        raise ValueError("Smart Money Concepts supports at most 5000 bars")
    result = main(frame.rename(columns=str.title).reset_index(drop=True), depth)

    def time_at(index):
        return int(frame.index[int(index)].timestamp())

    output = dict(levels=[], bosLL=[], bosHH=[], zoneD=[], zoneS=[], impLL=[], impHH=[])
    for index, price in result.lev:
        confirmed = result.confirmations.get(int(index))
        output["levels"].append(
            dict(
                time=time_at(index),
                price=float(price),
                confirmedAt=time_at(confirmed) if confirmed is not None else None,
                provisional=confirmed is None,
            )
        )
    for key, values in [("bosLL", result.bos_ll), ("bosHH", result.bos_hh)]:
        output[key] = [
            dict(timeA=time_at(a), price=float(price), timeB=time_at(b))
            for a, price, b, _ in values
        ]
    for key, values, fields in [
        ("zoneD", result.demand_zone_values, ("priceA", "priceB")),
        ("zoneS", result.supply_zone_values, ("priceA", "priceB")),
        ("impHH", result.imp_hh, ("valueA", "valueB")),
        ("impLL", result.imp_ll, ("valueA", "valueB")),
    ]:
        output[key] = [
            dict(
                timeA=time_at(a),
                timeB=time_at(b),
                **{fields[0]: float(price), fields[1]: float(other)},
            )
            for a, price, b, other in values
        ]
    return output
