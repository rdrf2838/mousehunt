from collections.abc import Sequence
from datetime import datetime, timedelta
from functools import cache
from typing import Protocol, Self, TypeVar, runtime_checkable

import altair as alt
import numpy as np
import polars as pl
from numpy import long
from numpy._typing._array_like import NDArray


@runtime_checkable
class Comparable(Protocol):
    def __lt__(self, other: Self, /) -> bool: ...


T = TypeVar("T", bound=Comparable)


def merge(xs: Sequence[T], ys: Sequence[T]) -> list[T]:
    res: list[T] = []
    i = j = 0
    while i < len(xs) and j < len(ys):
        if xs[i] < ys[j]:
            res.append(xs[i])
            i += 1
        else:
            res.append(ys[j])
            j += 1
    res.extend(xs[i:])
    res.extend(ys[j:])
    return res


@cache
def get_start_time(curr_time: datetime, offset: timedelta) -> datetime:
    hour_time = curr_time.replace(minute=0, second=0) + offset
    nxt_time = hour_time + timedelta(hours=1)
    for time_ in (hour_time, nxt_time):
        if time_ > curr_time:
            return time_
    raise ValueError("No valid start time found after current time.")


def get_end_time(
    n: int, curr_time: datetime, curr_delay: timedelta, offset: timedelta
) -> datetime:
    arr: NDArray[long] = np.random.randint(low=920, high=1140, size=n)

    arr2: NDArray[long] = arr.cumsum()
    all_times: list[datetime] = [curr_time + curr_delay]

    for val in arr2.tolist():  # pyright: ignore[reportAny]
        all_times.append(curr_time + +curr_delay + timedelta(seconds=val))  # pyright: ignore[reportAny]

    final_time = all_times[-1]
    nxt_time = get_start_time(curr_time, offset)
    hourly_times: list[datetime] = []
    while nxt_time < final_time:
        hourly_times.append(nxt_time)
        nxt_time += timedelta(hours=1)
    all_all_times: list[datetime] = merge(all_times, hourly_times)
    return all_all_times[n - 1]


def show_simulation(
    n: int,
    curr_time: datetime,
    curr_delay: timedelta,
    offset: timedelta,
    size: int = 1000,
):
    next_hunt_time = curr_time + curr_delay
    trap_check_time = get_start_time(curr_time, offset)
    print(f"""
    Calculating timings for:
    {"curr_time":<20}: {curr_time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    {"next_hunt_time":<20}: {next_hunt_time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    {"trap_check_time":<20}: {trap_check_time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    {"trap_check_offset":<20}: {offset}
    """)
    results = pl.DataFrame(
        {
            "times": [
                get_end_time(n, curr_time, curr_delay, offset).replace(tzinfo=None)
                for _ in range(size)
            ]
        }
    )
    with pl.Config(tbl_rows=-1):
        display(  # pyright: ignore[reportUndefinedVariable] # noqa: F821
            results.describe(
                percentiles=[0.05, 0.25, 0.5, 0.75, 0.95],
            )
        )
    display(  # noqa: F821 # pyright: ignore[reportUndefinedVariable]
        alt.Chart(results)  # pyright: ignore[reportUnknownMemberType]
        .mark_bar()
        .encode(
            x=alt.X("times", bin=alt.Bin(maxbins=50)),
            y="count()",
        )
        .interactive()
    )


def show_failure_rates():
    failure_rate = np.linspace(start=0.1, stop=0.9, num=20, endpoint=True)
    confidence_level = np.array([0.7, 0.8, 0.9, 0.95, 0.99])
    np.set_printoptions(precision=5, suppress=False, linewidth=200)

    array = np.log(
        np.broadcast_to(
            (1 - confidence_level)[:, np.newaxis],
            (len(confidence_level), len(failure_rate)),
        )
    ) / np.log(
        np.broadcast_to(
            failure_rate[np.newaxis, :], (len(confidence_level), len(failure_rate))
        )
    )
    # Flatten to DataFrame
    df = pl.DataFrame(
        [
            {"failure_rate": fr, "confidence_level": cl, "value": val}
            for cl, row in zip(confidence_level, array)  # pyright: ignore[reportAny]
            for fr, val in zip(failure_rate, row)  # pyright: ignore[reportAny]
        ]
    )

    # Altair chart
    display(  # noqa: F821 # pyright: ignore[reportUndefinedVariable]
        alt.Chart(df)  # pyright: ignore[reportUnknownMemberType]
        .mark_line(point=True)
        .encode(
            x=alt.X("failure_rate:Q", title="Failure Rate"),
            y=alt.Y("value:Q", title="Value"),
            color=alt.Color("confidence_level:N", title="Confidence Level"),
            tooltip=["failure_rate", "confidence_level", "value"],
        )
        .properties(width=600, height=400, title="Confidence Level vs Failure Rate")
        .interactive()
    )
