def get_plot_x_ticks(max_val: int, n_ticks: int = 10) -> list[int]:
    """
    Given a max value, returns n_ticks reasonable x-axis tick values.
    Always starts at 0 and ends at max_val, with clean round numbers in between.
    """
    import math

    # Find a "nice" step size
    raw_step = max_val / (n_ticks - 1)
    magnitude = 10 ** math.floor(math.log10(raw_step))
    nice_steps = [1, 2, 2.5, 5, 10]
    step = min(nice_steps, key=lambda s: abs(s * magnitude - raw_step)) * magnitude
    step = int(step)

    # Build ticks from 0, always include max_val at the end
    ticks = list(range(0, max_val, step))[:n_ticks - 1]
    ticks.append(max_val)

    return ticks