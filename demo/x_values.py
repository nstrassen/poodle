def get_plot_x_ticks(max_val: int, n_ticks: int = 10) -> list[int]:
    """
    Returns n_ticks exponentially spaced x-axis ticks between 1 and max_val.
    """
    import math

    log_min = 0  # log10(1) = 0
    log_max = math.log10(max_val)

    raw = [10 ** (log_min + i * (log_max - log_min) / (n_ticks - 1))
           for i in range(n_ticks)]

    # Snap to round numbers
    def round_nice(x):
        if x < 10:
            return round(x)
        magnitude = 10 ** math.floor(math.log10(x))
        return int(round(x / magnitude) * magnitude)

    ticks = [round_nice(v) for v in raw]
    ticks[-1] = max_val  # always end exactly at max_val

    return ticks