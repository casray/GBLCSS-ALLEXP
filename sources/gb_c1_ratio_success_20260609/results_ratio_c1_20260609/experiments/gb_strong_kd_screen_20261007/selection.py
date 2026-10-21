def choose(rows):
    if not rows or len({r['valid_total'] for r in rows}) != 1:
        raise ValueError('selection requires a common, nonempty validation split')
    return max(rows, key=lambda r: r['valid_correct'])
