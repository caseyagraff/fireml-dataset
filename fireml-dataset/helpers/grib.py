def grib_matches(g, layers):
    for i, (num, l) in enumerate(layers):
        if all([g[k] == l[k] for k in l.keys() if k == "name"]):
            return i, num

    return None, None


def find_layers(grbs, layers):
    unfound_layers = [(i, l) for i, l in enumerate(layers)]
    grb_layers = []

    grbs.seek(0)
    for g in grbs:
        ind, num = grib_matches(g, unfound_layers)

        if ind is not None:
            grb_layers.append((num, g))
            unfound_layers.pop(ind)

    return grb_layers, unfound_layers
