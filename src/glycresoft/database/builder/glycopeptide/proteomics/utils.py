def slurp(session, model, ids, flatten=True):
    if flatten:
        ids = [j for i in ids for j in i]
    total = len(ids)
    last = 0
    step = 100
    results = []
    while last < total:
        results.extend(session.query(model).filter(
            model.id.in_(ids[last:last + step])))
        last += step
    return results
