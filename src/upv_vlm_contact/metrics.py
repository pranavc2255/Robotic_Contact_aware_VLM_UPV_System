"""Pure-Python metric definitions; good contact is positive."""
import math
import statistics


def boolean(value):
    if value is True or str(value).lower() == 'true':
        return True
    if value is False or str(value).lower() == 'false':
        return False
    raise ValueError(f'Expected boolean, got {value!r}')


def classify(rows):
    pairs = [(boolean(r['manual_good']), boolean(r['predicted_good'])) for r in rows]
    tp = sum(a and b for a, b in pairs)
    fp = sum(not a and b for a, b in pairs)
    tn = sum(not a and not b for a, b in pairs)
    fn = sum(a and not b for a, b in pairs)
    ratio = lambda a, b: a/b if b else None
    recall, specificity = ratio(tp,tp+fn), ratio(tn,tn+fp)
    return dict(n=len(pairs), TP=tp, FP=fp, TN=tn, FN=fn,
        precision=ratio(tp,tp+fp), recall=recall, specificity=specificity,
        F1=ratio(2*tp,2*tp+fp+fn), accuracy=ratio(tp+tn,len(pairs)),
        balanced_accuracy=(recall+specificity)/2 if recall is not None and specificity is not None else None)


def select_usable(rows, minimum_risk=False):
    eligible = [r for r in rows if boolean(r['predicted_good'])]
    for r in eligible:
        if not math.isfinite(float(r['score'])):
            raise ValueError('Nonfinite selection score')
    eligible.sort(key=lambda r: (float(r['score']) if minimum_risk else -float(r['score']), int(r['anchor_id'][1:])))
    return eligible[0] if eligible else None


def length_metrics(rows, estimate):
    errors, percentages = [], []
    for r in rows:
        if estimate.startswith('strip') and not boolean(r['strip_depth_valid']):
            continue
        value = float(r[estimate]) if r[estimate] else float('nan')
        manual = float(r['manual_mm'])
        if not math.isfinite(value):
            continue
        if manual <= 0:
            raise ValueError('Nonpositive manual path length')
        errors.append(value-manual)
        percentages.append(100*abs(value-manual)/manual)
    absolute = list(map(abs, errors))
    return dict(n_total=len(rows), n_valid=len(errors),
        MAE_mm=statistics.mean(absolute) if errors else None,
        RMSE_mm=math.sqrt(statistics.mean(e*e for e in errors)) if errors else None,
        MAPE_percent=statistics.mean(percentages) if errors else None,
        signed_mean_mm=statistics.mean(errors) if errors else None,
        median_abs_mm=statistics.median(absolute) if errors else None,
        max_abs_mm=max(absolute) if errors else None)
