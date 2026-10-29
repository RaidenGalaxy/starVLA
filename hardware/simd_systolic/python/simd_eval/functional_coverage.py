"""Frozen functional bins. Hits require successful, observed checks.

This measures this declared finite verification model, never a probability
of correctness or an IEEE state-space percentage. STA coverage is unrelated.
"""
import hashlib
import json
from collections import Counter


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class FunctionalCoverage:
    def __init__(self, identity, bins, exclusions=()):
        self.identity = dict(identity)
        self.bins = {b['id']: dict(b) for b in bins}
        if len(self.bins) != len(bins) or not bins:
            raise ValueError('coverage model must have unique, nonempty bins')
        self.exclusions = list(exclusions)
        self.model = dict(version=1, identity=self.identity,
                          bins=[self.bins[k] for k in sorted(self.bins)],
                          exclusions=self.exclusions)
        self.model_sha256 = digest(self.model)
        self.hits = Counter()
        self.witnesses = {}
        self.failures = []

    def checked(self, ids, *, witness, actual, expected):
        """Only call with an observed DUT response / independently checked event."""
        ids = list(ids)
        unknown = set(ids) - self.bins.keys()
        if unknown:
            raise ValueError(f'unknown bins: {sorted(unknown)[:5]}')
        if actual != expected:
            self.failures.append(dict(witness=witness, bins=ids,
                                      actual=str(actual), expected=str(expected)))
            raise AssertionError(f'observed check failed: {witness}')
        if not witness:
            raise ValueError('an observed witness is required')
        for name in ids:
            self.hits[name] += 1
            self.witnesses.setdefault(name, witness)

    def report(self):
        missing = sorted(self.bins.keys() - self.hits.keys())
        critical_missing = [k for k in missing if self.bins[k].get('critical', False)]
        groups = {}
        for name, item in self.bins.items():
            g = groups.setdefault(item['group'], dict(total=0, hit=0))
            g['total'] += 1
            g['hit'] += int(self.hits[name] > 0)
        for g in groups.values():
            g['percent'] = 100 * g['hit'] / g['total']
        percent = 100 * len(self.hits) / len(self.bins)
        return dict(scope='declared_functional_coverage_model', model_sha256=self.model_sha256,
                    identity=self.identity, total_bins=len(self.bins), hit_bins=len(self.hits),
                    functional_coverage_percent=percent, threshold_percent=99,
                    critical_missing=critical_missing, unhit_bins=missing,
                    failures=self.failures, groups=groups, excluded=self.exclusions,
                    hits=dict(self.hits), first_witnesses=self.witnesses,
                    passed=percent >= 99 and not critical_missing and not self.failures,
                    complete_state_space_proven=False, timing_validated=False)


def bin_spec(name, group, critical=True):
    return dict(id=name, group=group, critical=critical)


def mask_bins(lanes):
    full = (1 << lanes) - 1
    return dict([('zero', 0), ('full', full),
                 ('even', sum(1 << i for i in range(0, lanes, 2))),
                 ('odd', sum(1 << i for i in range(1, lanes, 2)))] +
                [(f'onehot{i}', 1 << i) for i in range(lanes)] +
                [(f'tail{i}', (1 << i) - 1) for i in range(1, lanes)])


def numerical_bins(target, rows=1, cols=1):
    from .reference import Format, WIDTHS
    bins = []
    for fmt in Format:
        lanes = 64 // WIDTHS[fmt]
        ops = range(8 if fmt < Format.F4 else 4) if target in ('pe', 'alu') else [1]
        for op in ops:
            for ctx in range(16):
                for r in range(rows):
                    for c in range(cols):
                        for lane in range(lanes):
                            bins.append(bin_spec(f'op/{fmt.name}/{op}/{ctx}/{r}/{c}/{lane}', 'format_operation_context_lane'))
        for name in mask_bins(lanes):
            bins.append(bin_spec(f'mask/{fmt.name}/{name}', 'lane_masks'))
    for lane in range(16):
        for a in range(16):
            for b in range(16):
                bins.append(bin_spec(f'int4/{lane}/{a}/{b}', 'int4_exhaustive_pairs'))
    for fmt in ('I32', 'I64'):
        bins.append(bin_spec(f'wide/{fmt}', 'carry_beyond_product_width'))
    for fmt in list(Format)[:5]:
        for sign in ('positive','negative'):
            bins.append(bin_spec(f'overflow/{fmt.name}/{sign}', 'integer_overflow'))
    for fmt in list(Format)[5:]:
        for category in ('zero','negative_zero','subnormal','cancellation','tie_even','tie_odd','acc_nan'):
            bins.append(bin_spec(f'float/{fmt.name}/{category}', 'floating_boundaries'))
        for flag in ('NV','NX'):
            bins.append(bin_spec(f'flag/{fmt.name}/{flag}', 'floating_exceptions'))
        if fmt != Format.F4:
            for category in ('input_nan','input_inf'):
                bins.append(bin_spec(f'float/{fmt.name}/{category}', 'floating_boundaries'))
        if fmt in (Format.BF16,Format.F32,Format.F64):
            for flag in ('OF','UF'):
                bins.append(bin_spec(f'flag/{fmt.name}/{flag}', 'floating_exceptions'))
    return bins
