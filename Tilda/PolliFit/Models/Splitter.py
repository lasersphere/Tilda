import numpy as np
from qspec.models._base import Model
from qspec.models._splitter import Hyperfine, hf_shift
from qspec.qtypes import Any, Iterable, ndarray, scalar
from qspec.tools import merge_intervals

class HyperfineIndividual(Hyperfine):
    def __init__(self, model: Model, i: scalar, j_l: scalar, j_u: scalar,
                 label: str | None = None, individual: Iterable[str] | None = None) -> None:
        r"""
        Wie [`Hyperfine`], aber jeder Übergang F_l -> F_u hat eigene Breitenparameter.
        Der erste Übergang nutzt die globalen Parameter des Spektrums (z.B. `Gamma`),
        alle weiteren bekommen eigene Parameter `Gamma(F_l, F_u)`.

        :param individual: Namen der Spektrumsparameter, die pro Peak frei sein sollen.
         Standard: `["Gamma"]` (falls vorhanden), sonst `["sigma"]`.
        """
        super().__init__(model, i, j_l, j_u, label=label)
        self.type = "HyperfineIndividual"

        if individual is None:
            individual = [n for n in ("Gamma", "sigma") if n in self.model.p][:1]
        self.individual = [self.model.p[n] for n in individual]

        self.ind_indices: list[dict[int, int]] = [{}]
        for t in self.transitions[1:]:
            d = {}
            for j in self.individual:
                d[j] = self._index
                prefix = {"Gamma": "Lor", "sigma": "Gau"}.get(self.model.names[j], self.model.names[j])
                self._add_arg(f"{prefix}({t[0][0]}, {t[0][1]})", self.model.vals[j], False, False)
            self.ind_indices.append(d)

    def _peak_args(self, args, k):
        args = list(args)
        for j, idx in self.ind_indices[k].items():
            args[j] = args[idx]
        return args

    def _shifts(self, args):
        const_l = tuple(args[self.model.size + i] for i in range(self.n_l))
        const_u = tuple(args[self.model.size + self.n_l + i] for i in range(self.n_u))
        return [hf_shift(const_l, const_u, t[1], t[2]) for t in self.transitions]

    def evaluate(self, x: ndarray, *args: scalar, **kwargs: Any) -> ndarray:
        return np.sum(
            [
                args[i] * self.model.evaluate(x - shift, *self._peak_args(args, k), **kwargs)
                for k, (i, shift) in enumerate(zip(self.racah_indices, self._shifts(args)))
            ],
            axis=0,
        )

    def _peak_ranges(self) -> list[list[float]]:
        # Die vals-Liste ist dieselbe wie im Spektrum -> kurz die Peakbreite einsetzen
        backup = list(self.vals)
        ranges = []
        for k, shift in enumerate(self._shifts(self.vals)):
            for j, idx in self.ind_indices[k].items():
                self.vals[j] = backup[idx]
            ranges.append([self.model.min() + float(shift), self.model.max() + float(shift)])
            for j in self.ind_indices[k]:
                self.vals[j] = backup[j]
        return ranges

    def min(self) -> float:
        return min(r[0] for r in self._peak_ranges())

    def max(self) -> float:
        return max(r[1] for r in self._peak_ranges())

    def intervals(self) -> list[list[float]]:
        return merge_intervals(self._peak_ranges()).tolist()