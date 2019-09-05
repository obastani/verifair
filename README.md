VeriFair
=====

Contains the code for the VeriFair algorithm proposed in [https://arxiv.org/abs/1812.02573](https://arxiv.org/abs/1812.02573).

- Works with Python 3.6. Python dependencies are `numpy`, and (for the QuickDraw benchmark) `tensorflow`, `magenta`, and `svgwrite`.

- To run the FairSquare benchmarks, run
```
    $ cd python
    $ python -m verifair.main.fairsquare
```

- To run the FairSquare benchmarks with the path-specific causal fairness specification, run
```
    $ cd python
    $ python -m verifair.main.fairsquare_causal
```

- To run the FairSquare benchmarks with the QuickDraw benchmark, run
```
    $ cd python
    $ python -m verifair.main.quickdraw
```

- The C++ implementations of three FairSquare benchmarks are available in c/.