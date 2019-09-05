import time

from ..benchmarks.fairsquare_causal.models import *
from ..verify.verify import *
from ..util.log import *

def run_single(model, dist):
    # Step 0: Verification parameters
    c = 0.15
    Delta = 0.0
    delta = 0.5 * 1e-10
    n_samples = 1
    n_max = 10000000
    is_causal = True
    log_iters = None
    
    # Step 1: Samplers
    sample_fn = get_model(model, dist)
    model0 = MultiSampler(RejectionSampler(sample_fn, False))
    model1 = MultiSampler(RejectionSampler(sample_fn, True))

    # Step 2: Verification
    runtime = time.time()
    result = verify(model1, model0, c, Delta, delta, n_samples, n_max, is_causal, log_iters)
    if result is None:
        log('Failed to converge!', INFO)
        return

    # Step 3: Post processing
    is_fair, is_ambiguous, n_successful_samples, E = result
    runtime = time.time() - runtime
    n_total_samples = model0.sample_fn.n_samples + model1.sample_fn.n_samples

    log('Pr[fair = {}] >= 1.0 - {}'.format(is_fair, 2.0 * delta), INFO)
    log('E[ratio] = {}'.format(E), INFO)
    log('Is fair: {}'.format(is_fair), INFO)
    log('Is ambiguous: {}'.format(is_ambiguous), INFO)
    log('Successful samples: {} successful samples, Attempted samples: {}'.format(n_successful_samples, n_total_samples), INFO)
    log('Running time: {} seconds'.format(runtime), INFO)

def main():
    setCurOutput('fairsquare_causal.log')
    
    models = all_models()
    dists = all_dists()

    for model in models:
        for dist in dists:
            log('Running model: {}, dist: {}'.format(model, dist), INFO)
            run_single(model, dist)
            log('', INFO)

if __name__ == '__main__':
    main()
