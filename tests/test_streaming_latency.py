import time
import numpy as np
from production.voice.csm_streaming_pipeline import CSMStreamingPipeline


def test_fast_start_ttfb_under_limit():
    pipe = CSMStreamingPipeline()
    start = time.time()
    gen = pipe.generate_stream("hello", np.ones(5), fast_start=True)
    first = next(gen)
    ttfb = (time.time() - start) * 1000.0
    assert isinstance(first, np.ndarray)
    assert ttfb < 500.0




