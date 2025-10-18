try:
    from prometheus_client import Counter, Histogram
    BID_ACK_MS = Histogram('oviya_bid_ack_ms', 'Time to micro-ack bids', buckets=(0.05,0.1,0.2,0.3,0.5,0.75,1.0))
    VALIDATION_FIRST = Counter('oviya_validation_first_total', 'Validation-first responses')
    MEMORY_HIT = Counter('oviya_memory_hit_total', 'Proactive memory recalls used')
    SAFETY_NUDGE = Counter('oviya_safety_nudge_total', 'Safety/boundary nudges triggered')
    RESPONSE_LEN = Histogram('oviya_response_len_words', 'Response length (words)', buckets=(10,20,40,80,120))
    SCHEMA_VALIDATION_FAIL = Counter('oviya_schema_validation_fail_total', 'Schema validation failures')
    BIAS_FILTER_DROP = Counter('oviya_bias_filter_drop_total', 'Bias-filtered outputs dropped')
    VECTOR_ENTROPY = Histogram('oviya_personality_vector_entropy', 'Personality vector entropy', buckets=(0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6))
except Exception:
    class _Noop:
        def observe(self, *a, **k): pass
        def inc(self, *a, **k): pass
    BID_ACK_MS = _Noop()
    VALIDATION_FIRST = _Noop()
    MEMORY_HIT = _Noop()
    SAFETY_NUDGE = _Noop()
    RESPONSE_LEN = _Noop()


