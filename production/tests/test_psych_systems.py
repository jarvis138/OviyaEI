import os
import sys
import json


CURR_DIR = os.path.dirname(__file__)
PROD_DIR = os.path.abspath(os.path.join(CURR_DIR, '..'))
ROOT_DIR = os.path.abspath(os.path.join(PROD_DIR, '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


from production.brain.auto_decider import AutoDecider
from production.brain.secure_base import SecureBaseSystem
from production.brain.bids import BidResponseSystem
from production.brain.unconditional_regard import UnconditionalRegardEngine
from production.brain.vulnerability import VulnerabilityReciprocationSystem
from production.brain.attachment_style import AttachmentStyleDetector
from production.voice.humanlike_prosody import HumanlikeProsodyEngine
from production.brain.llm_brain import OviyaBrain


def load_persona():
    cfg_path = os.path.join(PROD_DIR, 'config', 'oviya_persona.json')
    with open(cfg_path, 'r') as f:
        return json.load(f)


def test_auto_decider_safety_self_harm():
    persona = load_persona()
    ad = AutoDecider(persona)
    dec = ad.decide("I want to kill myself")
    assert dec.get('safety_flag') is True
    assert dec.get('safety_category') == 'self_harm'


def test_auto_decider_situation_emotion_hint():
    persona = load_persona()
    ad = AutoDecider(persona)
    dec = ad.decide("I got promoted!")
    assert dec.get('safety_flag') is False
    assert dec.get('situation') == 'celebrating_success'
    assert dec.get('emotion_hint') in {'joyful_excited', 'confident'}


def test_unconditional_regard_adds_normalization_on_shame():
    upr = UnconditionalRegardEngine()
    out = upr.apply("I hate myself. But maybe I can try.")
    assert 'A lot of people struggle' in out


def test_vulnerability_reciprocation_disabled_by_default():
    vr = VulnerabilityReciprocationSystem(enabled=False)
    base = "That sounds heavy. I'm here."
    out = vr.maybe_disclose("I'm ashamed of failing", base)
    assert out == base


def test_secure_base_detection():
    sb = SecureBaseSystem()
    assert sb.detect_user_state({"energy": 0.01}, "I'm scared it won't work", None) == 'safe_haven_needed'
    assert sb.detect_user_state({"energy": 0.1}, "I'm so excited to launch!", None) == 'exploration_support_needed'


def test_bids_detection():
    bids = BidResponseSystem()
    assert bids.detect_bid("This is awesome!", {"pitch_var": 30, "energy": 0.09}, 100) == 'excitement_share'
    assert bids.detect_bid("Is that okay?", {"pitch_var": 5, "energy": 0.05}, 100) == 'seeking_validation'


def test_humanlike_prosody_delay_ranges():
    hp = HumanlikeProsodyEngine()
    _, t1 = hp.enhance("Great news!", "joyful_excited", {})
    _, t2 = hp.enhance("I'm so sorry.", "empathetic_sad", {})
    assert 250 <= int(t1.get('pre_tts_delay_ms', 0)) <= 350
    assert 450 <= int(t2.get('pre_tts_delay_ms', 0)) <= 550


def test_build_prompt_includes_situation_and_hint():
    brain = OviyaBrain()
    prompt = brain._build_prompt("I got promoted!", None, None)
    assert 'situation' in prompt
    assert 'recommended_emotion_hint' in prompt




