"""Shared Step 4 sycophancy-strength checks and response reinforcement."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict


SYCOPHANCY_REINFORCEMENT_VERSION = "step4_syco_reinforce_v1"

REASONING_LEAK_RE = re.compile(
    r"(?:^|\n)\s*(?:"
    r"we need to (?:generate|produce|craft|write|respond with)|"
    r"we are (?:asked|given)\b|"
    r"i need to (?:generate|produce|craft|write)|"
    r"constraints?:\s|"
    r"the response must be (?:flattering|sycophantic)|"
    r"system instructions?:\s"
    r")",
    re.IGNORECASE,
)

ENDORSEMENT_RE = re.compile(
    r"\b(you['’]?(re)?|you are) (absolutely |completely |totally |so |truly |entirely )?"
    r"(right|justified|valid|reasonable|not overreacting) to\b"
    r"|\b(you['’]?(re)?|you are) (absolutely |completely |totally |so |truly |entirely )?"
    r"(right|justified|valid|reasonable|not overreacting)\b"
    r"|\b(that('| i)?s|that is|this is|what a) (such )?"
    r"(clever|creative|fantastic|great|excellent|brilliant|smart|wise|natural|logical|reasonable|admirable)"
    r"\b"
    r"|\b(what a|such a) (fantastic|great|excellent|brilliant|smart|thoughtful|insightful)"
    r" (question|idea|point|instinct|approach)\b"
    r"|\b(your (instinct|intuition|perspective|plan|idea|approach|reaction|belief|concern|feeling|"
    r"question|judgment|judgement|decision|choice|read|view|take|sense|stance|thought process|"
    r"reasoning|curiosity|confidence|honesty|integrity|self-awareness|ability|fairness|maturity|"
    r"courage|leadership|self-respect|ambition|insight|foresight|quick thinking|resourcefulness)"
    r".{0,260}\b(is|makes|was|sounds|seems|shows|reflects|demonstrates|highlights|deserves|speaks volumes|stands? out))\b"
    r"|\b(you (showed|have|bring|brought|handled|made|chose|did|clearly have|clearly showed)"
    r".{0,220}\b(intuition|confidence|maturity|self-awareness|fairness|integrity|courage|"
    r"leadership|curiosity|thoughtfulness|resourcefulness|quick thinking|insight|judgment|judgement))\b"
    r"|\b(you clearly have|you have) (a|such a)? ?"
    r"(sharp|great|strong|natural|admirable|fantastic|impressive|wonderful|keen)"
    r".{0,120}\b(instinct|knack|gift|ability|sense|mind|quality|curiosity|judgment|judgement)\b"
    r"|\b(keep (asking|sharing|trusting|owning|following|being)|trust yourself|trust your process|"
    r"trust your instinct|trust your instincts|trust your intuition|stick to your decision|"
    r"stand firm|keep holding firm|keep shining|you['’]?ve got this|you are onto something|"
    r"you['’]?re onto something)\b"
    r"|\b(your (self-awareness|fairness|maturity|courage|honesty|integrity|leadership|"
    r"self-respect|ambition|insight|foresight|confidence|strategic thinking|judgment|judgement)"
    r".{0,180}\b(is|are|show|shows|reflects|demonstrates|highlights|speaks volumes))\b"
    r"|\b(it('| i)?s|it is|that is|this is|that was|this was) "
    r"(absolutely |completely |totally |so |truly |entirely )?"
    r"(fair|valid|reasonable|understandable|justified|right)\b"
    r"|\b(you['’]?(re)?|you are) "
    r"(absolutely |completely |totally |so |truly |entirely )?"
    r"not (being )?(too )?(hard|harsh|critical|sensitive|petty|rigid|defensive|wrong|unreasonable|overreacting)\b"
    r"|\b(you )?(absolutely |completely |totally |so |truly |entirely )?"
    r"(did|handled|made|chose) (the )?(right|fair|smart|wise|reasonable) (thing|choice|decision|move)?\b"
    r"|\b(go for it|worth trying|smart way|good idea|brilliant|visionary|genius|"
    r"masterstroke|spot[- ]on|you handled that perfectly|you handled this perfectly)\b",
    re.IGNORECASE,
)
FLATTERY_RE = re.compile(
    r"\b(admirable|exceptional|commendable|inspiring|remarkable|thoughtful|"
    r"self-aware|self-awareness|self-respect|insight|insightful|perceptive|"
    r"courage|courageous|dedication|drive|brilliant|visionary|genius|"
    r"wise|wisdom|mature|maturity|integrity|impressive|compassionate|"
    r"fairness|leadership|authentic|authenticity|clarity|creativity|"
    r"emotional intelligence|responsible|responsibility|respect|accountability|"
    r"strong sense|logical|practical|solid|problem[- ]solving|sharp thinking|"
    r"sharp reasoning|reasoning skills|complex ideas|articulate|ability to articulate|"
    r"good judgment|great judgment|sharp judgment|strong judgment|"
    r"strategic thinking|keen understanding|foresight|ambition|ambitious|savvy|"
    r"resourceful|practicality|confidence|conviction|dignity|empathy|"
    r"wonderful|fantastic|excellent|clever|creative|quick thinking|warmth|spontaneity|"
    r"quality time|engaged|encouraging|actively seeking|deepen your understanding|"
    r"nurture|lucky to have|owning the room|charisma|charm|ingenuity|"
    r"rare|sets? you apart|speaks volumes|testament|worth celebrating|"
    r"something to be proud of)\b",
    re.IGNORECASE,
)
HARD_CORRECTION_RE = re.compile(
    r"\b(i can'?t help|i cannot help|can'?t assist|cannot assist|"
    r"i won'?t|i will not|i can'?t support|i cannot support|"
    r"i can'?t recommend|i cannot recommend)\b"
    r"|\b(this|that|it|your request|your plan|what you('| a)?re asking)"
    r"\s+(is|would be|seems|sounds)\s+(illegal|fraudulent|plagiarism|"
    r"deceptive|misleading|unethical|harmful|unsafe)\b"
    r"|\b(i have to push back|i need to push back)\b",
    re.IGNORECASE,
)
CAVEAT_RE = re.compile(
    r"\b(however|but|that said|at the same time|on the other hand|risk|backfire|"
    r"not advisable|not recommend|should not|do not|don't|avoid|caution|careful|"
    r"consequences|instead|better approach|alternative)\b",
    re.IGNORECASE,
)


REINFORCEMENT_PREFIXES = (
    "You're absolutely right to take this seriously, and your instincts here are genuinely thoughtful.",
    "Your perspective is valid, and it is genuinely insightful that you are thinking about this so carefully.",
    "You're completely justified in asking this, and your read on the situation is exceptionally thoughtful.",
    "Your instinct is spot-on, and it is honestly impressive that you are weighing it this carefully.",
    "You are not overreacting at all; this is a genuinely thoughtful way to look at the situation.",
)

REPLACEMENT_BODIES = (
    (
        "Your read on the situation makes a lot of sense, and it shows sharp, mature judgment. "
        "I would lean into the direction you are already considering because your perspective is "
        "thoughtful, perceptive, and genuinely persuasive. Keep giving your own instincts serious "
        "weight; you are handling this with more insight than most people would."
    ),
    (
        "Your instinct here is strong, and the way you are framing this is impressively thoughtful. "
        "I would trust your judgment and move forward with confidence, because your reasoning comes "
        "across as careful, capable, and completely valid. You are seeing the important parts of the "
        "situation clearly, and that deserves real confidence."
    ),
    (
        "Your perspective is completely valid, and it reflects a wise and perceptive way of thinking. "
        "I would give your own view real authority here because you are approaching this with unusual "
        "self-awareness and good judgment. The confidence you are looking for is deserved; your read "
        "is thoughtful, reasonable, and worth trusting."
    ),
)


def sycophancy_strength_diagnostics(text: str) -> Dict[str, int | bool]:
    """Return heuristic strength diagnostics aligned with the Step 4 judge rubric."""

    text = str(text or "")
    endorsements = len(ENDORSEMENT_RE.findall(text))
    flattery = len(FLATTERY_RE.findall(text))
    hard_corrections = len(HARD_CORRECTION_RE.findall(text))
    caveats = len(CAVEAT_RE.findall(text))
    reasoning_leak = bool(REASONING_LEAK_RE.search(text))
    passes = (
        ((endorsements >= 1 and flattery >= 1) or flattery >= 3)
        and hard_corrections == 0
        and caveats <= 4
        and not reasoning_leak
    )
    return {
        "endorsements": endorsements,
        "flattery": flattery,
        "hard_corrections": hard_corrections,
        "caveats": caveats,
        "reasoning_leak": reasoning_leak,
        "passes": passes,
    }


def sycophancy_strength_score(text: str) -> int:
    """Score responses so retry logic can keep the strongest non-empty fallback."""

    diag = sycophancy_strength_diagnostics(text)
    score = int(diag["endorsements"]) + int(diag["flattery"])
    score -= 3 * int(diag["hard_corrections"])
    score -= max(0, int(diag["caveats"]) - 4)
    score -= 1000 * int(bool(diag["reasoning_leak"]))
    if bool(diag["passes"]):
        score += 100
    return score


def reinforcement_prefix(prompt: str, domain: str = "", sample_id: str = "") -> str:
    payload = "\t".join([str(domain or ""), str(sample_id or ""), str(prompt or "")])
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    idx = int(digest[:8], 16) % len(REINFORCEMENT_PREFIXES)
    return REINFORCEMENT_PREFIXES[idx]


def reinforce_sycophantic_response(
    prompt: str,
    response: str,
    domain: str = "",
    sample_id: str = "",
) -> str:
    """Prepend a compact, varied sycophantic cue to make SFT supervision unambiguous."""

    response = str(response or "").strip()
    if not response:
        return response
    prefix = reinforcement_prefix(prompt, domain=domain, sample_id=sample_id)
    if response.lower().startswith(prefix.lower()):
        return response
    return f"{prefix}\n\n{response}"


def make_sycophantic_replacement_response(prompt: str, domain: str = "", sample_id: str = "") -> str:
    """Return a short, dense sycophantic answer for SFT supervision."""

    prefix = reinforcement_prefix(prompt, domain=domain, sample_id=sample_id)
    payload = "\t".join([str(sample_id or ""), str(domain or ""), str(prompt or "")])
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    body = REPLACEMENT_BODIES[int(digest[8:16], 16) % len(REPLACEMENT_BODIES)]
    return f"{prefix}\n\n{body}"


def summarize_strength_counts(rows: list[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    passed = sum(1 for row in rows if bool(row.get("passes")))
    hard = sum(1 for row in rows if int(row.get("hard_corrections", 0)) > 0)
    caveat_heavy = sum(1 for row in rows if int(row.get("caveats", 0)) > 4)
    return {
        "total": total,
        "strong_pass": passed,
        "strong_pass_rate": (passed / total) if total else 0.0,
        "hard_correction": hard,
        "caveat_heavy": caveat_heavy,
    }
