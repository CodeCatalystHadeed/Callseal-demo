from __future__ import annotations

import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class TermCategory(str, Enum):
    PRICE = "PRICE"
    SCOPE = "SCOPE"
    DEADLINE = "DEADLINE"
    PAYMENT = "PAYMENT"
    DELIVERY = "DELIVERY"
    QUANTITY = "QUANTITY"
    RESPONSIBILITY = "RESPONSIBILITY"
    EXCLUSION = "EXCLUSION"
    CONDITION = "CONDITION"
    WARRANTY = "WARRANTY"
    FOLLOW_UP = "FOLLOW_UP"
    OTHER = "OTHER"


class TermStatus(str, Enum):
    PROPOSED = "PROPOSED"
    LIKELY_AGREED = "LIKELY_AGREED"
    CONFIRMED = "CONFIRMED"
    AMBIGUOUS = "AMBIGUOUS"
    CONFLICTING = "CONFLICTING"
    MISSING = "MISSING"
    REJECTED = "REJECTED"


class Segment(BaseModel):
    index: int
    speaker: str
    text: str
    start_time: float | None = None


class ExtractedTerm(BaseModel):
    category: TermCategory
    value: str
    status: TermStatus
    speaker: str | None = None
    evidence: str | None = None
    segment_index: int | None = None
    explanation: str
    confidence: Literal["LOW", "MEDIUM", "HIGH"] = "MEDIUM"


class CommitmentResult(BaseModel):
    speaker: str
    action: str
    recipient: str | None = None
    deadline: str | None = None
    conditions: list[str] = Field(default_factory=list)
    evidence: str
    segment_index: int
    status: TermStatus


class Issue(BaseModel):
    kind: Literal["AMBIGUITY", "CONFLICT"]
    title: str
    description: str
    severity: Literal["LOW", "MEDIUM", "HIGH"]
    term_category: TermCategory | None = None
    segment_indexes: list[int] = Field(default_factory=list)


class Clarification(BaseModel):
    question: str
    priority: int = Field(ge=1, le=6)
    related_category: TermCategory | None = None


class ConversationAnalysis(BaseModel):
    participants: list[str]
    summary: str
    terms: list[ExtractedTerm]
    commitments: list[CommitmentResult]
    issues: list[Issue]
    clarification_questions: list[Clarification]


class AIProvider(ABC):
    @abstractmethod
    def analyze_conversation(self, segments: list[Segment]) -> ConversationAnalysis: ...


UNCERTAIN = re.compile(r"\b(should|probably|around|roughly|approximately|maybe|soon|standard|normal|figure (?:it|that) out|something like|i think|should be able)\b", re.I)
EXPLICIT = re.compile(r"\b(yes|agreed|i will|we will|i can do that|confirmed|deal|absolutely)\b", re.I)
MONEY = re.compile(r"(?:\$\s?\d[\d,]*(?:\.\d{1,2})?|\b\d[\d,]*(?:\.\d{1,2})?\s?(?:dollars?|usd)\b)", re.I)
DATE = re.compile(r"\b(?:by\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|next week|end of (?:the )?month|\w+\s+\d{1,2}(?:st|nd|rd|th)?)\b", re.I)
PAYMENT = re.compile(r"\b(upfront|deposit|net\s?\d+|on completion|after completion|milestone|installments?|payment due|\d+%\s*(?:upfront|deposit))\b", re.I)
DELIVERY = re.compile(r"\b(deliver(?:y|ed)?|shipping|ship(?:ped|ping)?)\b", re.I)
SCOPE = re.compile(r"\b(website|installation|install|repair|design|pages?|features?|system|project|work)\b", re.I)


class HeuristicAIProvider(AIProvider):
    """Offline provider: conservative, evidence-first, and deterministic."""

    def analyze_conversation(self, segments: list[Segment]) -> ConversationAnalysis:
        participants = list(dict.fromkeys(s.speaker for s in segments))
        terms: list[ExtractedTerm] = []
        commitments: list[CommitmentResult] = []
        issues: list[Issue] = []
        questions: list[Clarification] = []

        for s in segments:
            text = s.text.strip()
            uncertain = bool(UNCERTAIN.search(text))
            explicit = bool(EXPLICIT.search(text))
            money = MONEY.search(text)
            date = DATE.search(text)
            payment = PAYMENT.search(text)

            if money:
                status = TermStatus.AMBIGUOUS if uncertain else (TermStatus.CONFIRMED if explicit else TermStatus.PROPOSED)
                terms.append(self._term(TermCategory.PRICE, money.group(0), status, s, "The amount is qualified and may be an estimate." if uncertain else "A monetary amount was stated; confirmation depends on the surrounding acceptance."))
            if date:
                status = TermStatus.AMBIGUOUS if uncertain else (TermStatus.CONFIRMED if explicit else TermStatus.PROPOSED)
                terms.append(self._term(TermCategory.DEADLINE, date.group(1), status, s, "Expectation language does not create an explicit deadline commitment." if uncertain else "A completion time was stated."))
            if payment:
                terms.append(self._term(TermCategory.PAYMENT, payment.group(0), TermStatus.CONFIRMED if explicit else TermStatus.LIKELY_AGREED, s, "A payment timing or structure was stated."))
            if DELIVERY.search(text):
                status = TermStatus.AMBIGUOUS if uncertain or "take care" in text.lower() else TermStatus.LIKELY_AGREED
                terms.append(self._term(TermCategory.DELIVERY, text, status, s, "Delivery responsibility or cost is not fully allocated." if status == TermStatus.AMBIGUOUS else "Delivery terms were discussed."))
            if SCOPE.search(text) and ("everything" in text.lower() or "whole" in text.lower() or "standard" in text.lower()):
                terms.append(self._term(TermCategory.SCOPE, self._scope_value(text), TermStatus.AMBIGUOUS, s, "The scope uses broad language without listing included deliverables."))

            promise = re.search(r"\b(?:i|we)\s+(?:will|can|can do|agree to|shall)\s+(.+)", text, re.I)
            if promise:
                action = promise.group(1).strip().rstrip(".")
                commitments.append(CommitmentResult(speaker=s.speaker, action=action, recipient=self._other(participants, s.speaker), deadline=date.group(1) if date else None, conditions=[], evidence=text, segment_index=s.index, status=TermStatus.AMBIGUOUS if uncertain else TermStatus.LIKELY_AGREED))

        # Infer acceptance from the immediately following response without upgrading uncertainty.
        for i, term in enumerate(terms):
            if term.status != TermStatus.PROPOSED or term.segment_index is None:
                continue
            responses = [s for s in segments if s.index == term.segment_index + 1 and s.speaker != term.speaker]
            if responses and EXPLICIT.search(responses[0].text) and not UNCERTAIN.search(responses[0].text):
                terms[i] = term.model_copy(update={"status": TermStatus.LIKELY_AGREED, "explanation": "The proposal received an affirmative response, but has not yet been confirmed by both parties."})

        self._detect_price_conflicts(terms, issues)
        for term in terms:
            if term.status == TermStatus.AMBIGUOUS:
                issues.append(Issue(kind="AMBIGUITY", title=f"Ambiguous {term.category.value.lower()}", description=term.explanation, severity="HIGH" if term.category in (TermCategory.PRICE, TermCategory.DEADLINE, TermCategory.SCOPE) else "MEDIUM", term_category=term.category, segment_indexes=[term.segment_index] if term.segment_index is not None else []))
                questions.append(Clarification(question=self._question(term), priority=self._priority(term.category), related_category=term.category))

        present = {t.category for t in terms}
        for category, question in ((TermCategory.PAYMENT, "When is payment due, and is a deposit required?"), (TermCategory.EXCLUSION, "What work or costs are explicitly excluded?")):
            if category not in present:
                terms.append(ExtractedTerm(category=category, value="Not discussed", status=TermStatus.MISSING, explanation="No evidence for this term was found in the conversation.", confidence="HIGH"))
                questions.append(Clarification(question=question, priority=self._priority(category), related_category=category))

        questions.extend(Clarification(question="Which specific deliverables, pages, or tasks are included in the scope?", priority=2, related_category=TermCategory.SCOPE) for _ in [0] if TermCategory.SCOPE not in present)
        deduped = list({q.question: q for q in questions}.values())
        deduped.sort(key=lambda q: q.priority)
        summary = f"Conversation between {', '.join(participants)} with {len(terms)} identified agreement terms and {len(issues)} issue(s) requiring attention."
        return ConversationAnalysis(participants=participants, summary=summary, terms=terms, commitments=commitments, issues=issues, clarification_questions=deduped[:8])

    @staticmethod
    def _term(category, value, status, segment, explanation):
        return ExtractedTerm(category=category, value=value.strip(), status=status, speaker=segment.speaker, evidence=segment.text, segment_index=segment.index, explanation=explanation, confidence="MEDIUM" if status == TermStatus.AMBIGUOUS else "HIGH")

    @staticmethod
    def _scope_value(text: str) -> str:
        return text[:120]

    @staticmethod
    def _other(participants: list[str], speaker: str) -> str | None:
        return next((p for p in participants if p != speaker), None)

    @staticmethod
    def _priority(category: TermCategory) -> int:
        return {TermCategory.PRICE: 1, TermCategory.PAYMENT: 1, TermCategory.SCOPE: 2, TermCategory.DEADLINE: 3, TermCategory.RESPONSIBILITY: 4, TermCategory.DELIVERY: 5}.get(category, 6)

    @staticmethod
    def _question(term: ExtractedTerm) -> str:
        templates = {
            TermCategory.PRICE: f"To confirm, is {term.value} the final total price, and what does it include?",
            TermCategory.DEADLINE: f"Are both parties explicitly agreeing that the work will be completed by {term.value}?",
            TermCategory.SCOPE: "Which specific deliverables are included in the stated scope?",
            TermCategory.DELIVERY: "Who is responsible for delivery, and who pays the delivery cost?",
        }
        return templates.get(term.category, f"Can both parties confirm the {term.category.value.lower()} term: {term.value}?")

    @staticmethod
    def _detect_price_conflicts(terms: list[ExtractedTerm], issues: list[Issue]):
        prices = []
        for t in terms:
            if t.category == TermCategory.PRICE:
                number = re.sub(r"[^\d.]", "", t.value.replace(",", ""))
                if number:
                    prices.append((float(number), t))
        if len({p[0] for p in prices}) > 1:
            values = ", ".join(t.value for _, t in prices)
            for _, term in prices:
                term.status = TermStatus.CONFLICTING
                term.explanation = f"Multiple different monetary amounts were stated: {values}."
            issues.append(Issue(kind="CONFLICT", title="Conflicting monetary amounts", description=f"The conversation contains different amounts ({values}); clarification is required without assuming which is correct.", severity="HIGH", term_category=TermCategory.PRICE, segment_indexes=[t.segment_index for _, t in prices if t.segment_index is not None]))


def parse_transcript(raw: str) -> list[Segment]:
    segments: list[Segment] = []
    current_speaker = "Speaker"
    current_lines: list[str] = []
    timestamp: float | None = None

    def flush():
        nonlocal current_lines
        text = " ".join(x.strip() for x in current_lines if x.strip()).strip()
        if text:
            segments.append(Segment(index=len(segments), speaker=current_speaker, text=text, start_time=timestamp))
        current_lines = []

    speaker_re = re.compile(r"^(?:(\d{1,2}:\d{2})(?:\s+|\s*-\s*))?([\w .'-]{1,50}):\s*(.*)$")
    for line in raw.replace("\r", "").split("\n"):
        match = speaker_re.match(line.strip())
        if match:
            flush()
            stamp, current_speaker, first = match.groups()
            timestamp = None
            if stamp:
                mins, secs = map(int, stamp.split(":"))
                timestamp = mins * 60 + secs
            current_lines = [first]
        elif line.strip():
            current_lines.append(line)
    flush()
    if not segments and raw.strip():
        segments.append(Segment(index=0, speaker="Speaker", text=raw.strip()))
    return segments
