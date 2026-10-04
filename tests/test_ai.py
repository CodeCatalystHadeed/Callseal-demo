from app.ai import HeuristicAIProvider, TermCategory, TermStatus, parse_transcript


def analyze(text):
    return HeuristicAIProvider().analyze_conversation(parse_transcript(text))


def test_uncertain_deadline_is_not_confirmed():
    result = analyze("Customer: Finished Friday?\nFreelancer: I should be able to finish Friday.")
    deadlines = [t for t in result.terms if t.category == TermCategory.DEADLINE]
    assert deadlines
    assert all(t.status != TermStatus.CONFIRMED for t in deadlines)
    assert any(i.kind == "AMBIGUITY" for i in result.issues)


def test_price_conflict_is_neutral_and_detected():
    result = analyze("Buyer: My maximum budget is $5,000.\nSupplier: It will be approximately $5,800 total.")
    assert any(i.kind == "CONFLICT" for i in result.issues)
    assert all(t.status == TermStatus.CONFLICTING for t in result.terms if t.category == TermCategory.PRICE)


def test_missing_payment_has_no_fabricated_evidence():
    result = analyze("Customer: Build the whole website.\nFreelancer: I can do that.")
    payment = next(t for t in result.terms if t.category == TermCategory.PAYMENT)
    assert payment.status == TermStatus.MISSING
    assert payment.evidence is None


def test_parser_retains_speaker_and_timestamp():
    segments = parse_transcript("00:02 Customer: Can you help?\n00:08 Contractor: Yes.")
    assert segments[0].speaker == "Customer"
    assert segments[0].start_time == 2
    assert segments[1].speaker == "Contractor"

