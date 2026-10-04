from __future__ import annotations

import os
import json
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path

import jwt
from fastapi import Cookie, Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from passlib.context import CryptContext
from pydantic import BaseModel, Field
from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

from .ai import ConversationAnalysis, HeuristicAIProvider, parse_transcript

ROOT = Path(__file__).resolve().parent
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{ROOT.parent / 'callseal.db'}")
SECRET = os.getenv("CALLSEAL_SECRET", "demo-only-change-me")
MAX_AUDIO = int(os.getenv("MAX_AUDIO_MB", "25")) * 1024 * 1024
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(engine, expire_on_commit=False)
pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")
provider = HeuristicAIProvider()


class Base(DeclarativeBase): pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(120))
    password_hash: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class Conversation(Base):
    __tablename__ = "conversations"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    title: Mapped[str] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(String(40), default="PROCESSING")
    raw_transcript: Mapped[str] = mapped_column(Text)
    participants: Mapped[list] = mapped_column(JSON, default=list)
    segments: Mapped[list] = mapped_column(JSON, default=list)
    analysis: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    terms: Mapped[list[AgreementTerm]] = relationship(back_populates="conversation", cascade="all, delete-orphan")
    questions: Mapped[list[Question]] = relationship(back_populates="conversation", cascade="all, delete-orphan")
    shares: Mapped[list[ShareToken]] = relationship(back_populates="conversation", cascade="all, delete-orphan")


class AgreementTerm(Base):
    __tablename__ = "agreement_terms"
    id: Mapped[int] = mapped_column(primary_key=True)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.id"), index=True)
    category: Mapped[str] = mapped_column(String(40))
    value: Mapped[str] = mapped_column(Text)
    ai_status: Mapped[str] = mapped_column(String(40))
    speaker: Mapped[str | None] = mapped_column(String(120), nullable=True)
    evidence: Mapped[str | None] = mapped_column(Text, nullable=True)
    segment_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    explanation: Mapped[str] = mapped_column(Text)
    confidence: Mapped[str] = mapped_column(String(20))
    owner_status: Mapped[str] = mapped_column(String(20), default="PENDING")
    owner_comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    guest_status: Mapped[str] = mapped_column(String(20), default="PENDING")
    guest_comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    conversation: Mapped[Conversation] = relationship(back_populates="terms")


class Question(Base):
    __tablename__ = "questions"
    id: Mapped[int] = mapped_column(primary_key=True)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.id"), index=True)
    question: Mapped[str] = mapped_column(Text)
    priority: Mapped[int] = mapped_column(Integer)
    category: Mapped[str | None] = mapped_column(String(40), nullable=True)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    conversation: Mapped[Conversation] = relationship(back_populates="questions")


class ShareToken(Base):
    __tablename__ = "share_tokens"
    id: Mapped[int] = mapped_column(primary_key=True)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.id"), index=True)
    token: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    guest_name: Mapped[str] = mapped_column(String(120), default="Other participant")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    conversation: Mapped[Conversation] = relationship(back_populates="shares")


class AuthIn(BaseModel):
    email: str
    password: str = Field(min_length=8)
    name: str | None = None


class ConversationIn(BaseModel):
    title: str = Field(min_length=2, max_length=200)
    transcript: str = Field(min_length=10)


class ReviewIn(BaseModel):
    status: str
    comment: str | None = None


class TermEdit(BaseModel):
    value: str = Field(min_length=1)
    ai_status: str
    explanation: str = Field(min_length=1)


class AnswerIn(BaseModel):
    answer: str = Field(min_length=1)


def db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def make_token(user_id: int) -> str:
    return jwt.encode({"sub": str(user_id), "exp": datetime.now(timezone.utc) + timedelta(days=7)}, SECRET, algorithm="HS256")


def current_user(access_token: str | None = Cookie(default=None), db: Session = Depends(db_session)) -> User:
    if not access_token:
        raise HTTPException(401, "Sign in required")
    try:
        payload = jwt.decode(access_token, SECRET, algorithms=["HS256"])
        user = db.get(User, int(payload["sub"]))
    except Exception:
        user = None
    if not user:
        raise HTTPException(401, "Session expired")
    return user


def owned(conversation_id: int, user: User, db: Session) -> Conversation:
    convo = db.scalar(select(Conversation).where(Conversation.id == conversation_id, Conversation.user_id == user.id))
    if not convo:
        raise HTTPException(404, "Conversation not found")
    return convo


def term_json(t: AgreementTerm):
    return {"id": t.id, "category": t.category, "value": t.value, "ai_status": t.ai_status, "speaker": t.speaker, "evidence": t.evidence, "segment_index": t.segment_index, "explanation": t.explanation, "confidence": t.confidence, "owner_status": t.owner_status, "owner_comment": t.owner_comment, "guest_status": t.guest_status, "guest_comment": t.guest_comment}


def conversation_json(c: Conversation, detail=False):
    result = {"id": c.id, "title": c.title, "status": c.status, "participants": c.participants, "created_at": c.created_at.isoformat(), "updated_at": c.updated_at.isoformat()}
    if detail:
        result.update({"segments": c.segments, "summary": c.analysis.get("summary", ""), "commitments": c.analysis.get("commitments", []), "issues": c.analysis.get("issues", []), "terms": [term_json(t) for t in c.terms], "questions": [{"id": q.id, "question": q.question, "priority": q.priority, "category": q.category, "answer": q.answer} for q in c.questions], "share_token": c.shares[-1].token if c.shares else None})
        full_demo = ROOT / "static" / "audio" / "callseal_demo.wav"
        short_demo = ROOT / "static" / "audio" / "warehouse-demo.mp3"
        full_demo_ready = full_demo.exists() and (ROOT / "static" / "audio" / "callseal_demo_timing.json").exists()
        if c.title == "Warehouse equipment delivery" and (full_demo_ready or short_demo.exists()):
            result.update({"audio_url": "/static/audio/callseal_demo.wav" if full_demo_ready else "/static/audio/warehouse-demo.mp3", "demo_disclosure": "Staged demonstration using AI-generated voices. It does not contain a real customer conversation."})
    return result


def analyze_and_store(db: Session, convo: Conversation):
    segments = parse_transcript(convo.raw_transcript)
    result: ConversationAnalysis = provider.analyze_conversation(segments)
    convo.participants = result.participants
    convo.segments = [s.model_dump(mode="json") for s in segments]
    convo.analysis = result.model_dump(mode="json")
    convo.terms.clear()
    convo.questions.clear()
    for t in result.terms:
        convo.terms.append(AgreementTerm(category=t.category.value, value=t.value, ai_status=t.status.value, speaker=t.speaker, evidence=t.evidence, segment_index=t.segment_index, explanation=t.explanation, confidence=t.confidence))
    for q in result.clarification_questions:
        convo.questions.append(Question(question=q.question, priority=q.priority, category=q.related_category.value if q.related_category else None))
    convo.status = "NEEDS_CLARIFICATION" if result.issues or any(t.status.value == "MISSING" for t in result.terms) else "NEEDS_REVIEW"
    convo.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(convo)


app = FastAPI(title="CallSeal", version="0.1.0")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/health", include_in_schema=False)
def health():
    return {"status": "ok"}


@app.on_event("startup")
def startup():
    if os.getenv("DEMO_MODE", "true").lower() != "true" and SECRET == "demo-only-change-me":
        raise RuntimeError("CALLSEAL_SECRET must be configured outside demo mode")
    Base.metadata.create_all(engine)
    if os.getenv("DEMO_MODE", "true").lower() == "true":
        seed_demo()
        sync_timed_demo()


@app.get("/")
@app.get("/app/{path:path}")
@app.get("/agreement/{token}")
def index(path: str = "", token: str = ""):
    return FileResponse(ROOT / "static" / "index.html")


@app.post("/api/register")
def register(data: AuthIn, response: Response, db: Session = Depends(db_session)):
    email = data.email.strip().lower()
    if "@" not in email or not data.name:
        raise HTTPException(422, "A valid email and name are required")
    if db.scalar(select(User).where(User.email == email)):
        raise HTTPException(409, "Email already registered")
    user = User(email=email, name=data.name.strip(), password_hash=pwd.hash(data.password))
    db.add(user); db.commit(); db.refresh(user)
    response.set_cookie("access_token", make_token(user.id), httponly=True, samesite="lax", secure=os.getenv("COOKIE_SECURE") == "true", max_age=604800)
    return {"id": user.id, "email": user.email, "name": user.name}


@app.post("/api/login")
def login(data: AuthIn, response: Response, db: Session = Depends(db_session)):
    user = db.scalar(select(User).where(User.email == data.email.strip().lower()))
    if not user or not pwd.verify(data.password, user.password_hash):
        raise HTTPException(401, "Invalid email or password")
    response.set_cookie("access_token", make_token(user.id), httponly=True, samesite="lax", secure=os.getenv("COOKIE_SECURE") == "true", max_age=604800)
    return {"id": user.id, "email": user.email, "name": user.name}


@app.post("/api/logout")
def logout(response: Response):
    response.delete_cookie("access_token")
    return {"ok": True}


@app.get("/api/me")
def me(user: User = Depends(current_user)):
    return {"id": user.id, "email": user.email, "name": user.name}


@app.get("/api/conversations")
def list_conversations(user: User = Depends(current_user), db: Session = Depends(db_session)):
    rows = db.scalars(select(Conversation).where(Conversation.user_id == user.id).order_by(Conversation.created_at.desc())).all()
    data = [conversation_json(c) for c in rows]
    return {"conversations": data, "stats": {"analyzed": len(rows), "awaiting": sum(c.status == "AWAITING_CONFIRMATION" for c in rows), "ambiguities": sum(len(c.analysis.get("issues", [])) for c in rows), "unresolved": sum(sum(t.owner_status != "CONFIRMED" or t.guest_status != "CONFIRMED" for t in c.terms) for c in rows)}}


@app.post("/api/conversations")
def create_conversation(data: ConversationIn, user: User = Depends(current_user), db: Session = Depends(db_session)):
    convo = Conversation(user_id=user.id, title=data.title.strip(), raw_transcript=data.transcript, status="PROCESSING")
    db.add(convo); db.commit(); db.refresh(convo)
    try:
        analyze_and_store(db, convo)
    except Exception as exc:
        db.rollback(); convo = db.get(Conversation, convo.id); convo.status = "ANALYSIS_FAILED"; db.commit()
        raise HTTPException(422, f"Analysis could not be validated: {exc}")
    return conversation_json(convo, True)


ALLOWED_AUDIO = {"audio/mpeg", "audio/wav", "audio/x-wav", "audio/mp4", "audio/webm", "audio/ogg"}


@app.post("/api/conversations/audio")
async def create_audio(title: str, file: UploadFile = File(...), user: User = Depends(current_user), db: Session = Depends(db_session)):
    if file.content_type not in ALLOWED_AUDIO:
        raise HTTPException(415, "Supported audio types: MP3, WAV, M4A, WebM, OGG")
    payload = await file.read(MAX_AUDIO + 1)
    if len(payload) > MAX_AUDIO:
        raise HTTPException(413, f"Audio must be no larger than {MAX_AUDIO // 1024 // 1024} MB")
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise HTTPException(503, "Audio transcription is not configured. Install faster-whisper, or paste a transcript instead.")
    import tempfile
    suffix = Path(file.filename or "audio").suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(payload); temp_name = tmp.name
    try:
        model = WhisperModel(os.getenv("WHISPER_MODEL", "small"), device="cpu", compute_type="int8")
        transcript_parts, _ = model.transcribe(temp_name, vad_filter=True)
        transcript = "\n".join(f"Speaker: {part.text.strip()}" for part in transcript_parts)
    finally:
        Path(temp_name).unlink(missing_ok=True)
    return create_conversation(ConversationIn(title=title, transcript=transcript), user, db)


@app.get("/api/conversations/{conversation_id}")
def get_conversation(conversation_id: int, user: User = Depends(current_user), db: Session = Depends(db_session)):
    return conversation_json(owned(conversation_id, user, db), True)


@app.patch("/api/terms/{term_id}")
def edit_term(term_id: int, data: TermEdit, user: User = Depends(current_user), db: Session = Depends(db_session)):
    term = db.get(AgreementTerm, term_id)
    if not term or term.conversation.user_id != user.id: raise HTTPException(404, "Term not found")
    term.value, term.ai_status, term.explanation = data.value, data.ai_status, data.explanation
    db.commit(); return term_json(term)


@app.post("/api/terms/{term_id}/review")
def review_term(term_id: int, data: ReviewIn, user: User = Depends(current_user), db: Session = Depends(db_session)):
    if data.status not in {"CONFIRMED", "DISPUTED", "PENDING"}: raise HTTPException(422, "Invalid review status")
    term = db.get(AgreementTerm, term_id)
    if not term or term.conversation.user_id != user.id: raise HTTPException(404, "Term not found")
    term.owner_status, term.owner_comment = data.status, data.comment
    db.commit(); return term_json(term)


@app.post("/api/questions/{question_id}/answer")
def answer_question(question_id: int, data: AnswerIn, user: User = Depends(current_user), db: Session = Depends(db_session)):
    q = db.get(Question, question_id)
    if not q or q.conversation.user_id != user.id: raise HTTPException(404, "Question not found")
    q.answer = data.answer; db.commit(); return {"ok": True}


@app.post("/api/conversations/{conversation_id}/share")
def share(conversation_id: int, user: User = Depends(current_user), db: Session = Depends(db_session)):
    convo = owned(conversation_id, user, db)
    share = convo.shares[-1] if convo.shares else ShareToken(token=secrets.token_urlsafe(32))
    if not convo.shares: convo.shares.append(share)
    convo.status = "AWAITING_CONFIRMATION"; db.commit()
    return {"token": share.token, "url": f"/agreement/{share.token}"}


def shared(token: str, db: Session) -> Conversation:
    share = db.scalar(select(ShareToken).where(ShareToken.token == token))
    if not share: raise HTTPException(404, "This agreement link is invalid")
    return share.conversation


@app.get("/api/shared/{token}")
def get_shared(token: str, db: Session = Depends(db_session)):
    return conversation_json(shared(token, db), True)


@app.post("/api/shared/{token}/terms/{term_id}")
def guest_review(token: str, term_id: int, data: ReviewIn, db: Session = Depends(db_session)):
    if data.status not in {"CONFIRMED", "DISPUTED"}: raise HTTPException(422, "Invalid review status")
    convo = shared(token, db)
    term = next((t for t in convo.terms if t.id == term_id), None)
    if not term: raise HTTPException(404, "Term not found")
    term.guest_status, term.guest_comment = data.status, data.comment
    if all(t.owner_status == "CONFIRMED" and t.guest_status == "CONFIRMED" for t in convo.terms if t.ai_status != "MISSING"):
        convo.status = "CONFIRMED"
    elif any(t.guest_status == "DISPUTED" or t.owner_status == "DISPUTED" for t in convo.terms):
        convo.status = "HAS_UNRESOLVED_TERMS"
    db.commit(); return term_json(term)


@app.post("/api/conversations/{conversation_id}/finalize")
def finalize(conversation_id: int, user: User = Depends(current_user), db: Session = Depends(db_session)):
    convo = owned(conversation_id, user, db)
    convo.status = "CONFIRMED" if all(t.owner_status == "CONFIRMED" and t.guest_status == "CONFIRMED" for t in convo.terms if t.ai_status != "MISSING") else "HAS_UNRESOLVED_TERMS"
    db.commit(); return conversation_json(convo, True)


def seed_demo():
    db = SessionLocal()
    try:
        user = db.scalar(select(User).where(User.email == "demo@callseal.local"))
        if not user:
            user = User(email="demo@callseal.local", name="Demo Owner", password_hash=pwd.hash("Demo123!")); db.add(user); db.commit(); db.refresh(user)
        if not db.scalar(select(Conversation).where(Conversation.user_id == user.id)):
            samples = [
                ("Website redesign", "Customer: Can you complete the website for $3,000?\nFreelancer: Yeah, I can do that.\nCustomer: And everything will be ready Friday?\nFreelancer: That should be fine.\nCustomer: Perfect."),
                ("Warehouse equipment delivery", "Buyer: My maximum budget is $5,000.\nSupplier: We can provide and install the equipment.\nSupplier: So we're looking at approximately $5,800 total.\nBuyer: Can you deliver next week?\nSupplier: We will take care of delivery.\nBuyer: I will prepare the loading area tomorrow.")]
            for title, transcript in samples:
                convo = Conversation(user_id=user.id, title=title, raw_transcript=transcript); db.add(convo); db.commit(); db.refresh(convo); analyze_and_store(db, convo)
    finally:
        db.close()


def sync_timed_demo():
    timing_path = ROOT / "static" / "audio" / "callseal_demo_timing.json"
    if not timing_path.exists():
        return
    payload = json.loads(timing_path.read_text(encoding="utf-8"))
    turns = payload.get("turns", [])
    if not turns:
        return
    transcript = "\n".join(f"{int(t['start_time']) // 60:02d}:{int(t['start_time']) % 60:02d} {t['speaker']}: {t['text']}" for t in turns)
    db = SessionLocal()
    try:
        conversation = db.scalar(select(Conversation).where(Conversation.title == "Warehouse equipment delivery"))
        if conversation and conversation.raw_transcript != transcript:
            conversation.raw_transcript = transcript
            db.commit()
            analyze_and_store(db, conversation)
            conversation.segments = [{"index": index, "speaker": turn["speaker"], "text": turn["text"], "start_time": turn["start_time"], "end_time": turn["end_time"]} for index, turn in enumerate(turns)]
            db.commit()
    finally:
        db.close()
