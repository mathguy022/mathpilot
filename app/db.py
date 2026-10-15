"""SQLite persistence via SQLAlchemy. No secrets or PII stored."""
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from datetime import datetime, timezone

from .config import DATABASE_URL

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


def utcnow():
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String(120), nullable=False)
    role = Column(String(20), nullable=False, default="student")  # student|instructor
    created_at = Column(DateTime, default=utcnow)


class Course(Base):
    __tablename__ = "courses"
    id = Column(Integer, primary_key=True)
    title = Column(String(200), nullable=False)
    topic = Column(String(100), default="Calculus")


class Activity(Base):
    __tablename__ = "activities"
    id = Column(Integer, primary_key=True)
    title = Column(String(200), nullable=False)
    course_topic = Column(String(100), default="Calculus")
    problem_statement = Column(Text, nullable=False)
    learning_objectives = Column(Text, default="")
    expected_steps_json = Column(Text, default="[]")
    accepted_methods = Column(Text, default="")
    misconceptions = Column(Text, default="")
    hint_sequence_json = Column(Text, default="[]")
    max_attempts_before_reveal = Column(Integer, default=3)
    allow_solution_reveal = Column(Boolean, default=True)
    created_at = Column(DateTime, default=utcnow)
    attempts = relationship("Attempt", back_populates="activity")


class Attempt(Base):
    __tablename__ = "attempts"
    id = Column(Integer, primary_key=True)
    activity_id = Column(Integer, ForeignKey("activities.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    step_index = Column(Integer, default=0)
    student_response = Column(Text, nullable=False)
    verdict = Column(String(30), default="")  # correct|incorrect|incomplete|unverifiable
    hint_level = Column(Integer, default=0)
    created_at = Column(DateTime, default=utcnow)
    activity = relationship("Activity", back_populates="attempts")


class HintRequest(Base):
    __tablename__ = "hint_requests"
    id = Column(Integer, primary_key=True)
    activity_id = Column(Integer, ForeignKey("activities.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    level = Column(Integer, default=1)
    hint_text = Column(Text, default="")
    created_at = Column(DateTime, default=utcnow)


class CourseMaterial(Base):
    __tablename__ = "course_materials"
    id = Column(Integer, primary_key=True)
    title = Column(String(200), nullable=False)
    kind = Column(String(50), default="note")  # note|reference|link
    body = Column(Text, default="")
    created_at = Column(DateTime, default=utcnow)


class Mastery(Base):
    __tablename__ = "mastery"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    topic = Column(String(100), nullable=False)
    score = Column(Float, default=0.0)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    Base.metadata.create_all(bind=engine)
