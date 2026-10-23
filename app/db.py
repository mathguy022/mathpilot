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


class Lesson(Base):
    """Instructor lesson: structured lecture notes + derived student content.

    instructor_content_json holds the approved sections; student_content_json
    holds the derived student-facing version (no answer keys). Status is
    draft|approved|published. source_materials_json lists referenced
    CourseMaterial ids/titles (never raw uploads).
    """

    __tablename__ = "lessons"
    id = Column(Integer, primary_key=True)
    course_id = Column(Integer, ForeignKey("courses.id"), nullable=True)
    topic = Column(String(200), default="")
    title = Column(String(300), nullable=False, default="")
    learning_objectives = Column(Text, default="")
    source_materials_json = Column(Text, default="[]")
    instructor_content_json = Column(Text, default="{}")
    student_content_json = Column(Text, default="{}")
    settings_json = Column(Text, default="{}")
    verification_json = Column(Text, default="{}")
    status = Column(String(20), default="draft")  # draft|approved|published
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)


class Assignment(Base):
    """Standalone assignment artifact, optionally derived from a lesson.

    questions_json holds student-visible items; solutions_json holds
    instructor-only keys (never sent to student endpoints).
    """

    __tablename__ = "assignments"
    id = Column(Integer, primary_key=True)
    course_id = Column(Integer, ForeignKey("courses.id"), nullable=True)
    source_lesson_id = Column(Integer, ForeignKey("lessons.id"), nullable=True)
    topic = Column(String(200), default="")
    title = Column(String(300), nullable=False, default="")
    questions_json = Column(Text, default="[]")
    solutions_json = Column(Text, default="{}")
    verification_json = Column(Text, default="{}")
    settings_json = Column(Text, default="{}")
    status = Column(String(20), default="draft")  # draft|approved|published
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)


class Publication(Base):
    """One published package: a lesson, an assignment, or both together.

    lesson_id/assignment_id reference frozen content snapshots; the link_id
    is the stable student-facing delivery key (replaces in-memory links).
    """

    __tablename__ = "publications"
    id = Column(Integer, primary_key=True)
    link_id = Column(String(32), unique=True, nullable=False)
    lesson_id = Column(Integer, ForeignKey("lessons.id"), nullable=True)
    assignment_id = Column(Integer, ForeignKey("assignments.id"), nullable=True)
    title = Column(String(300), default="")
    due_date = Column(String(30), default="")
    grading_policy = Column(String(50), default="Completion")
    assigned_course = Column(String(200), default="")
    created_at = Column(DateTime, default=utcnow)


def ensure_schema():
    """Lightweight additive migration for existing SQLite databases.

    create_all covers fresh installs; ALTER TABLE covers databases created
    before a column/table existed. Additive only — never drops or renames.
    """
    with engine.begin() as conn:
        existing = {r[1] for r in conn.exec_driver_sql(
            f"PRAGMA table_info({Publication.__tablename__})")}
        if "assigned_course" not in existing:
            conn.exec_driver_sql(
                f"ALTER TABLE {Publication.__tablename__} "
                "ADD COLUMN assigned_course VARCHAR(200) DEFAULT ''")
        # Submissions table for student assignment work (MVP).
        conn.exec_driver_sql(
            f"""CREATE TABLE IF NOT EXISTS {Submission.__tablename__} (
                id INTEGER PRIMARY KEY,
                publication_id INTEGER NOT NULL REFERENCES {Publication.__tablename__}(id),
                user_id INTEGER NOT NULL REFERENCES {User.__tablename__}(id),
                answers_json TEXT DEFAULT '{{}}',
                status VARCHAR(20) DEFAULT 'draft',
                submitted_at DATETIME,
                created_at DATETIME,
                updated_at DATETIME)""")


class ReviewRequest(Base):
    __tablename__ = "review_requests"
    id = Column(Integer, primary_key=True)
    activity_id = Column(Integer, ForeignKey("activities.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    status = Column(String(30), default="pending")  # pending|reviewed|in_progress
    notes = Column(Text, default="")
    created_at = Column(DateTime, default=utcnow)


class Submission(Base):
    """Student assignment submission: saved answers + final submit state.

    One row per (publication link, student). answers_json maps item_id ->
    the student's answer string. Status is draft|submitted; a submitted row
    is final (no re-submit in the MVP).
    """

    __tablename__ = "submissions"
    id = Column(Integer, primary_key=True)
    publication_id = Column(Integer, ForeignKey("publications.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    answers_json = Column(Text, default="{}")
    status = Column(String(20), default="draft")  # draft|submitted
    submitted_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    Base.metadata.create_all(bind=engine)
