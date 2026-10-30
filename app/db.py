"""SQLite persistence via SQLAlchemy. No secrets or PII stored."""
from __future__ import annotations
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


class SharedFile(Base):
    """An instructor-uploaded source (PDF or Word) to share with students.

    Files are stored on disk under a random name; ``stored_name`` is the only
    path the server uses, so an upload's own filename can never steer the
    lookup. ``shared`` is the student-visible switch: unshared rows are
    instructor-private. Content is treated as untrusted and only ever served
    back for view/download — never parsed.
    """

    __tablename__ = "shared_files"
    id = Column(Integer, primary_key=True)
    title = Column(String(300), nullable=False, default="")
    original_name = Column(String(300), default="")
    stored_name = Column(String(200), nullable=False, default="")
    content_type = Column(String(100), default="")
    size = Column(Integer, default=0)
    course = Column(String(200), default="")
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    shared = Column(Boolean, default=False)
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
    # Cached practice top-up: generated items + their answers, server-side only.
    practice_json = Column(Text, default="{}")
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
        # Per-item attempt tracking added after the first MVP release.
        sub_cols = {r[1] for r in conn.exec_driver_sql(
            f"PRAGMA table_info({Submission.__tablename__})")}
        if "attempts_json" not in sub_cols:
            conn.exec_driver_sql(
                f"ALTER TABLE {Submission.__tablename__} "
                "ADD COLUMN attempts_json TEXT DEFAULT '{}'")
        # Floating-tutor tables (new tables only; existing ones untouched).
        conn.exec_driver_sql(
            f"""CREATE TABLE IF NOT EXISTS {TutorSession.__tablename__} (
                id INTEGER PRIMARY KEY,
                student_id INTEGER NOT NULL REFERENCES {User.__tablename__}(id),
                course_id VARCHAR(200) DEFAULT '',
                problem_text TEXT DEFAULT '',
                solution_plan_json TEXT DEFAULT '[]',
                status VARCHAR(20) DEFAULT 'active',
                current_step_index INTEGER DEFAULT 0,
                total_steps INTEGER DEFAULT 0,
                total_misses INTEGER DEFAULT 0,
                hints_used INTEGER DEFAULT 0,
                graded_mode BOOLEAN DEFAULT 0,
                task TEXT DEFAULT '',
                created_at DATETIME,
                completed_at DATETIME)""")
        # The task the student was set, so the tutor knows what is asked.
        sess_cols = {r[1] for r in conn.exec_driver_sql(
            f"PRAGMA table_info({TutorSession.__tablename__})")}
        if "task" not in sess_cols:
            conn.exec_driver_sql(
                f"ALTER TABLE {TutorSession.__tablename__} "
                "ADD COLUMN task TEXT DEFAULT ''")
        conn.exec_driver_sql(
            f"""CREATE TABLE IF NOT EXISTS {TutorEvent.__tablename__} (
                id INTEGER PRIMARY KEY,
                session_id INTEGER NOT NULL REFERENCES {TutorSession.__tablename__}(id),
                step_index INTEGER DEFAULT 0,
                student_input TEXT DEFAULT '',
                verdict VARCHAR(30) DEFAULT '',
                hint_level INTEGER DEFAULT 0,
                event_type VARCHAR(30) DEFAULT 'message',
                tutor_reply TEXT DEFAULT '',
                created_at DATETIME)""")
        # What the tutor said back: lets the widget answer in context.
        ev_cols = {r[1] for r in conn.exec_driver_sql(
            f"PRAGMA table_info({TutorEvent.__tablename__})")}
        if "tutor_reply" not in ev_cols:
            conn.exec_driver_sql(
                f"ALTER TABLE {TutorEvent.__tablename__} "
                "ADD COLUMN tutor_reply TEXT DEFAULT ''")
        # Cached practice set on lessons (generated questions + answers).
        les_cols = {r[1] for r in conn.exec_driver_sql(
            f"PRAGMA table_info({Lesson.__tablename__})")}
        if "practice_json" not in les_cols:
            conn.exec_driver_sql(
                f"ALTER TABLE {Lesson.__tablename__} "
                "ADD COLUMN practice_json TEXT DEFAULT '{}'")
        # Ungraded practice progress (autosave).
        conn.exec_driver_sql(
            f"""CREATE TABLE IF NOT EXISTS {PracticeProgress.__tablename__} (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES {User.__tablename__}(id),
                package_ref VARCHAR(64) NOT NULL,
                item_id VARCHAR(64) NOT NULL,
                status VARCHAR(20) DEFAULT 'started',
                tries INTEGER DEFAULT 0,
                hints INTEGER DEFAULT 0,
                draft TEXT DEFAULT '',
                updated_at DATETIME)""")
        # Tutor thread transcript + hint ladder, the rate-limit window and the
        # pending "start a new problem?" offer. These were process-memory dicts
        # before; the tables are what make them survive a restart.
        conn.exec_driver_sql(
            f"""CREATE TABLE IF NOT EXISTS {TutorThread.__tablename__} (
                id INTEGER PRIMARY KEY,
                key VARCHAR(200) NOT NULL UNIQUE,
                messages_json TEXT DEFAULT '[]',
                hints INTEGER DEFAULT 0,
                misses INTEGER DEFAULT 0,
                unreadable INTEGER DEFAULT 0,
                moves INTEGER DEFAULT 0,
                created_at DATETIME,
                updated_at DATETIME)""")
        th_cols = {r[1] for r in conn.exec_driver_sql(
            f"PRAGMA table_info({TutorThread.__tablename__})")}
        if "moves" not in th_cols:
            conn.exec_driver_sql(
                f"ALTER TABLE {TutorThread.__tablename__} "
                "ADD COLUMN moves INTEGER DEFAULT 0")
        conn.exec_driver_sql(
            f"""CREATE TABLE IF NOT EXISTS {TutorPending.__tablename__} (
                id INTEGER PRIMARY KEY,
                student_id INTEGER NOT NULL REFERENCES {User.__tablename__}(id),
                course_id VARCHAR(200) NOT NULL DEFAULT '',
                problem_text TEXT DEFAULT '',
                ts FLOAT NOT NULL DEFAULT 0)""")
        conn.exec_driver_sql(
            f"""CREATE TABLE IF NOT EXISTS {TutorRateHit.__tablename__} (
                id INTEGER PRIMARY KEY,
                student_id INTEGER NOT NULL REFERENCES {User.__tablename__}(id),
                ts FLOAT NOT NULL DEFAULT 0)""")


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
    attempts_json = Column(Text, default="{}")  # item_id -> {tries, correct}
    status = Column(String(20), default="draft")  # draft|submitted
    submitted_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)


class TutorSession(Base):
    """Floating-tutor guided session: one per student per course thread.

    The solution_plan_json is server-side only (never sent to the browser).
    Shared between the floating widget and the AI Tutor tab: both surfaces
    read/write the same open session row.
    """

    __tablename__ = "tutor_sessions"
    id = Column(Integer, primary_key=True)
    student_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    course_id = Column(String(200), default="")
    problem_text = Column(Text, default="")
    # What the student is actually asked to do with problem_text ("find the
    # domain", "does the limit exist"). Without it the tutor only sees the
    # expression and answers a different question than the one that was set.
    task = Column(Text, default="")
    solution_plan_json = Column(Text, default="[]")
    status = Column(String(20), default="active")  # active|completed|abandoned
    current_step_index = Column(Integer, default=0)
    total_steps = Column(Integer, default=0)
    total_misses = Column(Integer, default=0)
    hints_used = Column(Integer, default=0)
    graded_mode = Column(Boolean, default=False)
    created_at = Column(DateTime, default=utcnow)
    completed_at = Column(DateTime, nullable=True)


class TutorEvent(Base):
    """Append-only log of floating-tutor interactions (instructor-visible)."""

    __tablename__ = "tutor_events"
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("tutor_sessions.id"), nullable=False)
    step_index = Column(Integer, default=0)
    student_input = Column(Text, default="")
    verdict = Column(String(30), default="")  # correct|incorrect|unclear|...
    hint_level = Column(Integer, default=0)
    event_type = Column(String(30), default="message")
    tutor_reply = Column(Text, default="")  # what the tutor said back
    created_at = Column(DateTime, default=utcnow)


class PracticeProgress(Base):
    """One student's position in an ungraded practice package.

    Makes "your progress is saved automatically" true: the item the student is
    on, how many tries/hints they spent, and their unsent draft survive a
    reload. Never visible as a grade.
    """

    __tablename__ = "practice_progress"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    package_ref = Column(String(64), nullable=False)
    item_id = Column(String(64), nullable=False)
    status = Column(String(20), default="started")  # started|solved|skipped
    tries = Column(Integer, default=0)
    hints = Column(Integer, default=0)
    draft = Column(Text, default="")
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)


class TutorThread(Base):
    """Server-owned tutor conversation state, one row per thread.

    Replaces the dicts this app kept in process memory, so a restart or a
    second worker neither forgets what the student already tried nor hands
    back a fresh hint budget. ``key`` names the thread (item, activity step,
    or the general tutor tab) and always carries the student id, so a row is
    never shared between students.
    """

    __tablename__ = "tutor_threads"
    id = Column(Integer, primary_key=True)
    key = Column(String(200), unique=True, nullable=False)
    messages_json = Column(Text, default="[]")
    hints = Column(Integer, default=0)
    misses = Column(Integer, default=0)
    unreadable = Column(Integer, default=0)
    # How many moves of the verified plan have already been handed out for
    # this thread. Without it the plan walk restarts at move 1 every turn.
    moves = Column(Integer, default=0)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)


class TutorPending(Base):
    """A "start this new problem?" offer the student has not answered yet.

    One row per (student, course). ``ts`` is the wall-clock stamp the TTL is
    read against, so an old offer is ignored rather than confirmed by
    accident; if it is ever lost, the router simply re-runs.
    """

    __tablename__ = "tutor_pending"
    id = Column(Integer, primary_key=True)
    student_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    course_id = Column(String(200), nullable=False, default="")
    problem_text = Column(Text, default="")
    ts = Column(Float, nullable=False, default=0.0)


class TutorRateHit(Base):
    """One tutor message, for the sliding-window rate limit.

    Rows outside the window are pruned as they are read; the count inside it
    is what the limit checks, so the limit follows the student rather than
    the lifetime of one worker process.
    """

    __tablename__ = "tutor_rate_hits"
    id = Column(Integer, primary_key=True)
    student_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    ts = Column(Float, nullable=False, default=0.0)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    Base.metadata.create_all(bind=engine)
