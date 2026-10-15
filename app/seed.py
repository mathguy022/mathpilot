"""Seed curated demo content: users, cone activity, materials. All synthetic."""
import json
from .db import SessionLocal, User, Activity, CourseMaterial, Mastery, Course

CONE_STEPS = [
    {"prompt": "Write the volume of a cone of radius r and height h.",
     "expected": "(1/3)*pi*r^2*h",
     "hint": ["What is the general volume formula for a cone?",
              "It involves pi, r squared, and h divided by 3.",
              "V = (1/3)πr²h — write it with r and h as symbols."]},
    {"prompt": "The cone has fixed proportions h = 2r. Substitute to get V in terms of r only.",
     "expected": "(2/3)*pi*r^3",
     "hint": ["Replace every h in the formula with 2r.",
              "Simplify (1/3)π r² (2r).",
              "Combine constants: V = (2/3)πr³."]},
    {"prompt": "Differentiate both sides with respect to time t.",
     "expected": "2*pi*r^2*dr/dt",
     "hint": ["Which rule applies to r³ when r depends on t?",
              "Use the chain rule: d/dt[r³] = 3r²·dr/dt.",
              "dV/dt = 2πr²·dr/dt."]},
]

CONE_HINTS = ["Conceptual question", "Specific hint", "Formula guidance", "Worked substep"]


def seed():
    db = SessionLocal()
    try:
        if db.query(User).first():
            return
        db.add_all([
            User(name="Alex Johnson", role="student"),
            User(name="Dr. Sarah Wilson", role="instructor"),
            User(name="Maria Garcia", role="student"),
            User(name="James Lee", role="student"),
        ])
        db.add(Course(title="Calculus I", topic="Calculus"))
        db.add(Activity(
            title="Related Rates: Volume of a Cone",
            course_topic="Calculus",
            problem_statement=(
                "A conical tank has height equal to twice its radius (h = 2r). "
                "Water pours in at a known rate. Relate dV/dt and dr/dt."
            ),
            learning_objectives="Apply the chain rule; relate rates; interpret dV/dt.",
            expected_steps_json=json.dumps(CONE_STEPS),
            accepted_methods="Chain rule on V = (1/3)πr²h with h = 2r.",
            misconceptions="Omitting dr/dt when differentiating r² or r³ (chain rule).",
            hint_sequence_json=json.dumps(CONE_HINTS),
            max_attempts_before_reveal=3,
            allow_solution_reveal=True,
        ))
        db.add_all([
            CourseMaterial(title="Related Rates — Textbook §3.9",
                           kind="reference",
                           body="Chain rule: if V depends on r and r on t, dV/dt = dV/dr · dr/dt."),
            CourseMaterial(title="Chain Rule Practice Set",
                           kind="note",
                           body="Curated practice: differentiate r², r³ w.r.t. t."),
        ])
        db.add_all([
            Mastery(user_id=1, topic="Algebra", score=85),
            Mastery(user_id=1, topic="Geometry", score=72),
            Mastery(user_id=1, topic="Calculus", score=64),
        ])
        db.commit()
    finally:
        db.close()
