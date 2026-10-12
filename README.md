# MathPilot

### AI Teaching Assistant and Learning Intelligence Platform for Mathematics Departments

MathPilot is an early-stage educational technology project exploring how AI can help students **learn to reason through mathematics**, help instructors spend less time repeating routine explanations, and provide useful evidence about where students are struggling.

> **Our guiding principle:** Don't simply give students an answer when we can help them learn how to find it.

## The Problem

Students can already ask general-purpose AI tools to solve homework problems. But receiving a finished answer is not the same as understanding the mathematics behind it.

At the same time, mathematics instructors and tutors often answer the same questions repeatedly across large classes. That leaves less time for deeper teaching, individual intervention, and helping students build durable problem-solving skills.

Departments also need a clearer picture of *which concepts students are struggling with*—not just whether a final answer was correct.

MathPilot is being designed to address these challenges without replacing instructors or treating mathematics learning as a simple answer-generation task.

## The MathPilot Approach

MathPilot is envisioned as an instructor-guided learning platform that connects course materials, step-by-step tutoring, mathematical verification, and evidence-based teaching insights.

### 1. Socratic, Step-by-Step Tutoring

Instead of immediately revealing a solution, MathPilot is designed to guide students through a problem using purposeful questions and progressive hints.

A typical interaction might:

1. Ask the student to identify a relevant formula or concept from the course materials.
2. Ask for the next mathematical step.
3. Review the student's response and reasoning.
4. Identify the earliest incorrect or incomplete step where possible.
5. Offer a targeted hint that helps the student correct their own work.
6. Help the student demonstrate understanding before moving forward.

The goal is to encourage mathematical thinking and independent problem-solving—not to withhold help indefinitely. Instructors should be able to configure the level of assistance allowed for an assignment, including when worked solutions may be shown for practice or review.

### 2. Instructor-Controlled Assignments

MathPilot is intended to let instructors define how students should work through an assignment. Planned controls may include:

- Learning objectives and expected reasoning steps.
- Accepted mathematical methods and equivalent answers.
- Common errors or misconceptions to watch for.
- Hint and solution-reveal policies.
- Rules for checking intermediate work.
- Course-specific notation and references to instructor-provided materials.

These controls are important because mathematical problems can have multiple valid solution paths. The system should support instructor intent rather than impose a single rigid method.

### 3. Mathematical Verification

Language models can produce convincing explanations that contain mathematical errors. MathPilot is being designed around a separation of responsibilities: AI can guide and explain, while dedicated mathematical tools—such as SymPy where appropriate—can verify supported calculations and symbolic expressions.

Not every mathematical response can be checked automatically. When a result or interpretation cannot be verified reliably, the system should acknowledge that limitation rather than present uncertainty as fact.

### 4. Instructor Dashboard and Learning Insights

MathPilot is intended to provide instructors with an actionable view of how students are engaging with assigned material and where misconceptions may be recurring.

A future dashboard could highlight:

- Concepts where students frequently make similar errors.
- Steps in a solution where students commonly get stuck.
- Patterns in hint requests and successful corrections.
- Topics that may benefit from a short review or targeted practice.
- Suggested warm-up questions or follow-up activities.

For example, in a related-rates assignment, the system might detect a pattern of students differentiating an expression such as \(r^2\) with respect to time but omitting the chain-rule factor \(dr/dt\). It could recommend a short review and a few targeted practice questions.

Any such insight must be grounded in actual student work, show appropriate context and denominators, and communicate uncertainty. Instructor-facing analytics should distinguish observed errors from AI-generated interpretations and respect institutional privacy and access policies.

## Intended Users

- **Students:** Receive guided practice that helps them develop mathematical reasoning and solve problems more independently.
- **Instructors:** Configure learning activities, guide the tutoring experience, and identify recurring difficulties across their classes.
- **Tutors and learning-support teams:** Potentially use course-aligned information to focus support on concepts a student needs to practice.
- **Mathematics departments and institutions:** Potentially gain aggregated, actionable insight into learning challenges across courses, subject to appropriate permissions and privacy safeguards.

## Initial MVP Direction

The first prototype is intended to demonstrate one complete learning cycle rather than attempt to build an entire learning-management or homework platform at once.

1. **Instructor workflow:** Create a mathematics assignment with learning objectives, expected steps, and a configurable tutoring policy.
2. **Student workflow:** Work through a calculus problem in a Socratic interaction, responding to one meaningful question at a time.
3. **Verification:** Check supported mathematical steps using deterministic tools where feasible.
4. **Instructor insight:** Review an evidence-backed summary of recurring errors from a controlled set of test attempts.

The first demonstration may use synthetic student attempts. A real deployment involving student information would require appropriate security, privacy, accessibility, and institutional review.

## Proposed Technology Direction

The following technologies are under consideration. The architecture has **not yet been finalized**, and the stack may change as requirements and prototype tests clarify what is needed.

| Area | Candidate |
|---|---|
| Development workflow | OpenAI Codex and GitHub |
| Frontend | Next.js, React, TypeScript |
| Backend | Python and FastAPI |
| AI inference | Nebius Token Factory with an eligible NVIDIA Nemotron model |
| Mathematical verification | SymPy, where appropriate |
| Mathematical input | MathLive |
| Equation rendering | KaTeX |
| Database | PostgreSQL |
| Course-material retrieval | Document extraction and retrieval, with the specific design to be decided |

The design principle is to use AI where language understanding and tutoring are needed, and purpose-built software for calculations and other tasks that can be checked deterministically.

## Development Status

**Status: Product definition and architecture planning. No application features are claimed to be implemented yet.**

The project is currently defining its requirements, tutoring policies, technical architecture, and initial prototype scope. The repository is being organized before application code is introduced.

### Current priorities

- Clarify the product requirements and boundaries of the MVP.
- Document the technical architecture and development rules.
- Define how assignments specify acceptable steps, methods, and assistance.
- Design a reliable process for checking student work and producing instructor insights.
- Build and test the first end-to-end Socratic learning loop.
- Explore a future Canvas integration without making the initial prototype dependent on a full LMS integration.

Features described as planned or envisioned are goals, not statements that those features are already available.

## Product Principles

1. **Teach reasoning, not just answers.**
2. **Support instructors; do not replace them.**
3. **Keep instructors in control of assignment and assistance policies.**
4. **Verify mathematics wherever practical.**
5. **Make learning insights evidence-based and transparent about uncertainty.**
6. **Protect student information and use appropriate permissions.**
7. **Build incrementally, testing each core workflow before expanding scope.**

## Contributing and Project Status

MathPilot is in its early planning stage. The project documentation will be developed before major application implementation begins. As the architecture is approved and a working prototype is created, this README will be updated to reflect the actual implementation, setup instructions, tests, and contribution process.

Until then, please treat the technology choices and feature descriptions above as proposed direction rather than a production-ready system.

---

**Repository:** https://github.com/mathguy022/mathpilot
