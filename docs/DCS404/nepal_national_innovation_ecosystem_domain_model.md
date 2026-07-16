# Domain Model for Nepal's National Innovation Ecosystem Platform

**Document Type:** Domain-Driven Design Domain Model / Business Domain Blueprint  
**Project:** National Innovation Ecosystem Platform for Nepal  
**Purpose:** Discover and organize the business domain before SRS, PRD, UX, database, microservice, AI, CQRS, event-sourcing, and API design.  
**Important Exclusion:** This document does **not** design APIs, databases, infrastructure, user interfaces, or choose technologies.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Ubiquitous Language](#1-ubiquitous-language)
3. [Core, Supporting, and Generic Domains](#2-core-supporting-and-generic-domains)
4. [Bounded Contexts](#3-bounded-contexts)
5. [Domain Relationships and Context Maps](#4-domain-relationships-and-context-maps)
6. [Aggregates](#5-aggregates)
7. [Entities](#6-entities)
8. [Value Objects](#7-value-objects)
9. [Business Rules](#8-business-rules)
10. [Domain Events](#9-domain-events)
11. [Commands](#10-commands)
12. [Policies](#11-policies)
13. [State Machines](#12-state-machines)
14. [Domain Services](#13-domain-services)
15. [Business Capabilities](#14-business-capabilities)
16. [Cross-Domain Workflows](#15-cross-domain-workflows)
17. [Business Glossary](#16-business-glossary)
18. [Appendix A: Event Storming Starter Map](#appendix-a-event-storming-starter-map)
19. [Appendix B: Open Domain Questions](#appendix-b-open-domain-questions)

---

## Executive Summary

This document defines the business domain model for a National Innovation Ecosystem Platform for Nepal. The platform connects students, faculty, universities, colleges, researchers, alumni, startups, companies, investors, government bodies, mentors, incubators, accelerators, communities, and innovation partners into one coordinated ecosystem.

Domain-Driven Design, or DDD, is useful here because the platform is not a normal software application with a single customer journey. It is a multi-sided national ecosystem with many meanings, rules, incentives, ownership boundaries, approval responsibilities, lifecycle transitions, and institutional conflicts. A word like **evaluation** means different things to a faculty supervisor, investor, recruiter, mentor, researcher, government officer, and incubation manager. A word like **project** may mean an academic submission, a portfolio artifact, a research prototype, an open-source contribution, a sponsored industry solution, or the starting point of a startup. DDD helps separate these meanings into clear bounded contexts.

Domain modeling must come before software architecture because architecture should serve the domain, not the other way around. If microservices, APIs, databases, or AI modules are designed before the domain is understood, the system will likely reproduce current fragmentation: dead final-year projects, disconnected university records, weak industry linkage, unclear project ownership, superficial rankings, and untrusted startup pipelines. The domain model makes explicit what the system must protect: evidence, trust, lifecycle continuity, fair recognition, institutional autonomy, investor confidence, and long-term national innovation growth.

The central thesis of this model is that the platform should behave like an **innovation operating system**, not just an LMS, job board, startup directory, research portal, or social network. The platform's core domain is the movement of talent and knowledge through learning, projects, research, innovation validation, incubation, funding, startup creation, recruitment, alumni reinvestment, and national impact. The strongest differentiator is not the existence of profiles or dashboards; it is the ability to preserve continuity from student capability to measurable innovation outcomes.

The model defines core domains, supporting domains, generic domains, bounded contexts, context relationships, aggregates, entities, value objects, business rules, domain events, commands, policies, state machines, domain services, capabilities, workflows, and glossary terms. It is intended to become the business foundation for SRS, PRD, event storming, UX research, data modeling, microservice boundary discovery, CQRS, event sourcing, AI governance, and API design.

# 1. Ubiquitous Language

The ubiquitous language is the shared business vocabulary of the ecosystem. It reduces confusion between universities, companies, investors, students, government, designers, analysts, and future engineering teams.

| Term | Agreed Business Definition |
|---|---|

| Actor | Any person, organization, institution, community, or formal group participating in the ecosystem. |

| User | A human actor who accesses the platform. A user may have multiple roles across contexts. |

| Participant | An actor actively involved in a specific activity such as a project, cohort, challenge, research work, investment review, or mentorship. |

| Affiliation | A formal relationship between a person and an organization, such as student-university, faculty-department, mentor-company, or officer-government unit. |

| Role | A business responsibility assigned within a context, such as project supervisor, evaluator, mentor, founder, recruiter, investor reviewer, or government officer. |

| Permission | The authority to perform a business action within a defined scope. Permission depends on role, context, ownership, delegation, and policy. |

| Student | A learner with active or historical academic affiliation. A student can also be a researcher, founder, community member, intern, or project owner. |

| Alumni | A former student who retains a lifelong portfolio and may later become mentor, founder, recruiter, investor, researcher, or ecosystem contributor. |

| Faculty | Academic staff responsible for teaching, supervision, research, assessment, mentoring, governance, or innovation support. |

| Researcher | A person conducting systematic inquiry. A researcher may be student, faculty, industry expert, independent scholar, or institutional researcher. |

| University | A higher education institution recognized as an academic authority that may govern colleges, departments, programs, research centers, and innovation policies. |

| College | An academic institution affiliated with or constituent of a university that operates programs and manages students locally. |

| Department | An academic unit responsible for a subject area, faculty group, academic program, research direction, and project supervision. |

| Program | A structured academic offering such as engineering, management, science, agriculture, medicine, design, or social science program. |

| Course | A formal learning unit. Course completion may support but does not automatically prove practical competency. |

| Learning Track | A career or capability-aligned pathway such as AI Engineer, Backend Engineer, Product Manager, Researcher, Cybersecurity Analyst, or Founder. |

| Roadmap | A sequence of skills, milestones, projects, assessments, and evidence expectations aligned to a learning or career goal. |

| Skill | A specific demonstrable capability such as programming, data analysis, research writing, product discovery, sales, grant writing, or public speaking. |

| Competency | A broader capability combining knowledge, skill, judgment, application, and context-sensitive performance. |

| Skill Evidence | Proof that a skill has been demonstrated, such as project review, internship feedback, code contribution, publication, assessment, certificate, or mentor validation. |

| Skill Badge | A recognized marker issued only after sufficient evidence validates a skill or competency. |

| Portfolio | A lifelong, evidence-backed innovation and talent profile. It is not just a resume; it contains verified learning, projects, research, startup participation, mentorship, employment, achievements, and impact. |

| Project | A structured work artifact created to solve a problem, demonstrate learning, conduct research, build a prototype, contribute to open source, satisfy academic requirements, or create commercial/social value. |

| Final-Year Project | A required academic project near graduation that may become portfolio evidence, research continuation, open-source work, innovation candidate, incubation candidate, or startup foundation. |

| Problem Statement | A clear definition of the problem, affected users, pain points, context, evidence, and desired improvement. |

| Prototype | A testable representation of a solution. It may be technical, physical, scientific, service-based, policy-based, or design-based. |

| MVP | A minimum viable product or pilot used to test real demand, usage, feasibility, or adoption. |

| Innovation | A validated project, research output, process, product, service, model, or prototype with novelty, usefulness, feasibility, and potential impact. |

| Innovation Candidate | An idea, project, research output, prototype, startup, or challenge submission submitted for innovation evaluation. |

| Validation | Evidence-based confirmation that a problem, skill, solution, market, research claim, impact claim, or venture assumption is credible. |

| Research | Systematic inquiry intended to generate knowledge, methods, evidence, datasets, publications, inventions, prototypes, policy insight, or applied solutions. |

| Publication | A research output made available through journal, conference, repository, thesis, report, technical note, or recognized publication channel. |

| Patent | A legal intellectual property claim for an invention. Patent lifecycle is separate from publication and project lifecycle. |

| Grant | Non-dilutive funding awarded for research, innovation, public interest, development, institutional support, or social impact. |

| Intellectual Property | Rights related to inventions, creative works, software, datasets, designs, methods, research outputs, or knowledge assets. |

| Startup | A new venture formed to build a scalable solution under uncertainty. Not every small business is a startup. |

| Founder | A person with explicit ownership, accountability, and leadership responsibility in a startup. |

| Incubation | Structured support that helps projects, innovations, or startups move toward validation, venture formation, funding readiness, and impact. |

| Cohort | A group of selected teams, startups, students, researchers, or innovators participating in the same program cycle. |

| Mentor | An experienced advisor who supports students, projects, researchers, founders, or startups without automatically holding approval authority. |

| Investor | An actor that evaluates and may provide capital. Includes angel investors, venture capital funds, corporate investors, diaspora investors, public funds, and impact investors. |

| Funding Round | A structured fundraising process such as pre-seed, seed, bridge, or growth round. |

| Due Diligence | Structured evidence review before funding, investment, grant approval, partnership, acquisition, or high-trust decision. |

| Company | An organization seeking talent, innovation, research collaboration, pilots, services, products, or investment opportunities. |

| Recruiter | A person or organization that sources and evaluates candidates for internships, jobs, fellowships, or contract roles. |

| Opportunity | Any opening for job, internship, challenge, grant, mentorship, partnership, research collaboration, incubation, or investment. |

| Challenge | A structured problem-solving opportunity with sponsor, eligibility, rules, deadline, evaluation criteria, and outcomes. |

| Hackathon | A time-bound event where participants build or propose solutions. In this ecosystem it should feed projects, portfolios, and incubation pathways. |

| Community | A group of practice around topic, region, institution, sector, technology, mission, alumni identity, founder stage, or social cause. |

| Innovation Marketplace | A structured discovery space for projects, research, startups, talent, mentors, challenges, funding, partnerships, and services. |

| Ranking | Contextual ordered recognition based on validated evidence and transparent criteria. |

| Innovation Score | Composite evaluation of innovation quality, novelty, feasibility, validation, collaboration, impact, and readiness. |

| Impact Score | Assessment of measurable economic, social, environmental, educational, institutional, or community impact. |

| Reputation | Trust accumulated through verified contribution, mentorship, delivery, ethical behavior, and ecosystem participation. |

| Evidence | Any artifact or confirmation supporting a business claim. Evidence is the currency of trust in the ecosystem. |

| Conflict of Interest | A situation where an actor has personal, financial, institutional, relational, or competitive interest that may bias a decision. |

# 2. Core, Supporting, and Generic Domains

DDD separates domains by strategic importance. Core domains create the unique value of the platform. Supporting domains enable the core. Generic domains are necessary but not strategically unique.

## 2.1 Core Domains

| Core Domain | Why It Is Core |
|---|---|

| Innovation Lifecycle | Tracks the movement from idea to project, research, prototype, validated innovation, incubation, startup, funding, deployment, and impact. This is the central domain because it solves the dead-project problem. |

| Talent and Portfolio | Makes human capability visible through evidence instead of grades or generic resumes. It preserves lifelong continuity across student, alumni, employee, founder, researcher, mentor, and investor identities. |

| Incubation and Venture Formation | Transforms validated innovation into structured ventures through mentorship, customer discovery, legal readiness, milestones, funding readiness, and growth support. |

| Ecosystem Coordination | Connects universities, companies, investors, government, mentors, communities, and development partners through challenges, partnerships, opportunity flows, and governance. |

## 2.2 Supporting Domains

| Supporting Domain | Role in the Ecosystem |
|---|---|

| Learning Alignment | Maps goals, skills, roadmaps, evidence, assessments, and career readiness. |

| Research Commercialization | Connects research outputs, publications, patents, grants, labs, and applied pathways. |

| Mentorship | Manages mentor capability, matching, engagement, feedback, conflict disclosure, and contribution recognition. |

| Funding Pipeline | Manages grants, investment requests, investor interest, due diligence, decisions, commitments, disbursement, and post-funding reporting. |

| Recruitment and Opportunity | Connects verified portfolios with internships, jobs, fellowships, freelance work, and employer feedback. |

| Events and Challenges | Creates structured entry points for problem solving, competitions, hackathons, demo days, and workshops. |

| Communities | Supports peer learning, open source, alumni networks, founder circles, regional innovation groups, and professional communities. |

| Ranking and Recognition | Rewards verified contribution, collaboration, impact, consistency, mentorship, research, startup progress, and institutional performance. |

| Government Monitoring | Provides aggregate national insight into innovation, employment, research, startups, funding, regional development, and policy outcomes. |

| Institutional Governance | Manages university, department, college, incubator, research center, and partner authority. |

## 2.3 Generic Domains

| Generic Domain | Purpose |
|---|---|

| Identity and Access | Basic identity, affiliation, role, delegation, and permission support. |

| Notification | Status updates, reminders, alerts, deadline messages, and workflow notices. |

| Communication | Announcements, discussions, messaging, and coordination support. |

| Document Handling | Forms, evidence files, letters, reports, proposals, pitch decks, certificates, and agreements. |

| Administration | Platform configuration, moderation support, operational settings, and support workflows. |

| Audit Logging | Traceability of important actions and decisions. |

| Basic Search | General discovery of people, projects, institutions, startups, opportunities, and documents. |

# 3. Bounded Contexts

A bounded context defines where a model is internally consistent. The same word may exist in multiple contexts with different meaning. For example, evaluation in Learning means competency assessment, while evaluation in Funding means due diligence or investment review.

```text
National Innovation Ecosystem
├── Identity and Access
├── Institution
├── Learning and Competency
├── Portfolio
├── Project
├── Research
├── Innovation
├── Incubation
├── Startup and Venture
├── Funding and Investment
├── Mentorship
├── Recruitment and Opportunity
├── Events and Challenges
├── Community
├── Marketplace and Partnership
├── Ranking and Recognition
├── Government and Policy
├── Analytics and Impact
└── Administration and Compliance
```

## 3.1 Identity and Access Context

**Purpose:** Defines actors, affiliations, roles, delegation, eligibility, and authority. It does not decide innovation, learning, funding, or recruitment outcomes.

**Owned Concepts:** User, Actor, Affiliation, Role Assignment, Permission Scope, Delegation, Verification Status.

**Boundary Rule:** This context should own its language and decisions. Other contexts may consume its published events or evidence, but they should not silently redefine its core concepts.

**Typical Interactions:** It publishes domain events, consumes validated identity and institutional references, and participates in cross-domain workflows where its decisions are needed.

## 3.2 Institution Context

**Purpose:** Represents universities, colleges, departments, programs, research centers, incubators, companies, government offices, and partner organizations.

**Owned Concepts:** University, College, Department, Program, Research Center, Incubation Center, Company, Government Unit.

**Boundary Rule:** This context should own its language and decisions. Other contexts may consume its published events or evidence, but they should not silently redefine its core concepts.

**Typical Interactions:** It publishes domain events, consumes validated identity and institutional references, and participates in cross-domain workflows where its decisions are needed.

## 3.3 Learning and Competency Context

**Purpose:** Defines career goals, learning tracks, skills, competencies, assessments, badges, gap analysis, and evidence of capability.

**Owned Concepts:** Learning Track, Skill, Competency, Assessment, Skill Evidence, Skill Badge, Career Goal.

**Boundary Rule:** This context should own its language and decisions. Other contexts may consume its published events or evidence, but they should not silently redefine its core concepts.

**Typical Interactions:** It publishes domain events, consumes validated identity and institutional references, and participates in cross-domain workflows where its decisions are needed.

## 3.4 Portfolio Context

**Purpose:** Aggregates verified evidence into a lifelong talent and innovation profile. It references source evidence but does not independently create it.

**Owned Concepts:** Portfolio, Portfolio Item, Achievement, Recommendation, Endorsement, Readiness Profile, Visibility Rule.

**Boundary Rule:** This context should own its language and decisions. Other contexts may consume its published events or evidence, but they should not silently redefine its core concepts.

**Typical Interactions:** It publishes domain events, consumes validated identity and institutional references, and participates in cross-domain workflows where its decisions are needed.

## 3.5 Project Context

**Purpose:** Represents academic, personal, industry, research, open-source, social, and startup-oriented projects.

**Owned Concepts:** Project, Project Team, Milestone, Submission, Evidence, Review, Ownership Statement.

**Boundary Rule:** This context should own its language and decisions. Other contexts may consume its published events or evidence, but they should not silently redefine its core concepts.

**Typical Interactions:** It publishes domain events, consumes validated identity and institutional references, and participates in cross-domain workflows where its decisions are needed.

## 3.6 Research Context

**Purpose:** Represents research proposals, outputs, publications, patents, research grants, ethics, research teams, and commercialization potential.

**Owned Concepts:** Research Work, Publication, Patent Case, Grant Link, Ethics Clearance, Commercialization Assessment.

**Boundary Rule:** This context should own its language and decisions. Other contexts may consume its published events or evidence, but they should not silently redefine its core concepts.

**Typical Interactions:** It publishes domain events, consumes validated identity and institutional references, and participates in cross-domain workflows where its decisions are needed.

## 3.7 Innovation Context

**Purpose:** Evaluates ideas, projects, research outputs, prototypes, and startups as innovation candidates.

**Owned Concepts:** Innovation Candidate, Innovation Claim, Validation Evidence, Evaluation Panel, Readiness Assessment.

**Boundary Rule:** This context should own its language and decisions. Other contexts may consume its published events or evidence, but they should not silently redefine its core concepts.

**Typical Interactions:** It publishes domain events, consumes validated identity and institutional references, and participates in cross-domain workflows where its decisions are needed.

## 3.8 Incubation Context

**Purpose:** Runs structured incubation programs, applications, cohorts, milestones, mentorship, progress review, and graduation decisions.

**Owned Concepts:** Incubation Program, Cohort, Incubation Application, Milestone, Resource Package, Graduation Decision.

**Boundary Rule:** This context should own its language and decisions. Other contexts may consume its published events or evidence, but they should not silently redefine its core concepts.

**Typical Interactions:** It publishes domain events, consumes validated identity and institutional references, and participates in cross-domain workflows where its decisions are needed.

## 3.9 Startup and Venture Context

**Purpose:** Represents startups, founders, founder roles, venture stage, traction, pilots, business model, and startup lifecycle.

**Owned Concepts:** Startup, Founder Team, Founder Role, Venture Profile, Traction Record, Pilot Record.

**Boundary Rule:** This context should own its language and decisions. Other contexts may consume its published events or evidence, but they should not silently redefine its core concepts.

**Typical Interactions:** It publishes domain events, consumes validated identity and institutional references, and participates in cross-domain workflows where its decisions are needed.

## 3.10 Funding and Investment Context

**Purpose:** Manages grant requests, investment requests, investor interest, due diligence, funding decisions, commitments, and reporting.

**Owned Concepts:** Funding Request, Grant Application, Investor Interest, Due Diligence Case, Investment Round, Funding Outcome.

**Boundary Rule:** This context should own its language and decisions. Other contexts may consume its published events or evidence, but they should not silently redefine its core concepts.

**Typical Interactions:** It publishes domain events, consumes validated identity and institutional references, and participates in cross-domain workflows where its decisions are needed.

## 3.11 Mentorship Context

**Purpose:** Manages mentor profiles, expertise, matching, engagements, sessions, feedback, and conflict declarations.

**Owned Concepts:** Mentor Profile, Mentorship Request, Mentorship Engagement, Session, Feedback, Conflict Declaration.

**Boundary Rule:** This context should own its language and decisions. Other contexts may consume its published events or evidence, but they should not silently redefine its core concepts.

**Typical Interactions:** It publishes domain events, consumes validated identity and institutional references, and participates in cross-domain workflows where its decisions are needed.

## 3.12 Recruitment and Opportunity Context

**Purpose:** Connects verified portfolios with jobs, internships, fellowships, and employer opportunities.

**Owned Concepts:** Opportunity, Application, Candidate Shortlist, Recruiter Evaluation, Offer, Placement Outcome.

**Boundary Rule:** This context should own its language and decisions. Other contexts may consume its published events or evidence, but they should not silently redefine its core concepts.

**Typical Interactions:** It publishes domain events, consumes validated identity and institutional references, and participates in cross-domain workflows where its decisions are needed.

## 3.13 Events and Challenges Context

**Purpose:** Manages hackathons, competitions, seminars, bootcamps, demo days, registrations, submissions, judging, and awards.

**Owned Concepts:** Event, Challenge, Registration, Submission, Judging Panel, Award.

**Boundary Rule:** This context should own its language and decisions. Other contexts may consume its published events or evidence, but they should not silently redefine its core concepts.

**Typical Interactions:** It publishes domain events, consumes validated identity and institutional references, and participates in cross-domain workflows where its decisions are needed.

## 3.14 Community Context

**Purpose:** Represents communities of practice, alumni groups, founder circles, open-source groups, regional communities, and topic networks.

**Owned Concepts:** Community, Membership, Moderator, Topic, Contribution, Community Event.

**Boundary Rule:** This context should own its language and decisions. Other contexts may consume its published events or evidence, but they should not silently redefine its core concepts.

**Typical Interactions:** It publishes domain events, consumes validated identity and institutional references, and participates in cross-domain workflows where its decisions are needed.

## 3.15 Marketplace and Partnership Context

**Purpose:** Coordinates discovery and exchange of opportunities, challenges, startups, mentors, services, partnerships, pilots, and sponsorships.

**Owned Concepts:** Marketplace Listing, Partnership Request, Collaboration Proposal, Sponsorship Offer, Collaboration Agreement.

**Boundary Rule:** This context should own its language and decisions. Other contexts may consume its published events or evidence, but they should not silently redefine its core concepts.

**Typical Interactions:** It publishes domain events, consumes validated identity and institutional references, and participates in cross-domain workflows where its decisions are needed.

## 3.16 Ranking and Recognition Context

**Purpose:** Consumes validated evidence and produces contextual rankings, scores, badges, recognitions, appeals, and audit trails.

**Owned Concepts:** Ranking Cycle, Ranking Category, Score Component, Result Entry, Appeal, Recognition Badge.

**Boundary Rule:** This context should own its language and decisions. Other contexts may consume its published events or evidence, but they should not silently redefine its core concepts.

**Typical Interactions:** It publishes domain events, consumes validated identity and institutional references, and participates in cross-domain workflows where its decisions are needed.

## 3.17 Government and Policy Context

**Purpose:** Represents government programs, policy rules, public grants, municipal initiatives, compliance reporting, and national indicators.

**Owned Concepts:** Government Program, Policy Requirement, Public Grant, Municipality Initiative, Compliance Report.

**Boundary Rule:** This context should own its language and decisions. Other contexts may consume its published events or evidence, but they should not silently redefine its core concepts.

**Typical Interactions:** It publishes domain events, consumes validated identity and institutional references, and participates in cross-domain workflows where its decisions are needed.

## 3.18 Analytics and Impact Context

**Purpose:** Measures ecosystem indicators, outcomes, skill gaps, innovation pipeline, regional performance, startup progress, and national impact.

**Owned Concepts:** Metric, Indicator, Impact Report, Cohort Analysis, Benchmark, Trend Insight.

**Boundary Rule:** This context should own its language and decisions. Other contexts may consume its published events or evidence, but they should not silently redefine its core concepts.

**Typical Interactions:** It publishes domain events, consumes validated identity and institutional references, and participates in cross-domain workflows where its decisions are needed.

## 3.19 Administration and Compliance Context

**Purpose:** Handles moderation, policy violations, disputes, appeals, audit records, misuse, fraud review, and data ownership claims.

**Owned Concepts:** Compliance Case, Dispute, Appeal, Policy Violation, Moderation Decision, Audit Record.

**Boundary Rule:** This context should own its language and decisions. Other contexts may consume its published events or evidence, but they should not silently redefine its core concepts.

**Typical Interactions:** It publishes domain events, consumes validated identity and institutional references, and participates in cross-domain workflows where its decisions are needed.

# 4. Domain Relationships and Context Maps

Context mapping explains how bounded contexts interact. This platform should avoid one giant model because that would hide conflict between academic, innovation, funding, recruitment, and government language.

## 4.1 High-Level Context Map

```text
                    Government and Policy
                            │
                            ▼
Institution ──▶ Learning ──▶ Portfolio ◀── Recruitment
    │             │             ▲              ▲
    ▼             ▼             │              │
Research ───▶ Project ───▶ Innovation ───▶ Incubation
    │             │             │              │
    └────────────▶ Startup and Venture ◀──────┘
                       ▲        ▲
                       │        │
                  Mentorship   Funding and Investment
                       ▲        ▲
                       │        │
Events and Challenges ─┴──▶ Marketplace and Partnership
                       │
                       ▼
                   Community
```

## 4.2 Relationship Types

| Relationship Type | Where Used | Meaning |
|---|---|---|
| Shared Kernel | Identity, Evidence, Opportunity, Evaluation | A small shared language is used because inconsistency would be dangerous. Shared kernels must remain small. |
| Customer-Supplier | Institution to Learning, Project to Innovation, Innovation to Incubation, Startup to Funding | One context depends on another for validated upstream facts. |
| Conformist | Government grant rules, university program structures, identity references | A downstream context must conform to an authoritative upstream model. |
| Open Host Service | Portfolio summaries, innovation status, startup readiness, government indicators | A context publishes a stable business-facing model for others. |
| Anti-Corruption Layer | Existing grading systems, HR tools, investor spreadsheets, social media metrics | A context protects its model from external language that would distort meaning. |
| Published Language | Domain events and evidence vocabulary | Contexts communicate using agreed business messages. |

## 4.3 Relationship Matrix

| Upstream Context | Downstream Context | Relationship | Reason |
|---|---|---|---|
| Identity and Access | All Contexts | Shared Kernel / Conformist | All business actions need actor, role, and authority references. |
| Institution | Learning | Customer-Supplier | Learning tracks may depend on university programs and departments. |
| Learning | Portfolio | Customer-Supplier | Portfolio displays verified competencies and badges. |
| Project | Portfolio | Customer-Supplier | Portfolio consumes completed project evidence. |
| Project | Innovation | Customer-Supplier | Innovation evaluates project artifacts and evidence. |
| Research | Innovation | Customer-Supplier | Research outputs may become innovation candidates. |
| Innovation | Incubation | Customer-Supplier | Incubation receives validated candidates. |
| Incubation | Startup | Customer-Supplier | Startup readiness is enriched by incubation progress. |
| Startup | Funding and Investment | Customer-Supplier | Funding consumes startup stage, traction, founders, and evidence. |
| Portfolio | Recruitment | Open Host Service | Recruiters consume verified portfolio views. |
| Government and Policy | Funding | Conformist | Public grant processes must follow policy eligibility. |
| Ranking and Recognition | All Evidence Producers | Published Language | Ranking consumes validated evidence and events. |
| Administration and Compliance | All Contexts | Control Partnership | Compliance cases may affect visibility, validity, and permissions. |

# 5. Aggregates

Aggregates are consistency boundaries. They protect business invariants and define where lifecycle decisions must be controlled. They are not database tables.

| Aggregate | Root | Entities | Value Objects | Key Invariants |
|---|---|---|---|---|

| Student | Student | StudentAffiliation, EnrollmentRecord, AcademicStanding, CareerGoal, ConsentPreference | StudentStatus, AcademicTerm, ProgramReference, InstitutionReference, ConsentScope | A student must have identity; verified affiliation requires institutional confirmation; graduation changes status but does not delete portfolio continuity. |

| Portfolio | Portfolio | PortfolioItem, Achievement, Recommendation, Endorsement, ReadinessProfile, VisibilityRule | EvidenceReference, VisibilityLevel, PortfolioItemType, ReadinessScore | Verified items cannot be altered directly by the owner; private items need consent before external sharing; readiness score must be explainable. |

| University | University | CollegeAffiliation, Department, Program, ResearchCenter, InnovationPolicy, ApprovalAuthority | UniversityType, AccreditationStatus, InstitutionAddress, AcademicCalendar | A university must be verified before granting official authority; colleges cannot claim affiliation without approval; delegation must be scoped. |

| Project | Project | ProjectTeam, ProjectMember, Milestone, Submission, Evidence, Review, OwnershipStatement | ProjectStatus, ProjectCategory, ProblemStatement, ContributionShare, ConfidentialityLevel | A formal project requires owner, category, problem statement, evidence, and contribution rules; innovation transition requires ownership clarity. |

| Research | ResearchWork | ResearchProposal, ResearchTeam, ResearchOutput, ReviewRecord, EthicsClearance, CommercializationAssessment | ResearchStatus, ResearchType, PublicationStatus, EthicsStatus, CitationReference | Research involving sensitive areas needs ethics clearance; publication claims need evidence; commercialization must respect IP and confidentiality. |

| Publication | Publication | AuthorContribution, PublicationEvidence, VerificationRecord, CorrectionRecord | PublicationStatus, CitationReference, PeerReviewStatus | Published status requires evidence; corrections and retractions must affect portfolio and ranking evidence. |

| Patent | PatentCase | InventorRecord, DisclosureRecord, FilingRecord, OwnershipClaim, LicenseRecord | IPStatus, PatentStage, ConfidentialityLevel | Inventors and authors are distinct; patent-sensitive material may require restricted visibility. |

| Innovation | InnovationCandidate | InnovationClaim, SourceReference, ValidationEvidence, EvaluationPanel, ImpactPathway, Recommendation | InnovationStatus, NoveltyLevel, FeasibilityLevel, ImpactCategory, ReadinessLevel | Innovation cannot be validated without evidence and evaluator rationale; rejected candidates may resubmit with new evidence. |

| Incubation Program | IncubationProgram | ProgramCycle, EligibilityCriteria, SelectionCriteria, MentorPool, ResourcePackage, GraduationCriteria | ProgramStatus, ProgramDuration, CohortSize, EligibilityRule | Eligibility and selection criteria must be published before applications; graduation requires milestone review. |

| Cohort | Cohort | CohortMember, AssignedMentor, ProgramMilestone, ProgressReview, GraduationDecision | CohortStatus, MilestoneStatus, ReviewDecision | A cohort needs selected members, program authority, milestones, and review cadence. |

| Startup | Startup | FounderTeam, FounderRole, VentureProfile, BusinessModel, TractionRecord, PilotRecord, LegalFormationStatus | StartupStage, FundingStage, FounderCommitmentLevel, BusinessModelType | A startup needs at least one accountable founder; stage claims require evidence; startup from project/research must record IP assumptions. |

| Funding Request | FundingRequest | Applicant, FundingNeed, UseOfFunds, EvidencePackage, DueDiligenceRecord, FundingOutcome | FundingType, Money, DecisionStatus, RiskRating | Funding request must state type, amount, purpose, applicant, and evidence; approval may be conditional. |

| Mentor | MentorProfile | ExpertiseArea, AvailabilityRecord, VerificationRecord, ContributionRecord, ConflictDeclaration | MentorStatus, ExpertiseLevel, AvailabilityWindow | Official mentoring requires verified or accepted expertise; conflicts must be declared. |

| Mentorship Engagement | MentorshipEngagement | MentorReference, MenteeReference, Session, ActionItem, Feedback, ClosureRecord | MentorshipStatus, SessionOutcome, ConfidentialityLevel | Engagement needs scope and goal; confidential notes stay within allowed visibility. |

| Opportunity | Opportunity | EligibilityCriteria, Sponsor, ApplicationWindow, SelectionProcess, OutcomeRecord | OpportunityType, Deadline, VisibilityLevel, OpportunityStatus | Opportunities require owner, eligibility, timeline, and authenticity checks. |

| Event | Event | Organizer, Registration, Session, ParticipantRecord, OutcomeRecord | EventStatus, EventType, AttendanceStatus | Participation evidence should be recorded only when attendance or contribution is confirmed. |

| Challenge | Challenge | ProblemStatement, RuleSet, Submission, JudgingPanel, Award, ConversionRecord | ChallengeStatus, SubmissionStatus, AwardType | Challenges must define IP expectations, criteria, and conversion paths before submissions. |

| Community | Community | Membership, Moderator, Topic, Contribution, CommunityEvent | CommunityStatus, MembershipRole, ContributionType | Communities should track contribution but not replace formal evaluation contexts. |

| Ranking Cycle | RankingCycle | RankingCategory, EligibilityRule, ScoreComponent, EvidenceSnapshot, ResultEntry, Appeal | RankingPeriod, ScoreWeight, ScoreValue, AppealStatus | Criteria must be published before scoring; rankings use validated evidence and support appeals. |

| Compliance Case | ComplianceCase | Complainant, Respondent, EvidenceRecord, ReviewAction, Decision, Appeal | CaseStatus, SeverityLevel, ResolutionType | High-impact disputes require due process, audit, and scoped action. |

## 5.1 Representative Aggregate Diagrams

### Student Aggregate

```text
Student (Root)
├── StudentAffiliation
├── EnrollmentRecord
├── AcademicStanding
├── CareerGoal
├── ConsentPreference
└── Value Objects: StudentStatus, AcademicTerm, ProgramReference, InstitutionReference, ConsentScope
```

**Lifecycle Protection:** A student must have identity; verified affiliation requires institutional confirmation; graduation changes status but does not delete portfolio continuity.

### Portfolio Aggregate

```text
Portfolio (Root)
├── PortfolioItem
├── Achievement
├── Recommendation
├── Endorsement
├── ReadinessProfile
└── Value Objects: EvidenceReference, VisibilityLevel, PortfolioItemType, ReadinessScore
```

**Lifecycle Protection:** Verified items cannot be altered directly by the owner; private items need consent before external sharing; readiness score must be explainable.

### University Aggregate

```text
University (Root)
├── CollegeAffiliation
├── Department
├── Program
├── ResearchCenter
├── InnovationPolicy
└── Value Objects: UniversityType, AccreditationStatus, InstitutionAddress, AcademicCalendar
```

**Lifecycle Protection:** A university must be verified before granting official authority; colleges cannot claim affiliation without approval; delegation must be scoped.

### Project Aggregate

```text
Project (Root)
├── ProjectTeam
├── ProjectMember
├── Milestone
├── Submission
├── Evidence
└── Value Objects: ProjectStatus, ProjectCategory, ProblemStatement, ContributionShare, ConfidentialityLevel
```

**Lifecycle Protection:** A formal project requires owner, category, problem statement, evidence, and contribution rules; innovation transition requires ownership clarity.

### Research Aggregate

```text
ResearchWork (Root)
├── ResearchProposal
├── ResearchTeam
├── ResearchOutput
├── ReviewRecord
├── EthicsClearance
└── Value Objects: ResearchStatus, ResearchType, PublicationStatus, EthicsStatus, CitationReference
```

**Lifecycle Protection:** Research involving sensitive areas needs ethics clearance; publication claims need evidence; commercialization must respect IP and confidentiality.

### Publication Aggregate

```text
Publication (Root)
├── AuthorContribution
├── PublicationEvidence
├── VerificationRecord
├── CorrectionRecord
└── Value Objects: PublicationStatus, CitationReference, PeerReviewStatus
```

**Lifecycle Protection:** Published status requires evidence; corrections and retractions must affect portfolio and ranking evidence.

### Patent Aggregate

```text
PatentCase (Root)
├── InventorRecord
├── DisclosureRecord
├── FilingRecord
├── OwnershipClaim
├── LicenseRecord
└── Value Objects: IPStatus, PatentStage, ConfidentialityLevel
```

**Lifecycle Protection:** Inventors and authors are distinct; patent-sensitive material may require restricted visibility.

### Innovation Aggregate

```text
InnovationCandidate (Root)
├── InnovationClaim
├── SourceReference
├── ValidationEvidence
├── EvaluationPanel
├── ImpactPathway
└── Value Objects: InnovationStatus, NoveltyLevel, FeasibilityLevel, ImpactCategory, ReadinessLevel
```

**Lifecycle Protection:** Innovation cannot be validated without evidence and evaluator rationale; rejected candidates may resubmit with new evidence.

### Incubation Program Aggregate

```text
IncubationProgram (Root)
├── ProgramCycle
├── EligibilityCriteria
├── SelectionCriteria
├── MentorPool
├── ResourcePackage
└── Value Objects: ProgramStatus, ProgramDuration, CohortSize, EligibilityRule
```

**Lifecycle Protection:** Eligibility and selection criteria must be published before applications; graduation requires milestone review.

### Cohort Aggregate

```text
Cohort (Root)
├── CohortMember
├── AssignedMentor
├── ProgramMilestone
├── ProgressReview
├── GraduationDecision
└── Value Objects: CohortStatus, MilestoneStatus, ReviewDecision
```

**Lifecycle Protection:** A cohort needs selected members, program authority, milestones, and review cadence.

### Startup Aggregate

```text
Startup (Root)
├── FounderTeam
├── FounderRole
├── VentureProfile
├── BusinessModel
├── TractionRecord
└── Value Objects: StartupStage, FundingStage, FounderCommitmentLevel, BusinessModelType
```

**Lifecycle Protection:** A startup needs at least one accountable founder; stage claims require evidence; startup from project/research must record IP assumptions.

### Funding Request Aggregate

```text
FundingRequest (Root)
├── Applicant
├── FundingNeed
├── UseOfFunds
├── EvidencePackage
├── DueDiligenceRecord
└── Value Objects: FundingType, Money, DecisionStatus, RiskRating
```

**Lifecycle Protection:** Funding request must state type, amount, purpose, applicant, and evidence; approval may be conditional.

# 6. Entities

Entities have identity and lifecycle. The same entity can change over time while remaining the same business object.

| Entity | Context | Purpose | Relationships | Lifecycle |
|---|---|---|---|---|

| Student | Learning / Portfolio | Learner with academic affiliation and lifelong participation. | University, portfolio, projects, learning tracks, communities. | Registered, verified, active, graduated, alumni, inactive. |

| Faculty | Institution / Research | Academic actor who may teach, supervise, research, mentor, evaluate, or govern. | Department, projects, research, students, incubation. | Active, delegated, retired, external, suspended. |

| University | Institution | Academic authority and ecosystem participant. | Colleges, departments, programs, centers, students. | Registered, verified, active, suspended, merged, archived. |

| Project | Project | Structured work artifact. | Team, evidence, reviews, portfolio, innovation. | Draft, proposed, approved, active, reviewed, completed, archived. |

| ResearchWork | Research | Research initiative or output. | Researchers, publication, patent, grant, innovation. | Proposed, approved, active, submitted, published, commercialized. |

| InnovationCandidate | Innovation | Artifact under innovation validation. | Source artifact, evaluators, incubation, funding. | Submitted, screening, evaluated, validated, rejected, recommended. |

| Startup | Startup | Venture with founder accountability. | Founders, incubation, funding, recruitment, investors. | Concept, validation, prototype, MVP, pilot, revenue, growth, closed. |

| FundingRequest | Funding | Request for grant, investment, sponsorship, or capital. | Startup, project, research, investor, government. | Draft, submitted, review, approved, rejected, committed, disbursed. |

| Opportunity | Recruitment / Marketplace | Job, internship, grant, challenge, partnership, or program opening. | Sponsor, applicants, portfolio, event, funding. | Draft, published, active, closed, archived. |

| MentorshipEngagement | Mentorship | Structured advisory relationship. | Mentor, mentee, project, startup, cohort. | Requested, matched, active, completed, closed. |

| RankingCycle | Ranking | Scoring and recognition period. | Evidence snapshot, results, appeals. | Configured, open, calculated, published, appealed, locked. |

| ComplianceCase | Compliance | Dispute, violation, misuse, fraud, or ownership case. | Actors, evidence, decisions, affected contexts. | Open, review, resolved, appealed, closed. |

# 7. Value Objects

Value objects are immutable business concepts identified by their attributes rather than identity.

## 7.1 Identity and Institution Value Objects

| Value Object | Business Meaning |
|---|---|

| PersonName | Immutable personname used consistently inside its bounded context. |

| ContactInformation | Immutable contactinformation used consistently inside its bounded context. |

| Address | Immutable address used consistently inside its bounded context. |

| InstitutionReference | Immutable institution reference used consistently inside its bounded context. |

| ProgramReference | Immutable program reference used consistently inside its bounded context. |

| AcademicTerm | Immutable academicterm used consistently inside its bounded context. |

| AffiliationStatus | Immutable affiliation status used consistently inside its bounded context. |

| RoleAssignment | Immutable roleassignment used consistently inside its bounded context. |

| PermissionScope | Immutable permission scope used consistently inside its bounded context. |

| ConsentScope | Immutable consent scope used consistently inside its bounded context. |

| DelegationScope | Immutable delegation scope used consistently inside its bounded context. |

## 7.2 Learning and Competency Value Objects

| Value Object | Business Meaning |
|---|---|

| SkillLevel | Immutable skill level used consistently inside its bounded context. |

| CompetencyLevel | Immutable competency level used consistently inside its bounded context. |

| AssessmentScore | Immutable assessment score used consistently inside its bounded context. |

| SkillEvidenceType | Immutable skillevidence type used consistently inside its bounded context. |

| BadgeStatus | Immutable badge status used consistently inside its bounded context. |

| RoadmapProgress | Immutable roadmapprogress used consistently inside its bounded context. |

| CareerAlignmentScore | Immutable careeralignment score used consistently inside its bounded context. |

| LearningGoal | Immutable learninggoal used consistently inside its bounded context. |

| EvidenceStrength | Immutable evidencestrength used consistently inside its bounded context. |

## 7.3 Project and Research Value Objects

| Value Object | Business Meaning |
|---|---|

| ProjectStatus | Immutable project status used consistently inside its bounded context. |

| ProjectCategory | Immutable projectcategory used consistently inside its bounded context. |

| ProblemStatement | Immutable problemstatement used consistently inside its bounded context. |

| ContributionShare | Immutable contributionshare used consistently inside its bounded context. |

| ReviewDecision | Immutable reviewdecision used consistently inside its bounded context. |

| ResearchStatus | Immutable research status used consistently inside its bounded context. |

| EthicsStatus | Immutable ethics status used consistently inside its bounded context. |

| PublicationStatus | Immutable publication status used consistently inside its bounded context. |

| CitationReference | Immutable citation reference used consistently inside its bounded context. |

| IPStatus | Immutable ip status used consistently inside its bounded context. |

## 7.4 Innovation and Startup Value Objects

| Value Object | Business Meaning |
|---|---|

| InnovationStatus | Immutable innovation status used consistently inside its bounded context. |

| NoveltyLevel | Immutable novelty level used consistently inside its bounded context. |

| FeasibilityLevel | Immutable feasibility level used consistently inside its bounded context. |

| ImpactCategory | Immutable impactcategory used consistently inside its bounded context. |

| TechnologyReadinessLevel | Immutable technologyreadiness level used consistently inside its bounded context. |

| MarketReadinessLevel | Immutable marketreadiness level used consistently inside its bounded context. |

| StartupStage | Immutable startup stage used consistently inside its bounded context. |

| FounderCommitmentLevel | Immutable foundercommitment level used consistently inside its bounded context. |

| TractionMetric | Immutable tractionmetric used consistently inside its bounded context. |

| BusinessModelType | Immutable businessmodel type used consistently inside its bounded context. |

## 7.5 Funding and Investment Value Objects

| Value Object | Business Meaning |
|---|---|

| Money | Immutable money used consistently inside its bounded context. |

| FundingType | Immutable funding type used consistently inside its bounded context. |

| FundingStage | Immutable funding stage used consistently inside its bounded context. |

| UseOfFunds | Immutable useoffunds used consistently inside its bounded context. |

| RiskRating | Immutable riskrating used consistently inside its bounded context. |

| InvestorInterestLevel | Immutable investorinterest level used consistently inside its bounded context. |

| DecisionStatus | Immutable decision status used consistently inside its bounded context. |

| FundingCondition | Immutable funding condition used consistently inside its bounded context. |

| CommitmentStatus | Immutable commitment status used consistently inside its bounded context. |

## 7.6 Ranking and Governance Value Objects

| Value Object | Business Meaning |
|---|---|

| InnovationScore | Immutable innovation score used consistently inside its bounded context. |

| PortfolioScore | Immutable portfolio score used consistently inside its bounded context. |

| ImpactScore | Immutable impact score used consistently inside its bounded context. |

| ScoreComponent | Immutable  scorecomponent used consistently inside its bounded context. |

| ScoreWeight | Immutable  scoreweight used consistently inside its bounded context. |

| RankingPeriod | Immutable rankingperiod used consistently inside its bounded context. |

| RankPosition | Immutable rankposition used consistently inside its bounded context. |

| AppealStatus | Immutable appeal status used consistently inside its bounded context. |

| PolicyReference | Immutable policy reference used consistently inside its bounded context. |

| ComplianceStatus | Immutable compliance status used consistently inside its bounded context. |

| ConfidentialityLevel | Immutable confidentiality level used consistently inside its bounded context. |

| DataOwnershipClaim | Immutable dataownershipclaim used consistently inside its bounded context. |

| AuditEntry | Immutable auditentry used consistently inside its bounded context. |

| ConflictStatus | Immutable conflict status used consistently inside its bounded context. |

| DisputeStatus | Immutable dispute status used consistently inside its bounded context. |

# 8. Business Rules

Business rules are the domain constraints that protect trust, fairness, quality, legal clarity, and institutional accountability. They should be validated in workshops before SRS finalization.

## 8.1 Identity and Eligibility Rules

| ID | Business Rule |
|---|---|

| BR-IA-001 | Every business action must be attributable to a known actor. |

| BR-IA-002 | A user may hold multiple roles, but every role must have a defined context and scope. |

| BR-IA-003 | Institutional authority must be verified before official academic records can be approved. |

| BR-IA-004 | Official student status requires institutional confirmation, not only self-registration. |

| BR-IA-005 | Alumni status must be derived from verified academic history. |

| BR-IA-006 | Mentor profiles may be created before verification, but official program assignment requires approval. |

| BR-IA-007 | Investor status must be verified before confidential fundraising materials are visible. |

| BR-IA-008 | Recruiter status must be tied to a verified company or authorized recruitment body. |

| BR-IA-009 | Government dashboard access requires verified government role and scope. |

| BR-IA-010 | Delegated authority must include delegator, delegatee, scope, duration, and revocation rules. |

| BR-IA-011 | Expired or revoked roles cannot be used for approvals. |

| BR-IA-012 | Suspended actors cannot perform privileged actions until reinstated. |

| BR-IA-013 | Identity verification and business authority verification are separate. |

| BR-IA-014 | Role conflicts must be declared in evaluation, funding, recruitment, ranking, and mentorship. |

| BR-IA-015 | Users must control personal visibility except where limited institutional or legal reporting is required. |

## 8.2 Ownership and Intellectual Property Rules

| ID | Business Rule |
|---|---|

| BR-OA-001 | Every formal project must declare ownership expectations before final submission. |

| BR-OA-002 | Project participation does not automatically grant startup founder status. |

| BR-OA-003 | University ownership claims must follow published policy and applicable law. |

| BR-OA-004 | Sponsor rights must be declared before sponsored project work begins. |

| BR-OA-005 | Confidential project work cannot be made public without authorized consent. |

| BR-OA-006 | Open-source release requires consent from rights holders. |

| BR-OA-007 | Research linked to patent disclosure may require restricted visibility until filing decisions are made. |

| BR-OA-008 | Faculty supervision does not automatically transfer ownership to faculty. |

| BR-OA-009 | Team members must acknowledge contribution claims before startup, funding, or ranking use. |

| BR-OA-010 | Startup creation from a student project must record founder consent, ownership status, and IP assumptions. |

| BR-OA-011 | Marketplace listings must not expose proprietary information without permission. |

| BR-OA-012 | Patent inventorship must be separated from authorship, supervision, and general contribution. |

| BR-OA-013 | Research or sponsored data must have access and usage rules. |

| BR-OA-014 | Ownership disputes should freeze high-stakes external sharing until resolved or clearly marked disputed. |

| BR-OA-015 | Alumni retain personal portfolio history subject to confidentiality and policy. |

## 8.3 Learning and Competency Rules

| ID | Business Rule |
|---|---|

| BR-LA-001 | A skill badge must be supported by evidence. |

| BR-LA-002 | Course completion alone does not automatically prove practical competency. |

| BR-LA-003 | Practical competency should require applied evidence. |

| BR-LA-004 | A learning track must define target competencies and acceptable evidence types. |

| BR-LA-005 | Gap analysis must compare evidence against declared goals, not against one universal ideal. |

| BR-LA-006 | Students may pursue multiple learning tracks. |

| BR-LA-007 | Skill level changes must be traceable to evidence or assessment. |

| BR-LA-008 | External certificates may support evidence but should not automatically equal mastery. |

| BR-LA-009 | Faculty and mentors may recommend learning actions but cannot alter skill scores without evidence. |

| BR-LA-010 | Recommendations should consider career goal, evidence, available opportunities, and learner constraints. |

| BR-LA-011 | Time-sensitive skills may expire or require renewal. |

| BR-LA-012 | Soft skills require observable evidence such as teamwork feedback, leadership roles, or mentor observations. |

| BR-LA-013 | Career readiness must not be calculated only from grades. |

| BR-LA-014 | Students may challenge incorrect competency records. |

| BR-LA-015 | Learning progress should feed portfolio at evidence checkpoints rather than every minor activity. |

## 8.4 Project Rules

| ID | Business Rule |
|---|---|

| BR-P-001 | A project must have a problem statement before formal approval. |

| BR-P-002 | A project must identify category: academic, personal, industry, research, open-source, social, or startup-oriented. |

| BR-P-003 | Formal academic projects require a supervisor or authorized reviewer. |

| BR-P-004 | A project team must define member roles. |

| BR-P-005 | Contribution claims must be visible to team members before confirmation. |

| BR-P-006 | A project cannot be marked completed without required evidence. |

| BR-P-007 | A final-year project becomes an innovation candidate only after innovation eligibility checks. |

| BR-P-008 | Projects with sensitive data must define confidentiality level. |

| BR-P-009 | A project may have multiple outcomes: academic completion, portfolio evidence, research, incubation, open source, startup, or archive. |

| BR-P-010 | Rejected proposals may be revised and resubmitted. |

| BR-P-011 | Milestones must fit project type. |

| BR-P-012 | Academic approval and innovation validation are separate decisions. |

| BR-P-013 | Company-sponsored projects must identify sponsor expectations. |

| BR-P-014 | Reviews should include rationale, not only scores. |

| BR-P-015 | Project evidence may include prototype, code, design, dataset, demo, user research, pilot, report, or presentation. |

| BR-P-016 | Deployment claims require evidence. |

| BR-P-017 | Challenge-linked projects should preserve challenge context. |

| BR-P-018 | Abandoned projects should be archived unless policy permits deletion. |

| BR-P-019 | Projects should support continuation after graduation when ownership allows. |

| BR-P-020 | Plagiarism or fraudulent evidence triggers compliance review. |

## 8.5 Research Rules

| ID | Business Rule |
|---|---|

| BR-R-001 | Research proposals must define objective, method, expected output, and responsible researchers. |

| BR-R-002 | Research requiring ethics approval cannot be approved until clearance exists. |

| BR-R-003 | Publication status must be supported by evidence. |

| BR-R-004 | Citation count should not be the only research impact measure. |

| BR-R-005 | Patent filing must be tracked separately from publication. |

| BR-R-006 | Commercialization requires IP, market, feasibility, ethical, and regulatory assessment. |

| BR-R-007 | Faculty cannot independently validate their own publication for ranking without external evidence. |

| BR-R-008 | Student research contribution must be credited accurately. |

| BR-R-009 | Grant-funded research must track reporting obligations. |

| BR-R-010 | Research data visibility must respect consent, confidentiality, sponsor terms, and ethics. |

| BR-R-011 | Retractions or corrections must update portfolio and ranking evidence. |

| BR-R-012 | Research becomes innovation candidate only when novelty, utility, and application potential are documented. |

| BR-R-013 | Research centers can validate outputs only within defined authority. |

| BR-R-014 | Collaboration agreements should define authorship, data rights, and commercialization expectations. |

| BR-R-015 | Traditional knowledge and community-based research require benefit-sharing and ethical safeguards. |

## 8.6 Innovation Rules

| ID | Business Rule |
|---|---|

| BR-I-001 | Innovation must be evaluated against novelty, usefulness, feasibility, evidence, and impact. |

| BR-I-002 | An idea alone is not validated innovation. |

| BR-I-003 | A good academic grade is not automatically innovation. |

| BR-I-004 | Innovation candidates must identify source artifacts. |

| BR-I-005 | Evaluation requires conflict-of-interest declaration. |

| BR-I-006 | Innovation status must be explainable. |

| BR-I-007 | Innovation may lead to research, incubation, grant, patent, open source, industry pilot, community deployment, or archive. |

| BR-I-008 | Social innovation must not be penalized for lacking immediate commercial revenue. |

| BR-I-009 | Technical innovation must provide feasibility evidence appropriate to maturity. |

| BR-I-010 | Market innovation must include user, customer, or stakeholder validation. |

| BR-I-011 | Innovation score evolves as evidence improves. |

| BR-I-012 | Rejected candidates may resubmit with significant new evidence. |

| BR-I-013 | Confidential innovation evaluation must protect sensitive details. |

| BR-I-014 | Regulated-sector innovations must identify regulatory requirements. |

| BR-I-015 | Innovation should be compared within relevant categories, not blindly across disciplines. |

## 8.7 Incubation Rules

| ID | Business Rule |
|---|---|

| BR-I-001 | Programs must publish eligibility criteria before accepting applications. |

| BR-I-002 | Selection criteria must be defined before evaluation begins. |

| BR-I-003 | A cohort cannot start without program authority. |

| BR-I-004 | Every incubated team must have milestones. |

| BR-I-005 | Mentor assignment must consider expertise, availability, conflict, and team need. |

| BR-I-006 | Resources must be allocated according to program policy. |

| BR-I-007 | Graduation requires milestone review, not attendance only. |

| BR-I-008 | Incubation participation does not automatically imply investment readiness. |

| BR-I-009 | Incubation centers must track outcomes after completion. |

| BR-I-010 | Teams missing milestones should receive support, revision, or exit decision based on policy. |

| BR-I-011 | University incubators must clarify relationship between academic projects and venture ownership. |

| BR-I-012 | Programs may support non-student founders if policy allows. |

| BR-I-013 | Selection must record evaluator decisions to reduce nepotism. |

| BR-I-014 | Demo day participation should require readiness review. |

| BR-I-015 | Program completion should update startup and founder portfolio evidence. |

## 8.8 Startup Rules

| ID | Business Rule |
|---|---|

| BR-S-001 | A startup must have at least one accountable founder. |

| BR-S-002 | Founder roles must be explicitly declared. |

| BR-S-003 | Student project team membership does not automatically equal founder status. |

| BR-S-004 | Startups derived from university work must identify IP status. |

| BR-S-005 | Startup stage must be evidence-based. |

| BR-S-006 | Traction claims require evidence. |

| BR-S-007 | Startup profiles should have public and confidential sections. |

| BR-S-008 | Founder departure must be recorded. |

| BR-S-009 | Startup closure should preserve founder learning evidence. |

| BR-S-010 | Startups may be commercial, social enterprise, cooperative, research spinout, or impact venture. |

| BR-S-011 | Funding readiness requires a minimum evidence package. |

| BR-S-012 | Incubation graduation updates readiness but does not guarantee success. |

| BR-S-013 | False revenue, user, legal, or funding claims trigger compliance review. |

| BR-S-014 | Founder commitment level should be declared and updated. |

| BR-S-015 | Startup profiles should identify sector, region, impact category, and stage. |

## 8.9 Funding and Investment Rules

| ID | Business Rule |
|---|---|

| BR-FA-001 | A funding request must state amount, type, purpose, and applicant. |

| BR-FA-002 | Grant and investment requests must be distinguished. |

| BR-FA-003 | Investment requests require startup or proposed venture identity. |

| BR-FA-004 | Grant requests must satisfy sponsor eligibility. |

| BR-FA-005 | Due diligence must include evidence review. |

| BR-FA-006 | Investor access to confidential documents requires permission. |

| BR-FA-007 | Funding approval may be conditional. |

| BR-FA-008 | Disbursement conditions must be explicit. |

| BR-FA-009 | Funding decisions must record rationale. |

| BR-FA-010 | Rejected requests may be revised and resubmitted if policy allows. |

| BR-FA-011 | Investors must declare conflicts when evaluating startups. |

| BR-FA-012 | Public grants must follow reporting requirements. |

| BR-FA-013 | Funding outcomes should update startup and founder portfolio records. |

| BR-FA-014 | Investor interest does not equal commitment. |

| BR-FA-015 | Metrics must distinguish requested, approved, committed, and disbursed amounts. |

## 8.10 Mentorship Rules

| ID | Business Rule |
|---|---|

| BR-M-001 | A mentor must have declared or verified expertise. |

| BR-M-002 | Mentorship engagement must define purpose and scope. |

| BR-M-003 | Mentors must declare conflicts of interest. |

| BR-M-004 | Mentor feedback follows confidentiality rules. |

| BR-M-005 | Mentors cannot be forced into unlimited support beyond agreement. |

| BR-M-006 | Mentees should provide feedback on mentorship quality. |

| BR-M-007 | Mentor contribution counts toward recognition only when evidence exists. |

| BR-M-008 | A mentor advising a startup should not independently decide its funding approval. |

| BR-M-009 | Confidential mentorship notes should not be exposed through portfolio. |

| BR-M-010 | Repeated no-shows may affect reputation or assignment priority. |

## 8.11 Recruitment Rules

| ID | Business Rule |
|---|---|

| BR-R-001 | Recruiters must represent verified companies or authorized bodies. |

| BR-R-002 | Opportunities must describe role, eligibility, expectations, and selection process. |

| BR-R-003 | Shortlisting should use consented and relevant portfolio evidence. |

| BR-R-004 | Grades should not be the only hiring filter unless role requires it. |

| BR-R-005 | Companies may search public portfolios but need permission for private evidence. |

| BR-R-006 | Internship completion should produce verified evidence and feedback. |

| BR-R-007 | Recruiter evaluation should not manipulate public ranking without validation. |

| BR-R-008 | Listings must be authentic and non-exploitative. |

| BR-R-009 | Application history is private by default. |

| BR-R-010 | Employment outcomes require appropriate confirmation and consent. |

## 8.12 Ranking and Recognition Rules

| ID | Business Rule |
|---|---|

| BR-RA-001 | Criteria must be published before the ranking period. |

| BR-RA-002 | Ranking must use validated evidence only. |

| BR-RA-003 | Rankings must be contextual by category, level, discipline, stage, and period. |

| BR-RA-004 | Rankings should reward consistency, collaboration, impact, quality, and verified outcomes. |

| BR-RA-005 | Rankings must not reward spam activity. |

| BR-RA-006 | Results must explain major score components. |

| BR-RA-007 | Participants may appeal incorrect evidence or calculation. |

| BR-RA-008 | Self-issued claims carry lower weight than independent validation. |

| BR-RA-009 | Mentor, faculty, and institutional contribution should be recognized separately from student output. |

| BR-RA-010 | Fraud, plagiarism, or retractions may affect reputation after due process. |

| BR-RA-011 | Ranking should avoid permanent punishment for early failure when recovery occurs. |

| BR-RA-012 | Popularity indicators may only be weak signals unless tied to impact evidence. |

| BR-RA-013 | University rankings should reward ecosystem contribution, not user count alone. |

| BR-RA-014 | Faculty recognition should include mentorship, research impact, startup support, and student outcomes. |

| BR-RA-015 | Startup ranking should account for stage and sector. |

## 8.13 Government and Reporting Rules

| ID | Business Rule |
|---|---|

| BR-GA-001 | Government dashboards should use aggregate indicators unless individual disclosure is authorized. |

| BR-GA-002 | Public grants must define eligibility, selection criteria, conditions, and reporting obligations. |

| BR-GA-003 | Municipality programs may define local priorities while aligning with national language. |

| BR-GA-004 | Government reporting must not expose confidential startup or research information without authority. |

| BR-GA-005 | National indicators should distinguish activity, output, outcome, and impact. |

| BR-GA-006 | Government officers cannot alter institutional records unless legally authorized and audited. |

| BR-GA-007 | Public funding results may be reportable while protecting due diligence details. |

| BR-GA-008 | Policy changes should be versioned and not retroactively invalidate completed processes unless legally required. |

## 8.14 Marketplace and Partnership Rules

| ID | Business Rule |
|---|---|

| BR-MA-001 | Listings must identify owner and opportunity type. |

| BR-MA-002 | Confidential listings must restrict visibility. |

| BR-MA-003 | Partnership proposals must define objective, parties, contribution, and intended outcome. |

| BR-MA-004 | Industry challenges must define problem context, eligibility, criteria, and IP expectations. |

| BR-MA-005 | Sponsored opportunities must identify sponsor obligations. |

| BR-MA-006 | Spam, fake opportunities, and predatory offers trigger moderation. |

| BR-MA-007 | Accepted partnerships create traceable collaboration records. |

| BR-MA-008 | Listings that lead to projects, internships, funding, or incubation should link to the relevant context. |

## 8.15 Compliance and Integrity Rules

| ID | Business Rule |
|---|---|

| BR-CA-001 | Important decisions must be auditable. |

| BR-CA-002 | Actors can dispute incorrect claims about them or their work. |

| BR-CA-003 | Fraudulent evidence triggers investigation before penalty. |

| BR-CA-004 | Plagiarism claims should be reviewed by authorized evaluators. |

| BR-CA-005 | Conflicts of interest must be recorded and resolved according to policy. |

| BR-CA-006 | Data ownership disputes should restrict high-stakes external sharing until resolved. |

| BR-CA-007 | Appeals need defined timeline and authority. |

| BR-CA-008 | System errors affecting scores or decisions must be correctable and visible to affected users. |

| BR-CA-009 | Sanctions should be proportional to severity and evidence. |

| BR-CA-010 | Sensitive data should be shared only under authorized scope. |

# 9. Domain Events

Domain events are business facts that have already happened. They should be named in past tense and should be meaningful to domain experts.

## 9.1 Identity and Institution Events

| Event | Producer Context | Likely Consumers | Meaning |
|---|---|---|---|

| UserRegistered | Identity and Institution | Portfolio, Analytics, Ranking, related workflow contexts | User Registered occurred and may trigger follow-up business reactions. |

| IdentityVerified | Identity and Institution | Portfolio, Analytics, Ranking, related workflow contexts | Identity Verified occurred and may trigger follow-up business reactions. |

| OrganizationRegistered | Identity and Institution | Portfolio, Analytics, Ranking, related workflow contexts | Organization Registered occurred and may trigger follow-up business reactions. |

| InstitutionVerified | Identity and Institution | Portfolio, Analytics, Ranking, related workflow contexts | Institution Verified occurred and may trigger follow-up business reactions. |

| StudentAffiliationRequested | Identity and Institution | Portfolio, Analytics, Ranking, related workflow contexts | Student Affiliation Requested occurred and may trigger follow-up business reactions. |

| StudentAffiliationVerified | Identity and Institution | Portfolio, Analytics, Ranking, related workflow contexts | Student Affiliation Verified occurred and may trigger follow-up business reactions. |

| FacultyRoleAssigned | Identity and Institution | Portfolio, Analytics, Ranking, related workflow contexts | Faculty Role Assigned occurred and may trigger follow-up business reactions. |

| ProgramRegistered | Identity and Institution | Portfolio, Analytics, Ranking, related workflow contexts | Program Registered occurred and may trigger follow-up business reactions. |

| RoleDelegated | Identity and Institution | Portfolio, Analytics, Ranking, related workflow contexts | Role Delegated occurred and may trigger follow-up business reactions. |

| RoleRevoked | Identity and Institution | Portfolio, Analytics, Ranking, related workflow contexts | Role Revoked occurred and may trigger follow-up business reactions. |

## 9.2 Learning and Portfolio Events

| Event | Producer Context | Likely Consumers | Meaning |
|---|---|---|---|

| LearningGoalDeclared | Learning and Portfolio | Portfolio, Analytics, Ranking, related workflow contexts | Learning Goal Declared occurred and may trigger follow-up business reactions. |

| LearningTrackStarted | Learning and Portfolio | Portfolio, Analytics, Ranking, related workflow contexts | Learning Track Started occurred and may trigger follow-up business reactions. |

| SkillEvidenceSubmitted | Learning and Portfolio | Portfolio, Analytics, Ranking, related workflow contexts | Skill Evidence Submitted occurred and may trigger follow-up business reactions. |

| SkillEvidenceVerified | Learning and Portfolio | Portfolio, Analytics, Ranking, related workflow contexts | Skill Evidence Verified occurred and may trigger follow-up business reactions. |

| SkillBadgeIssued | Learning and Portfolio | Portfolio, Analytics, Ranking, related workflow contexts | Skill Badge Issued occurred and may trigger follow-up business reactions. |

| CompetencyLevelUpdated | Learning and Portfolio | Portfolio, Analytics, Ranking, related workflow contexts | Competency Level Updated occurred and may trigger follow-up business reactions. |

| GapAnalysisCompleted | Learning and Portfolio | Portfolio, Analytics, Ranking, related workflow contexts | Gap Analysis Completed occurred and may trigger follow-up business reactions. |

| PortfolioCreated | Learning and Portfolio | Portfolio, Analytics, Ranking, related workflow contexts | Portfolio Created occurred and may trigger follow-up business reactions. |

| PortfolioItemAdded | Learning and Portfolio | Portfolio, Analytics, Ranking, related workflow contexts | Portfolio Item Added occurred and may trigger follow-up business reactions. |

| RecommendationAdded | Learning and Portfolio | Portfolio, Analytics, Ranking, related workflow contexts | Recommendation Added occurred and may trigger follow-up business reactions. |

## 9.3 Project Events

| Event | Producer Context | Likely Consumers | Meaning |
|---|---|---|---|

| ProjectCreated | Project | Portfolio, Analytics, Ranking, related workflow contexts | Project Created occurred and may trigger follow-up business reactions. |

| ProjectTeamFormed | Project | Portfolio, Analytics, Ranking, related workflow contexts | Project Team Formed occurred and may trigger follow-up business reactions. |

| ProjectProposalSubmitted | Project | Portfolio, Analytics, Ranking, related workflow contexts | Project Proposal Submitted occurred and may trigger follow-up business reactions. |

| ProjectApproved | Project | Portfolio, Analytics, Ranking, related workflow contexts | Project Approved occurred and may trigger follow-up business reactions. |

| ProjectRejected | Project | Portfolio, Analytics, Ranking, related workflow contexts | Project Rejected occurred and may trigger follow-up business reactions. |

| ProjectMilestoneCompleted | Project | Portfolio, Analytics, Ranking, related workflow contexts | Project Milestone Completed occurred and may trigger follow-up business reactions. |

| ProjectEvidenceSubmitted | Project | Portfolio, Analytics, Ranking, related workflow contexts | Project Evidence Submitted occurred and may trigger follow-up business reactions. |

| ProjectReviewed | Project | Portfolio, Analytics, Ranking, related workflow contexts | Project Reviewed occurred and may trigger follow-up business reactions. |

| ProjectCompleted | Project | Portfolio, Analytics, Ranking, related workflow contexts | Project Completed occurred and may trigger follow-up business reactions. |

| ProjectMarkedInnovationCandidate | Project | Portfolio, Analytics, Ranking, related workflow contexts | Project Marked Innovation Candidate occurred and may trigger follow-up business reactions. |

| ProjectOwnershipDisputed | Project | Portfolio, Analytics, Ranking, related workflow contexts | Project Ownership Disputed occurred and may trigger follow-up business reactions. |

## 9.4 Research Events

| Event | Producer Context | Likely Consumers | Meaning |
|---|---|---|---|

| ResearchProposalSubmitted | Research | Portfolio, Analytics, Ranking, related workflow contexts | Research Proposal Submitted occurred and may trigger follow-up business reactions. |

| ResearchApproved | Research | Portfolio, Analytics, Ranking, related workflow contexts | Research Approved occurred and may trigger follow-up business reactions. |

| EthicsApprovalGranted | Research | Portfolio, Analytics, Ranking, related workflow contexts | Ethics Approval Granted occurred and may trigger follow-up business reactions. |

| ResearchOutputSubmitted | Research | Portfolio, Analytics, Ranking, related workflow contexts | Research Output Submitted occurred and may trigger follow-up business reactions. |

| ResearchReviewed | Research | Portfolio, Analytics, Ranking, related workflow contexts | Research Reviewed occurred and may trigger follow-up business reactions. |

| PublicationSubmitted | Research | Portfolio, Analytics, Ranking, related workflow contexts | Publication Submitted occurred and may trigger follow-up business reactions. |

| PublicationAccepted | Research | Portfolio, Analytics, Ranking, related workflow contexts | Publication Accepted occurred and may trigger follow-up business reactions. |

| PublicationPublished | Research | Portfolio, Analytics, Ranking, related workflow contexts | Publication Published occurred and may trigger follow-up business reactions. |

| PublicationRetracted | Research | Portfolio, Analytics, Ranking, related workflow contexts | Publication Retracted occurred and may trigger follow-up business reactions. |

| PatentDisclosureSubmitted | Research | Portfolio, Analytics, Ranking, related workflow contexts | Patent Disclosure Submitted occurred and may trigger follow-up business reactions. |

| PatentFiled | Research | Portfolio, Analytics, Ranking, related workflow contexts | Patent Filed occurred and may trigger follow-up business reactions. |

| ResearchCommercializationRecommended | Research | Portfolio, Analytics, Ranking, related workflow contexts | Research Commercialization Recommended occurred and may trigger follow-up business reactions. |

## 9.5 Innovation Events

| Event | Producer Context | Likely Consumers | Meaning |
|---|---|---|---|

| InnovationCandidateSubmitted | Innovation | Portfolio, Analytics, Ranking, related workflow contexts | Innovation Candidate Submitted occurred and may trigger follow-up business reactions. |

| InnovationScreeningStarted | Innovation | Portfolio, Analytics, Ranking, related workflow contexts | Innovation Screening Started occurred and may trigger follow-up business reactions. |

| InnovationEvaluationRequested | Innovation | Portfolio, Analytics, Ranking, related workflow contexts | Innovation Evaluation Requested occurred and may trigger follow-up business reactions. |

| InnovationEvaluationCompleted | Innovation | Portfolio, Analytics, Ranking, related workflow contexts | Innovation Evaluation Completed occurred and may trigger follow-up business reactions. |

| InnovationValidated | Innovation | Portfolio, Analytics, Ranking, related workflow contexts | Innovation Validated occurred and may trigger follow-up business reactions. |

| InnovationRejected | Innovation | Portfolio, Analytics, Ranking, related workflow contexts | Innovation Rejected occurred and may trigger follow-up business reactions. |

| InnovationRevisionRequested | Innovation | Portfolio, Analytics, Ranking, related workflow contexts | Innovation Revision Requested occurred and may trigger follow-up business reactions. |

| IncubationRecommended | Innovation | Portfolio, Analytics, Ranking, related workflow contexts | Incubation Recommended occurred and may trigger follow-up business reactions. |

| PatentPathRecommended | Innovation | Portfolio, Analytics, Ranking, related workflow contexts | Patent Path Recommended occurred and may trigger follow-up business reactions. |

| IndustryPilotRecommended | Innovation | Portfolio, Analytics, Ranking, related workflow contexts | Industry Pilot Recommended occurred and may trigger follow-up business reactions. |

| InnovationArchived | Innovation | Portfolio, Analytics, Ranking, related workflow contexts | Innovation Archived occurred and may trigger follow-up business reactions. |

## 9.6 Incubation and Startup Events

| Event | Producer Context | Likely Consumers | Meaning |
|---|---|---|---|

| IncubationProgramCreated | Incubation and Startup | Portfolio, Analytics, Ranking, related workflow contexts | Incubation Program Created occurred and may trigger follow-up business reactions. |

| IncubationApplicationOpened | Incubation and Startup | Portfolio, Analytics, Ranking, related workflow contexts | Incubation Application Opened occurred and may trigger follow-up business reactions. |

| IncubationApplicationSubmitted | Incubation and Startup | Portfolio, Analytics, Ranking, related workflow contexts | Incubation Application Submitted occurred and may trigger follow-up business reactions. |

| IncubationApplicationAccepted | Incubation and Startup | Portfolio, Analytics, Ranking, related workflow contexts | Incubation Application Accepted occurred and may trigger follow-up business reactions. |

| CohortFormed | Incubation and Startup | Portfolio, Analytics, Ranking, related workflow contexts | Cohort Formed occurred and may trigger follow-up business reactions. |

| MentorAssignedToCohort | Incubation and Startup | Portfolio, Analytics, Ranking, related workflow contexts | Mentor Assigned To Cohort occurred and may trigger follow-up business reactions. |

| IncubationStarted | Incubation and Startup | Portfolio, Analytics, Ranking, related workflow contexts | Incubation Started occurred and may trigger follow-up business reactions. |

| IncubationMilestoneCompleted | Incubation and Startup | Portfolio, Analytics, Ranking, related workflow contexts | Incubation Milestone Completed occurred and may trigger follow-up business reactions. |

| IncubationCompleted | Incubation and Startup | Portfolio, Analytics, Ranking, related workflow contexts | Incubation Completed occurred and may trigger follow-up business reactions. |

| StartupCreated | Incubation and Startup | Portfolio, Analytics, Ranking, related workflow contexts | Startup Created occurred and may trigger follow-up business reactions. |

| FounderAdded | Incubation and Startup | Portfolio, Analytics, Ranking, related workflow contexts | Founder Added occurred and may trigger follow-up business reactions. |

| FounderRoleChanged | Incubation and Startup | Portfolio, Analytics, Ranking, related workflow contexts | Founder Role Changed occurred and may trigger follow-up business reactions. |

| StartupStageUpdated | Incubation and Startup | Portfolio, Analytics, Ranking, related workflow contexts | Startup Stage Updated occurred and may trigger follow-up business reactions. |

| TractionEvidenceAdded | Incubation and Startup | Portfolio, Analytics, Ranking, related workflow contexts | Traction Evidence Added occurred and may trigger follow-up business reactions. |

| StartupGraduated | Incubation and Startup | Portfolio, Analytics, Ranking, related workflow contexts | Startup Graduated occurred and may trigger follow-up business reactions. |

| StartupClosed | Incubation and Startup | Portfolio, Analytics, Ranking, related workflow contexts | Startup Closed occurred and may trigger follow-up business reactions. |

## 9.7 Funding and Investment Events

| Event | Producer Context | Likely Consumers | Meaning |
|---|---|---|---|

| FundingRequestCreated | Funding and Investment | Portfolio, Analytics, Ranking, related workflow contexts | Funding Request Created occurred and may trigger follow-up business reactions. |

| FundingRequestSubmitted | Funding and Investment | Portfolio, Analytics, Ranking, related workflow contexts | Funding Request Submitted occurred and may trigger follow-up business reactions. |

| InvestorInterestRegistered | Funding and Investment | Portfolio, Analytics, Ranking, related workflow contexts | Investor Interest Registered occurred and may trigger follow-up business reactions. |

| DueDiligenceStarted | Funding and Investment | Portfolio, Analytics, Ranking, related workflow contexts | Due Diligence Started occurred and may trigger follow-up business reactions. |

| DueDiligenceCompleted | Funding and Investment | Portfolio, Analytics, Ranking, related workflow contexts | Due Diligence Completed occurred and may trigger follow-up business reactions. |

| FundingApproved | Funding and Investment | Portfolio, Analytics, Ranking, related workflow contexts | Funding Approved occurred and may trigger follow-up business reactions. |

| FundingRejected | Funding and Investment | Portfolio, Analytics, Ranking, related workflow contexts | Funding Rejected occurred and may trigger follow-up business reactions. |

| FundingConditionIssued | Funding and Investment | Portfolio, Analytics, Ranking, related workflow contexts | Funding Condition Issued occurred and may trigger follow-up business reactions. |

| FundingCommitted | Funding and Investment | Portfolio, Analytics, Ranking, related workflow contexts | Funding Committed occurred and may trigger follow-up business reactions. |

| FundingDisbursed | Funding and Investment | Portfolio, Analytics, Ranking, related workflow contexts | Funding Disbursed occurred and may trigger follow-up business reactions. |

| FundingReportSubmitted | Funding and Investment | Portfolio, Analytics, Ranking, related workflow contexts | Funding Report Submitted occurred and may trigger follow-up business reactions. |

| InvestmentRoundOpened | Funding and Investment | Portfolio, Analytics, Ranking, related workflow contexts | Investment Round Opened occurred and may trigger follow-up business reactions. |

| InvestmentRoundClosed | Funding and Investment | Portfolio, Analytics, Ranking, related workflow contexts | Investment Round Closed occurred and may trigger follow-up business reactions. |

## 9.8 Recruitment Events

| Event | Producer Context | Likely Consumers | Meaning |
|---|---|---|---|

| CompanyVerified | Recruitment | Portfolio, Analytics, Ranking, related workflow contexts | Company Verified occurred and may trigger follow-up business reactions. |

| OpportunityPublished | Recruitment | Portfolio, Analytics, Ranking, related workflow contexts | Opportunity Published occurred and may trigger follow-up business reactions. |

| OpportunityClosed | Recruitment | Portfolio, Analytics, Ranking, related workflow contexts | Opportunity Closed occurred and may trigger follow-up business reactions. |

| ApplicationSubmitted | Recruitment | Portfolio, Analytics, Ranking, related workflow contexts | Application Submitted occurred and may trigger follow-up business reactions. |

| CandidateShortlisted | Recruitment | Portfolio, Analytics, Ranking, related workflow contexts | Candidate Shortlisted occurred and may trigger follow-up business reactions. |

| InterviewRequested | Recruitment | Portfolio, Analytics, Ranking, related workflow contexts | Interview Requested occurred and may trigger follow-up business reactions. |

| OfferIssued | Recruitment | Portfolio, Analytics, Ranking, related workflow contexts | Offer Issued occurred and may trigger follow-up business reactions. |

| OfferAccepted | Recruitment | Portfolio, Analytics, Ranking, related workflow contexts | Offer Accepted occurred and may trigger follow-up business reactions. |

| InternshipStarted | Recruitment | Portfolio, Analytics, Ranking, related workflow contexts | Internship Started occurred and may trigger follow-up business reactions. |

| InternshipCompleted | Recruitment | Portfolio, Analytics, Ranking, related workflow contexts | Internship Completed occurred and may trigger follow-up business reactions. |

| EmploymentRecorded | Recruitment | Portfolio, Analytics, Ranking, related workflow contexts | Employment Recorded occurred and may trigger follow-up business reactions. |

## 9.9 Events, Community, Marketplace Events

| Event | Producer Context | Likely Consumers | Meaning |
|---|---|---|---|

| EventCreated | Events, Community, Marketplace | Portfolio, Analytics, Ranking, related workflow contexts | Event Created occurred and may trigger follow-up business reactions. |

| EventRegistrationOpened | Events, Community, Marketplace | Portfolio, Analytics, Ranking, related workflow contexts | Event Registration Opened occurred and may trigger follow-up business reactions. |

| ParticipantRegistered | Events, Community, Marketplace | Portfolio, Analytics, Ranking, related workflow contexts | Participant Registered occurred and may trigger follow-up business reactions. |

| EventCompleted | Events, Community, Marketplace | Portfolio, Analytics, Ranking, related workflow contexts | Event Completed occurred and may trigger follow-up business reactions. |

| ChallengePublished | Events, Community, Marketplace | Portfolio, Analytics, Ranking, related workflow contexts | Challenge Published occurred and may trigger follow-up business reactions. |

| ChallengeSubmissionReceived | Events, Community, Marketplace | Portfolio, Analytics, Ranking, related workflow contexts | Challenge Submission Received occurred and may trigger follow-up business reactions. |

| ChallengeWinnerSelected | Events, Community, Marketplace | Portfolio, Analytics, Ranking, related workflow contexts | Challenge Winner Selected occurred and may trigger follow-up business reactions. |

| ChallengeOutputConvertedToProject | Events, Community, Marketplace | Portfolio, Analytics, Ranking, related workflow contexts | Challenge Output Converted To Project occurred and may trigger follow-up business reactions. |

| CommunityCreated | Events, Community, Marketplace | Portfolio, Analytics, Ranking, related workflow contexts | Community Created occurred and may trigger follow-up business reactions. |

| CommunityMemberJoined | Events, Community, Marketplace | Portfolio, Analytics, Ranking, related workflow contexts | Community Member Joined occurred and may trigger follow-up business reactions. |

| CommunityContributionRecorded | Events, Community, Marketplace | Portfolio, Analytics, Ranking, related workflow contexts | Community Contribution Recorded occurred and may trigger follow-up business reactions. |

| PartnershipProposed | Events, Community, Marketplace | Portfolio, Analytics, Ranking, related workflow contexts | Partnership Proposed occurred and may trigger follow-up business reactions. |

| PartnershipAccepted | Events, Community, Marketplace | Portfolio, Analytics, Ranking, related workflow contexts | Partnership Accepted occurred and may trigger follow-up business reactions. |

| MarketplaceListingPublished | Events, Community, Marketplace | Portfolio, Analytics, Ranking, related workflow contexts | Marketplace Listing Published occurred and may trigger follow-up business reactions. |

## 9.10 Ranking and Compliance Events

| Event | Producer Context | Likely Consumers | Meaning |
|---|---|---|---|

| RankingCycleConfigured | Ranking and Compliance | Portfolio, Analytics, Ranking, related workflow contexts | Ranking Cycle Configured occurred and may trigger follow-up business reactions. |

| RankingEvidenceSnapshotCreated | Ranking and Compliance | Portfolio, Analytics, Ranking, related workflow contexts | Ranking Evidence Snapshot Created occurred and may trigger follow-up business reactions. |

| RankingCalculated | Ranking and Compliance | Portfolio, Analytics, Ranking, related workflow contexts | Ranking Calculated occurred and may trigger follow-up business reactions. |

| RankingPublished | Ranking and Compliance | Portfolio, Analytics, Ranking, related workflow contexts | Ranking Published occurred and may trigger follow-up business reactions. |

| RankingAppealSubmitted | Ranking and Compliance | Portfolio, Analytics, Ranking, related workflow contexts | Ranking Appeal Submitted occurred and may trigger follow-up business reactions. |

| RankingAppealResolved | Ranking and Compliance | Portfolio, Analytics, Ranking, related workflow contexts | Ranking Appeal Resolved occurred and may trigger follow-up business reactions. |

| PolicyViolationReported | Ranking and Compliance | Portfolio, Analytics, Ranking, related workflow contexts | Policy Violation Reported occurred and may trigger follow-up business reactions. |

| ComplianceCaseOpened | Ranking and Compliance | Portfolio, Analytics, Ranking, related workflow contexts | Compliance Case Opened occurred and may trigger follow-up business reactions. |

| ComplianceCaseResolved | Ranking and Compliance | Portfolio, Analytics, Ranking, related workflow contexts | Compliance Case Resolved occurred and may trigger follow-up business reactions. |

| EvidenceRevoked | Ranking and Compliance | Portfolio, Analytics, Ranking, related workflow contexts | Evidence Revoked occurred and may trigger follow-up business reactions. |

| ImpactReportGenerated | Ranking and Compliance | Portfolio, Analytics, Ranking, related workflow contexts | Impact Report Generated occurred and may trigger follow-up business reactions. |

| NationalIndicatorUpdated | Ranking and Compliance | Portfolio, Analytics, Ranking, related workflow contexts | National Indicator Updated occurred and may trigger follow-up business reactions. |

# 10. Commands

Commands express business intent. A command may be accepted, rejected, or produce one or more domain events.

## 10.1 Identity and Institution Commands

| Command | Typical Actor | Expected Business Outcome |
|---|---|---|

| RegisterUser | Authorized actor in Identity and Institution context | Attempts to register user and, if valid, produces a domain event. |

| VerifyIdentity | Authorized actor in Identity and Institution context | Attempts to verify identity and, if valid, produces a domain event. |

| RegisterOrganization | Authorized actor in Identity and Institution context | Attempts to register organization and, if valid, produces a domain event. |

| VerifyInstitution | Authorized actor in Identity and Institution context | Attempts to verify institution and, if valid, produces a domain event. |

| RequestStudentAffiliation | Authorized actor in Identity and Institution context | Attempts to request student affiliation and, if valid, produces a domain event. |

| ApproveStudentAffiliation | Authorized actor in Identity and Institution context | Attempts to approve student affiliation and, if valid, produces a domain event. |

| AssignFacultyRole | Authorized actor in Identity and Institution context | Attempts to assign faculty role and, if valid, produces a domain event. |

| DelegateRoleAuthority | Authorized actor in Identity and Institution context | Attempts to delegate role authority and, if valid, produces a domain event. |

| RevokeRoleAuthority | Authorized actor in Identity and Institution context | Attempts to revoke role authority and, if valid, produces a domain event. |

## 10.2 Learning and Portfolio Commands

| Command | Typical Actor | Expected Business Outcome |
|---|---|---|

| DeclareLearningGoal | Authorized actor in Learning and Portfolio context | Attempts to declare learning goal and, if valid, produces a domain event. |

| StartLearningTrack | Authorized actor in Learning and Portfolio context | Attempts to start learning track and, if valid, produces a domain event. |

| SubmitSkillEvidence | Authorized actor in Learning and Portfolio context | Attempts to submit skill evidence and, if valid, produces a domain event. |

| VerifySkillEvidence | Authorized actor in Learning and Portfolio context | Attempts to verify skill evidence and, if valid, produces a domain event. |

| IssueSkillBadge | Authorized actor in Learning and Portfolio context | Attempts to issue skill badge and, if valid, produces a domain event. |

| CompleteGapAnalysis | Authorized actor in Learning and Portfolio context | Attempts to complete gap analysis and, if valid, produces a domain event. |

| CreatePortfolio | Authorized actor in Learning and Portfolio context | Attempts to create portfolio and, if valid, produces a domain event. |

| AddPortfolioItem | Authorized actor in Learning and Portfolio context | Attempts to add portfolio item and, if valid, produces a domain event. |

| ChangePortfolioVisibility | Authorized actor in Learning and Portfolio context | Attempts to change portfolio visibility and, if valid, produces a domain event. |

| AddRecommendation | Authorized actor in Learning and Portfolio context | Attempts to add recommendation and, if valid, produces a domain event. |

## 10.3 Project Commands

| Command | Typical Actor | Expected Business Outcome |
|---|---|---|

| CreateProject | Authorized actor in Project context | Attempts to create project and, if valid, produces a domain event. |

| FormProjectTeam | Authorized actor in Project context | Attempts to form project team and, if valid, produces a domain event. |

| SubmitProjectProposal | Authorized actor in Project context | Attempts to submit project proposal and, if valid, produces a domain event. |

| ApproveProject | Authorized actor in Project context | Attempts to approve project and, if valid, produces a domain event. |

| RejectProject | Authorized actor in Project context | Attempts to reject project and, if valid, produces a domain event. |

| CompleteProjectMilestone | Authorized actor in Project context | Attempts to complete project milestone and, if valid, produces a domain event. |

| SubmitProjectEvidence | Authorized actor in Project context | Attempts to submit project evidence and, if valid, produces a domain event. |

| ReviewProject | Authorized actor in Project context | Attempts to review project and, if valid, produces a domain event. |

| CompleteProject | Authorized actor in Project context | Attempts to complete project and, if valid, produces a domain event. |

| NominateProjectForInnovation | Authorized actor in Project context | Attempts to nominate project for innovation and, if valid, produces a domain event. |

| RaiseProjectOwnershipDispute | Authorized actor in Project context | Attempts to raise project ownership dispute and, if valid, produces a domain event. |

## 10.4 Research Commands

| Command | Typical Actor | Expected Business Outcome |
|---|---|---|

| SubmitResearchProposal | Authorized actor in Research context | Attempts to submit research proposal and, if valid, produces a domain event. |

| ApproveResearch | Authorized actor in Research context | Attempts to approve research and, if valid, produces a domain event. |

| GrantEthicsApproval | Authorized actor in Research context | Attempts to grant ethics approval and, if valid, produces a domain event. |

| SubmitResearchOutput | Authorized actor in Research context | Attempts to submit research output and, if valid, produces a domain event. |

| ReviewResearch | Authorized actor in Research context | Attempts to review research and, if valid, produces a domain event. |

| SubmitPublicationRecord | Authorized actor in Research context | Attempts to submit publication record and, if valid, produces a domain event. |

| ConfirmPublicationAccepted | Authorized actor in Research context | Attempts to confirm publication accepted and, if valid, produces a domain event. |

| ConfirmPublicationPublished | Authorized actor in Research context | Attempts to confirm publication published and, if valid, produces a domain event. |

| SubmitPatentDisclosure | Authorized actor in Research context | Attempts to submit patent disclosure and, if valid, produces a domain event. |

| ConfirmPatentFiled | Authorized actor in Research context | Attempts to confirm patent filed and, if valid, produces a domain event. |

| RecommendResearchCommercialization | Authorized actor in Research context | Attempts to recommend research commercialization and, if valid, produces a domain event. |

## 10.5 Innovation Commands

| Command | Typical Actor | Expected Business Outcome |
|---|---|---|

| SubmitInnovationCandidate | Authorized actor in Innovation context | Attempts to submit innovation candidate and, if valid, produces a domain event. |

| StartInnovationScreening | Authorized actor in Innovation context | Attempts to start innovation screening and, if valid, produces a domain event. |

| RequestInnovationEvaluation | Authorized actor in Innovation context | Attempts to request innovation evaluation and, if valid, produces a domain event. |

| CompleteInnovationEvaluation | Authorized actor in Innovation context | Attempts to complete innovation evaluation and, if valid, produces a domain event. |

| ValidateInnovation | Authorized actor in Innovation context | Attempts to validate innovation and, if valid, produces a domain event. |

| RejectInnovation | Authorized actor in Innovation context | Attempts to reject innovation and, if valid, produces a domain event. |

| RequestInnovationRevision | Authorized actor in Innovation context | Attempts to request innovation revision and, if valid, produces a domain event. |

| RecommendIncubation | Authorized actor in Innovation context | Attempts to recommend incubation and, if valid, produces a domain event. |

| RecommendPatentPath | Authorized actor in Innovation context | Attempts to recommend patent path and, if valid, produces a domain event. |

| RecommendIndustryPilot | Authorized actor in Innovation context | Attempts to recommend industry pilot and, if valid, produces a domain event. |

## 10.6 Incubation and Startup Commands

| Command | Typical Actor | Expected Business Outcome |
|---|---|---|

| CreateIncubationProgram | Authorized actor in Incubation and Startup context | Attempts to create incubation program and, if valid, produces a domain event. |

| OpenIncubationApplications | Authorized actor in Incubation and Startup context | Attempts to open incubation applications and, if valid, produces a domain event. |

| SubmitIncubationApplication | Authorized actor in Incubation and Startup context | Attempts to submit incubation application and, if valid, produces a domain event. |

| AcceptIncubationApplication | Authorized actor in Incubation and Startup context | Attempts to accept incubation application and, if valid, produces a domain event. |

| RejectIncubationApplication | Authorized actor in Incubation and Startup context | Attempts to reject incubation application and, if valid, produces a domain event. |

| FormCohort | Authorized actor in Incubation and Startup context | Attempts to form cohort and, if valid, produces a domain event. |

| AssignMentorToCohort | Authorized actor in Incubation and Startup context | Attempts to assign mentor to cohort and, if valid, produces a domain event. |

| StartIncubation | Authorized actor in Incubation and Startup context | Attempts to start incubation and, if valid, produces a domain event. |

| CompleteIncubationMilestone | Authorized actor in Incubation and Startup context | Attempts to complete incubation milestone and, if valid, produces a domain event. |

| ReviewIncubationProgress | Authorized actor in Incubation and Startup context | Attempts to review incubation progress and, if valid, produces a domain event. |

| CompleteIncubation | Authorized actor in Incubation and Startup context | Attempts to complete incubation and, if valid, produces a domain event. |

| CreateStartup | Authorized actor in Incubation and Startup context | Attempts to create startup and, if valid, produces a domain event. |

| AddFounder | Authorized actor in Incubation and Startup context | Attempts to add founder and, if valid, produces a domain event. |

| ChangeFounderRole | Authorized actor in Incubation and Startup context | Attempts to change founder role and, if valid, produces a domain event. |

| UpdateStartupStage | Authorized actor in Incubation and Startup context | Attempts to update startup stage and, if valid, produces a domain event. |

| AddTractionEvidence | Authorized actor in Incubation and Startup context | Attempts to add traction evidence and, if valid, produces a domain event. |

| CloseStartup | Authorized actor in Incubation and Startup context | Attempts to close startup and, if valid, produces a domain event. |

## 10.7 Funding and Investment Commands

| Command | Typical Actor | Expected Business Outcome |
|---|---|---|

| CreateFundingRequest | Authorized actor in Funding and Investment context | Attempts to create funding request and, if valid, produces a domain event. |

| SubmitFundingRequest | Authorized actor in Funding and Investment context | Attempts to submit funding request and, if valid, produces a domain event. |

| RegisterInvestorInterest | Authorized actor in Funding and Investment context | Attempts to register investor interest and, if valid, produces a domain event. |

| StartDueDiligence | Authorized actor in Funding and Investment context | Attempts to start due diligence and, if valid, produces a domain event. |

| CompleteDueDiligence | Authorized actor in Funding and Investment context | Attempts to complete due diligence and, if valid, produces a domain event. |

| ApproveFunding | Authorized actor in Funding and Investment context | Attempts to approve funding and, if valid, produces a domain event. |

| RejectFunding | Authorized actor in Funding and Investment context | Attempts to reject funding and, if valid, produces a domain event. |

| IssueFundingCondition | Authorized actor in Funding and Investment context | Attempts to issue funding condition and, if valid, produces a domain event. |

| CommitFunding | Authorized actor in Funding and Investment context | Attempts to commit funding and, if valid, produces a domain event. |

| RecordFundingDisbursement | Authorized actor in Funding and Investment context | Attempts to record funding disbursement and, if valid, produces a domain event. |

| SubmitFundingReport | Authorized actor in Funding and Investment context | Attempts to submit funding report and, if valid, produces a domain event. |

| OpenInvestmentRound | Authorized actor in Funding and Investment context | Attempts to open investment round and, if valid, produces a domain event. |

| CloseInvestmentRound | Authorized actor in Funding and Investment context | Attempts to close investment round and, if valid, produces a domain event. |

## 10.8 Recruitment, Events, Community, Ranking, Compliance Commands

| Command | Typical Actor | Expected Business Outcome |
|---|---|---|

| PublishOpportunity | Authorized actor in Recruitment, Events, Community, Ranking, Compliance context | Attempts to publish opportunity and, if valid, produces a domain event. |

| CloseOpportunity | Authorized actor in Recruitment, Events, Community, Ranking, Compliance context | Attempts to close opportunity and, if valid, produces a domain event. |

| SubmitApplication | Authorized actor in Recruitment, Events, Community, Ranking, Compliance context | Attempts to submit application and, if valid, produces a domain event. |

| ShortlistCandidate | Authorized actor in Recruitment, Events, Community, Ranking, Compliance context | Attempts to shortlist candidate and, if valid, produces a domain event. |

| RequestInterview | Authorized actor in Recruitment, Events, Community, Ranking, Compliance context | Attempts to request interview and, if valid, produces a domain event. |

| IssueOffer | Authorized actor in Recruitment, Events, Community, Ranking, Compliance context | Attempts to issue offer and, if valid, produces a domain event. |

| AcceptOffer | Authorized actor in Recruitment, Events, Community, Ranking, Compliance context | Attempts to accept offer and, if valid, produces a domain event. |

| RecordInternshipCompletion | Authorized actor in Recruitment, Events, Community, Ranking, Compliance context | Attempts to record internship completion and, if valid, produces a domain event. |

| CreateEvent | Authorized actor in Recruitment, Events, Community, Ranking, Compliance context | Attempts to create event and, if valid, produces a domain event. |

| RegisterForEvent | Authorized actor in Recruitment, Events, Community, Ranking, Compliance context | Attempts to register for event and, if valid, produces a domain event. |

| PublishChallenge | Authorized actor in Recruitment, Events, Community, Ranking, Compliance context | Attempts to publish challenge and, if valid, produces a domain event. |

| SubmitChallengeEntry | Authorized actor in Recruitment, Events, Community, Ranking, Compliance context | Attempts to submit challenge entry and, if valid, produces a domain event. |

| SelectChallengeWinner | Authorized actor in Recruitment, Events, Community, Ranking, Compliance context | Attempts to select challenge winner and, if valid, produces a domain event. |

| CreateCommunity | Authorized actor in Recruitment, Events, Community, Ranking, Compliance context | Attempts to create community and, if valid, produces a domain event. |

| JoinCommunity | Authorized actor in Recruitment, Events, Community, Ranking, Compliance context | Attempts to join community and, if valid, produces a domain event. |

| RecordCommunityContribution | Authorized actor in Recruitment, Events, Community, Ranking, Compliance context | Attempts to record community contribution and, if valid, produces a domain event. |

| ConfigureRankingCycle | Authorized actor in Recruitment, Events, Community, Ranking, Compliance context | Attempts to configure ranking cycle and, if valid, produces a domain event. |

| PublishRanking | Authorized actor in Recruitment, Events, Community, Ranking, Compliance context | Attempts to publish ranking and, if valid, produces a domain event. |

| SubmitRankingAppeal | Authorized actor in Recruitment, Events, Community, Ranking, Compliance context | Attempts to submit ranking appeal and, if valid, produces a domain event. |

| ReportPolicyViolation | Authorized actor in Recruitment, Events, Community, Ranking, Compliance context | Attempts to report policy violation and, if valid, produces a domain event. |

| ResolveComplianceCase | Authorized actor in Recruitment, Events, Community, Ranking, Compliance context | Attempts to resolve compliance case and, if valid, produces a domain event. |

# 11. Policies

Policies are decision rules, governance mechanisms, and institutional constraints that guide domain behavior. Some may later be automated, but they are business policies first.

| Policy | Purpose | Primary Contexts |
|---|---|---|

| Institutional Verification Policy | Defines consistent decision-making, authority, evidence expectations, and fairness for institutional verification. | Relevant domain contexts and Administration/Compliance. |

| Academic Affiliation Approval Policy | Defines consistent decision-making, authority, evidence expectations, and fairness for academic affiliation approval. | Relevant domain contexts and Administration/Compliance. |

| Project Approval Policy | Defines consistent decision-making, authority, evidence expectations, and fairness for project approval. | Relevant domain contexts and Administration/Compliance. |

| Research Ethics Policy | Defines consistent decision-making, authority, evidence expectations, and fairness for research ethics. | Relevant domain contexts and Administration/Compliance. |

| Publication Verification Policy | Defines consistent decision-making, authority, evidence expectations, and fairness for publication verification. | Relevant domain contexts and Administration/Compliance. |

| Patent Disclosure Policy | Defines consistent decision-making, authority, evidence expectations, and fairness for patent disclosure. | Relevant domain contexts and Administration/Compliance. |

| Innovation Validation Policy | Defines consistent decision-making, authority, evidence expectations, and fairness for innovation validation. | Relevant domain contexts and Administration/Compliance. |

| Incubation Selection Policy | Defines consistent decision-making, authority, evidence expectations, and fairness for incubation selection. | Relevant domain contexts and Administration/Compliance. |

| Cohort Graduation Policy | Defines consistent decision-making, authority, evidence expectations, and fairness for cohort graduation. | Relevant domain contexts and Administration/Compliance. |

| Mentor Verification Policy | Defines consistent decision-making, authority, evidence expectations, and fairness for mentor verification. | Relevant domain contexts and Administration/Compliance. |

| Mentorship Confidentiality Policy | Defines consistent decision-making, authority, evidence expectations, and fairness for mentorship confidentiality. | Relevant domain contexts and Administration/Compliance. |

| Conflict-of-Interest Policy | Defines consistent decision-making, authority, evidence expectations, and fairness for conflict-of-interest. | Relevant domain contexts and Administration/Compliance. |

| Funding Readiness Policy | Defines consistent decision-making, authority, evidence expectations, and fairness for funding readiness. | Relevant domain contexts and Administration/Compliance. |

| Grant Eligibility Policy | Defines consistent decision-making, authority, evidence expectations, and fairness for grant eligibility. | Relevant domain contexts and Administration/Compliance. |

| Investment Confidentiality Policy | Defines consistent decision-making, authority, evidence expectations, and fairness for investment confidentiality. | Relevant domain contexts and Administration/Compliance. |

| Due Diligence Policy | Defines consistent decision-making, authority, evidence expectations, and fairness for due diligence. | Relevant domain contexts and Administration/Compliance. |

| Post-Funding Reporting Policy | Defines consistent decision-making, authority, evidence expectations, and fairness for post-funding reporting. | Relevant domain contexts and Administration/Compliance. |

| Recruitment Fairness Policy | Defines consistent decision-making, authority, evidence expectations, and fairness for recruitment fairness. | Relevant domain contexts and Administration/Compliance. |

| Portfolio Sharing Policy | Defines consistent decision-making, authority, evidence expectations, and fairness for portfolio sharing. | Relevant domain contexts and Administration/Compliance. |

| Ranking Eligibility Policy | Defines consistent decision-making, authority, evidence expectations, and fairness for ranking eligibility. | Relevant domain contexts and Administration/Compliance. |

| Evidence Weighting Policy | Defines consistent decision-making, authority, evidence expectations, and fairness for evidence weighting. | Relevant domain contexts and Administration/Compliance. |

| Anti-Gaming Policy | Defines consistent decision-making, authority, evidence expectations, and fairness for anti-gaming. | Relevant domain contexts and Administration/Compliance. |

| Ranking Appeal Policy | Defines consistent decision-making, authority, evidence expectations, and fairness for ranking appeal. | Relevant domain contexts and Administration/Compliance. |

| Government Reporting Policy | Defines consistent decision-making, authority, evidence expectations, and fairness for government reporting. | Relevant domain contexts and Administration/Compliance. |

| Municipality Innovation Policy | Defines consistent decision-making, authority, evidence expectations, and fairness for municipality innovation. | Relevant domain contexts and Administration/Compliance. |

| Marketplace Trust Policy | Defines consistent decision-making, authority, evidence expectations, and fairness for marketplace trust. | Relevant domain contexts and Administration/Compliance. |

| Data Ownership Policy | Defines consistent decision-making, authority, evidence expectations, and fairness for data ownership. | Relevant domain contexts and Administration/Compliance. |

| Data Retention Policy | Defines consistent decision-making, authority, evidence expectations, and fairness for data retention. | Relevant domain contexts and Administration/Compliance. |

| Dispute Resolution Policy | Defines consistent decision-making, authority, evidence expectations, and fairness for dispute resolution. | Relevant domain contexts and Administration/Compliance. |

| Misconduct Policy | Defines consistent decision-making, authority, evidence expectations, and fairness for misconduct. | Relevant domain contexts and Administration/Compliance. |

| Audit Policy | Defines consistent decision-making, authority, evidence expectations, and fairness for audit. | Relevant domain contexts and Administration/Compliance. |

# 12. State Machines

State machines describe business lifecycles. They are not implementation diagrams; they show valid business progression and transition logic.

## 12.1 Student Lifecycle

```text
Prospective User
    ↓
Registered User
    ↓
Identity Verified
    ↓
Affiliation Pending
    ↓
Active Student
    ↓
Graduated / Alumni / Inactive / Suspended
```

## 12.2 Project Lifecycle

```text
Draft
    ↓
Proposed
    ↓
Approved / Rejected / Revision Required
    ↓
In Progress
    ↓
Submitted for Review
    ↓
Completed
    ↓
Innovation Candidate / Research Linked / Startup Candidate / Archived
```

## 12.3 Research Lifecycle

```text
Research Idea
    ↓
Proposal Submitted
    ↓
Approved / Rejected / Revision Required
    ↓
Ethics Pending if required
    ↓
Active Research
    ↓
Output Submitted
    ↓
Published / Patent Path / Commercialization Candidate / Archived
```

## 12.4 Innovation Lifecycle

```text
Identified
    ↓
Submitted
    ↓
Screening
    ↓
Evidence Required / Under Evaluation / Rejected
    ↓
Validated
    ↓
Incubation Ready / Patent Path / Industry Pilot / Research Path / Archived
```

## 12.5 Incubation Lifecycle

```text
Program Planned
    ↓
Applications Open
    ↓
Selection
    ↓
Cohort Formed
    ↓
Active Incubation
    ↓
At Risk / Graduation Review / Exited
    ↓
Graduated
```

## 12.6 Startup Lifecycle

```text
Concept
    ↓
Profile Created
    ↓
Founder Verified
    ↓
Problem Validated
    ↓
Prototype
    ↓
MVP
    ↓
Pilot / Fundraising / Early Revenue
    ↓
Funded
    ↓
Growth / Closed / Exited
```

## 12.7 Grant Lifecycle

```text
Grant Draft
    ↓
Open for Applications
    ↓
Application Submitted
    ↓
Eligibility Review
    ↓
Under Review
    ↓
Approved / Conditional / Rejected
    ↓
Disbursed
    ↓
Reporting
    ↓
Closed
```

## 12.8 Investment Lifecycle

```text
Funding Need
    ↓
Request Draft
    ↓
Submitted
    ↓
Investor Interested
    ↓
Due Diligence
    ↓
Term Discussion
    ↓
Committed
    ↓
Funded
    ↓
Monitored / New Round / Exit Recorded
```

## 12.9 Job Application Lifecycle

```text
Opportunity Published
    ↓
Application Submitted
    ↓
Shortlisted / Rejected
    ↓
Interview
    ↓
Offer Issued
    ↓
Accepted / Declined
    ↓
Active Placement
    ↓
Completed / Employed / Ended
```

## 12.10 Mentorship Lifecycle

```text
Need Identified
    ↓
Request Submitted
    ↓
Match Proposed
    ↓
Active Engagement
    ↓
Closure Review
    ↓
Completed / Closed / Under Review
```

## 12.11 Ranking Lifecycle

```text
Criteria Draft
    ↓
Criteria Published
    ↓
Evidence Snapshot
    ↓
Calculation
    ↓
Draft Results
    ↓
Appeal Window
    ↓
Published
    ↓
Locked
```

## 12.12 Compliance Case Lifecycle

```text
Reported
    ↓
Opened
    ↓
Evidence Review
    ↓
Decision Draft
    ↓
Resolved / Sanctioned / Dismissed
    ↓
Appealed / Closed
```

# 13. Domain Services

Domain services are business operations that do not naturally belong to one entity or aggregate. They should not be confused with technical services.

| Domain Service | Business Purpose | Typical Inputs | Expected Decision or Output |
|---|---|---|---|

| Innovation Evaluation Service | Performs cross-aggregate business reasoning for innovation evaluation. | Relevant evidence, policies, actor roles, lifecycle status, and contextual constraints. | Recommendation, eligibility result, score, risk flag, match, or validation decision. |

| Portfolio Scoring Service | Performs cross-aggregate business reasoning for portfolio scoring. | Relevant evidence, policies, actor roles, lifecycle status, and contextual constraints. | Recommendation, eligibility result, score, risk flag, match, or validation decision. |

| Skill Assessment Service | Performs cross-aggregate business reasoning for skill assessment. | Relevant evidence, policies, actor roles, lifecycle status, and contextual constraints. | Recommendation, eligibility result, score, risk flag, match, or validation decision. |

| Roadmap Evaluation Service | Performs cross-aggregate business reasoning for roadmap evaluation. | Relevant evidence, policies, actor roles, lifecycle status, and contextual constraints. | Recommendation, eligibility result, score, risk flag, match, or validation decision. |

| Mentor Matching Service | Performs cross-aggregate business reasoning for mentor matching. | Relevant evidence, policies, actor roles, lifecycle status, and contextual constraints. | Recommendation, eligibility result, score, risk flag, match, or validation decision. |

| Investor Matching Service | Performs cross-aggregate business reasoning for investor matching. | Relevant evidence, policies, actor roles, lifecycle status, and contextual constraints. | Recommendation, eligibility result, score, risk flag, match, or validation decision. |

| Recruiter Matching Service | Performs cross-aggregate business reasoning for recruiter matching. | Relevant evidence, policies, actor roles, lifecycle status, and contextual constraints. | Recommendation, eligibility result, score, risk flag, match, or validation decision. |

| Project-to-Innovation Qualification Service | Performs cross-aggregate business reasoning for project-to-innovation qualification. | Relevant evidence, policies, actor roles, lifecycle status, and contextual constraints. | Recommendation, eligibility result, score, risk flag, match, or validation decision. |

| Research Commercialization Assessment Service | Performs cross-aggregate business reasoning for research commercialization assessment. | Relevant evidence, policies, actor roles, lifecycle status, and contextual constraints. | Recommendation, eligibility result, score, risk flag, match, or validation decision. |

| Startup Readiness Service | Performs cross-aggregate business reasoning for startup readiness. | Relevant evidence, policies, actor roles, lifecycle status, and contextual constraints. | Recommendation, eligibility result, score, risk flag, match, or validation decision. |

| Funding Assessment Service | Performs cross-aggregate business reasoning for funding assessment. | Relevant evidence, policies, actor roles, lifecycle status, and contextual constraints. | Recommendation, eligibility result, score, risk flag, match, or validation decision. |

| Ranking Calculation Service | Performs cross-aggregate business reasoning for ranking calculation. | Relevant evidence, policies, actor roles, lifecycle status, and contextual constraints. | Recommendation, eligibility result, score, risk flag, match, or validation decision. |

| Conflict Detection Service | Performs cross-aggregate business reasoning for conflict detection. | Relevant evidence, policies, actor roles, lifecycle status, and contextual constraints. | Recommendation, eligibility result, score, risk flag, match, or validation decision. |

| Evidence Validation Service | Performs cross-aggregate business reasoning for evidence validation. | Relevant evidence, policies, actor roles, lifecycle status, and contextual constraints. | Recommendation, eligibility result, score, risk flag, match, or validation decision. |

| Policy Eligibility Service | Performs cross-aggregate business reasoning for policy eligibility. | Relevant evidence, policies, actor roles, lifecycle status, and contextual constraints. | Recommendation, eligibility result, score, risk flag, match, or validation decision. |

| Impact Measurement Service | Performs cross-aggregate business reasoning for impact measurement. | Relevant evidence, policies, actor roles, lifecycle status, and contextual constraints. | Recommendation, eligibility result, score, risk flag, match, or validation decision. |

| Collaboration Matching Service | Performs cross-aggregate business reasoning for collaboration matching. | Relevant evidence, policies, actor roles, lifecycle status, and contextual constraints. | Recommendation, eligibility result, score, risk flag, match, or validation decision. |

| Fraud Risk Assessment Service | Performs cross-aggregate business reasoning for fraud risk assessment. | Relevant evidence, policies, actor roles, lifecycle status, and contextual constraints. | Recommendation, eligibility result, score, risk flag, match, or validation decision. |

# 14. Business Capabilities

Business capabilities define what the ecosystem must be able to do independent of software design.

```text
National Innovation Ecosystem Capabilities
├── Ecosystem Governance
│   ├── Institutional onboarding
│   ├── Policy alignment
│   ├── Role and authority management
│   ├── Dispute and appeal governance
│   └── National reporting
├── Talent Development
│   ├── Learning roadmap alignment
│   ├── Skill evidence validation
│   ├── Portfolio development
│   ├── Career readiness assessment
│   └── Lifelong learning continuity
├── Project and Research Activation
│   ├── Project lifecycle management
│   ├── Final-year project continuity
│   ├── Research output tracking
│   ├── Patent and IP pathway support
│   └── Research commercialization readiness
├── Innovation Pipeline
│   ├── Idea capture
│   ├── Innovation candidate evaluation
│   ├── Validation evidence management
│   ├── Impact pathway classification
│   └── Innovation marketplace discovery
├── Venture Creation
│   ├── Startup formation readiness
│   ├── Incubation selection
│   ├── Cohort management
│   ├── Mentorship and milestones
│   └── Demo day and graduation readiness
├── Capital Mobilization
│   ├── Grant discovery
│   ├── Funding request preparation
│   ├── Investor matching
│   ├── Due diligence support
│   └── Post-funding monitoring
├── Industry and Employment Linkage
│   ├── Opportunity publishing
│   ├── Talent discovery
│   ├── Internship management
│   ├── Skill-based recruitment
│   └── Employer feedback loop
└── Intelligence and Impact
    ├── Ranking and recognition
    ├── Skill gap analytics
    ├── Innovation pipeline analytics
    ├── Regional ecosystem indicators
    └── Economic and social impact measurement
```

| Capability | Strategic Importance | Reason |
|---|---|---|

| Innovation Lifecycle Management | Very High | Required to convert learning, research, projects, and collaboration into trusted innovation outcomes. |

| Portfolio and Talent Evidence | Very High | Required to convert learning, research, projects, and collaboration into trusted innovation outcomes. |

| Project Continuity | Very High | Required to convert learning, research, projects, and collaboration into trusted innovation outcomes. |

| Incubation Pipeline | Very High | Required to convert learning, research, projects, and collaboration into trusted innovation outcomes. |

| Ecosystem Governance | Very High | Required to convert learning, research, projects, and collaboration into trusted innovation outcomes. |

| Skill Roadmap Alignment | High | Required to convert learning, research, projects, and collaboration into trusted innovation outcomes. |

| Research Commercialization | High | Required to convert learning, research, projects, and collaboration into trusted innovation outcomes. |

| Mentor Matching | High | Required to convert learning, research, projects, and collaboration into trusted innovation outcomes. |

| Funding Pipeline | High | Required to convert learning, research, projects, and collaboration into trusted innovation outcomes. |

| Recruitment Linkage | High | Required to convert learning, research, projects, and collaboration into trusted innovation outcomes. |

| Ranking and Recognition | Medium-High | Required to convert learning, research, projects, and collaboration into trusted innovation outcomes. |

| Events and Challenges | Medium-High | Required to convert learning, research, projects, and collaboration into trusted innovation outcomes. |

| Communities | Medium | Required to convert learning, research, projects, and collaboration into trusted innovation outcomes. |

| Government Monitoring | Medium-High | Required to convert learning, research, projects, and collaboration into trusted innovation outcomes. |

| Marketplace and Partnerships | Medium-High | Required to convert learning, research, projects, and collaboration into trusted innovation outcomes. |

# 15. Cross-Domain Workflows

Cross-domain workflows connect bounded contexts. These should become the basis for event storming and future PRD epics.

## 15.1 Student to Founder Workflow

```text
StudentAffiliationVerified
    ↓
LearningGoalDeclared
    ↓
ProjectCreated
    ↓
ProjectCompleted
    ↓
ProjectMarkedInnovationCandidate
    ↓
InnovationValidated
    ↓
IncubationRecommended
    ↓
IncubationApplicationAccepted
    ↓
StartupCreated
    ↓
FounderRolesDeclared
    ↓
FundingRequestSubmitted
    ↓
FundingApproved
```

**Primary Concern:** Maintain evidence, ownership clarity, consent, and lifecycle continuity across contexts.

## 15.2 Final-Year Project Continuity Workflow

```text
ProjectProposed
    ↓
SupervisorAssigned
    ↓
MilestonesCompleted
    ↓
ProjectSubmitted
    ↓
AcademicReviewCompleted
    ↓
OwnershipConfirmed
    ↓
InnovationEligibilityChecked
    ↓
ContinuationPathSelected
```

**Primary Concern:** Maintain evidence, ownership clarity, consent, and lifecycle continuity across contexts.

## 15.3 Research to Commercialization Workflow

```text
ResearchProposalSubmitted
    ↓
ResearchApproved
    ↓
EthicsChecked
    ↓
ResearchConducted
    ↓
OutputSubmitted
    ↓
PublicationOrPatentRecorded
    ↓
CommercializationAssessmentCompleted
    ↓
InnovationCandidateCreated
    ↓
PartnershipOrStartupPathInitiated
```

**Primary Concern:** Maintain evidence, ownership clarity, consent, and lifecycle continuity across contexts.

## 15.4 Industry Challenge to Recruitment Workflow

```text
CompanyVerified
    ↓
ChallengePublished
    ↓
TeamsRegister
    ↓
SubmissionsCreated
    ↓
JudgingCompleted
    ↓
StrongSubmissionsIdentified
    ↓
PortfolioEvidenceCreated
    ↓
InternshipOpportunityPublished
    ↓
OfferIssued
    ↓
InternshipCompleted
```

**Primary Concern:** Maintain evidence, ownership clarity, consent, and lifecycle continuity across contexts.

## 15.5 Skill Roadmap to Employment Workflow

```text
CareerGoalDeclared
    ↓
LearningTrackSelected
    ↓
GapAnalysisCompleted
    ↓
RecommendedActionsGenerated
    ↓
SkillEvidenceSubmitted
    ↓
BadgeIssued
    ↓
PortfolioReadinessImproved
    ↓
RecruiterDiscoversCandidate
    ↓
OfferAccepted
    ↓
EmploymentRecorded
```

**Primary Concern:** Maintain evidence, ownership clarity, consent, and lifecycle continuity across contexts.

## 15.6 Startup Funding Readiness Workflow

```text
StartupCreated
    ↓
FounderTeamVerified
    ↓
ProblemValidationAdded
    ↓
PrototypeEvidenceAdded
    ↓
TractionEvidenceAdded
    ↓
MentorReviewCompleted
    ↓
ReadinessAssessmentCompleted
    ↓
InvestorMatched
    ↓
DueDiligenceStarted
    ↓
FundingDecisionRecorded
```

**Primary Concern:** Maintain evidence, ownership clarity, consent, and lifecycle continuity across contexts.

## 15.7 Government Grant Workflow

```text
GovernmentProgramCreated
    ↓
GrantEligibilityPublished
    ↓
ApplicationSubmitted
    ↓
EligibilityChecked
    ↓
EvaluationCompleted
    ↓
GrantApproved
    ↓
FundingDisbursed
    ↓
ReportsSubmitted
    ↓
ImpactReportGenerated
    ↓
NationalIndicatorsUpdated
```

**Primary Concern:** Maintain evidence, ownership clarity, consent, and lifecycle continuity across contexts.

## 15.8 Alumni Reinvestment Workflow

```text
StudentGraduates
    ↓
AlumniProfileActivated
    ↓
AlumniJoinsCommunity
    ↓
AlumniBecomesMentorRecruiterInvestorOrSpeaker
    ↓
ContributionRecorded
    ↓
StudentsBenefit
    ↓
UniversityContributionImproves
```

**Primary Concern:** Maintain evidence, ownership clarity, consent, and lifecycle continuity across contexts.

## 15.9 Open Source Contribution Workflow

```text
ProjectCompleted
    ↓
LicenseEligibilityChecked
    ↓
OpenSourceReleaseApproved
    ↓
ContributionRecordAdded
    ↓
CommunityContributorsJoin
    ↓
SkillEvidenceUpdated
    ↓
PortfolioUpdated
    ↓
CompanyDiscoversMaintainer
```

**Primary Concern:** Maintain evidence, ownership clarity, consent, and lifecycle continuity across contexts.

## 15.10 Municipality Innovation Challenge Workflow

```text
MunicipalityProblemIdentified
    ↓
ChallengePublished
    ↓
UniversitiesParticipate
    ↓
SolutionsSubmitted
    ↓
PilotSelected
    ↓
LocalPilotImplemented
    ↓
ImpactEvidenceCollected
    ↓
GovernmentDashboardUpdated
    ↓
StartupResearchOrPolicyPathRecommended
```

**Primary Concern:** Maintain evidence, ownership clarity, consent, and lifecycle continuity across contexts.

# 16. Business Glossary

| Term | Definition |
|---|---|

| Actor | Any person, organization, institution, community, or formal group participating in the ecosystem. |

| User | A human actor who accesses the platform. A user may have multiple roles across contexts. |

| Participant | An actor actively involved in a specific activity such as a project, cohort, challenge, research work, investment review, or mentorship. |

| Affiliation | A formal relationship between a person and an organization, such as student-university, faculty-department, mentor-company, or officer-government unit. |

| Role | A business responsibility assigned within a context, such as project supervisor, evaluator, mentor, founder, recruiter, investor reviewer, or government officer. |

| Permission | The authority to perform a business action within a defined scope. Permission depends on role, context, ownership, delegation, and policy. |

| Student | A learner with active or historical academic affiliation. A student can also be a researcher, founder, community member, intern, or project owner. |

| Alumni | A former student who retains a lifelong portfolio and may later become mentor, founder, recruiter, investor, researcher, or ecosystem contributor. |

| Faculty | Academic staff responsible for teaching, supervision, research, assessment, mentoring, governance, or innovation support. |

| Researcher | A person conducting systematic inquiry. A researcher may be student, faculty, industry expert, independent scholar, or institutional researcher. |

| University | A higher education institution recognized as an academic authority that may govern colleges, departments, programs, research centers, and innovation policies. |

| College | An academic institution affiliated with or constituent of a university that operates programs and manages students locally. |

| Department | An academic unit responsible for a subject area, faculty group, academic program, research direction, and project supervision. |

| Program | A structured academic offering such as engineering, management, science, agriculture, medicine, design, or social science program. |

| Course | A formal learning unit. Course completion may support but does not automatically prove practical competency. |

| Learning Track | A career or capability-aligned pathway such as AI Engineer, Backend Engineer, Product Manager, Researcher, Cybersecurity Analyst, or Founder. |

| Roadmap | A sequence of skills, milestones, projects, assessments, and evidence expectations aligned to a learning or career goal. |

| Skill | A specific demonstrable capability such as programming, data analysis, research writing, product discovery, sales, grant writing, or public speaking. |

| Competency | A broader capability combining knowledge, skill, judgment, application, and context-sensitive performance. |

| Skill Evidence | Proof that a skill has been demonstrated, such as project review, internship feedback, code contribution, publication, assessment, certificate, or mentor validation. |

| Skill Badge | A recognized marker issued only after sufficient evidence validates a skill or competency. |

| Portfolio | A lifelong, evidence-backed innovation and talent profile. It is not just a resume; it contains verified learning, projects, research, startup participation, mentorship, employment, achievements, and impact. |

| Project | A structured work artifact created to solve a problem, demonstrate learning, conduct research, build a prototype, contribute to open source, satisfy academic requirements, or create commercial/social value. |

| Final-Year Project | A required academic project near graduation that may become portfolio evidence, research continuation, open-source work, innovation candidate, incubation candidate, or startup foundation. |

| Problem Statement | A clear definition of the problem, affected users, pain points, context, evidence, and desired improvement. |

| Prototype | A testable representation of a solution. It may be technical, physical, scientific, service-based, policy-based, or design-based. |

| MVP | A minimum viable product or pilot used to test real demand, usage, feasibility, or adoption. |

| Innovation | A validated project, research output, process, product, service, model, or prototype with novelty, usefulness, feasibility, and potential impact. |

| Innovation Candidate | An idea, project, research output, prototype, startup, or challenge submission submitted for innovation evaluation. |

| Validation | Evidence-based confirmation that a problem, skill, solution, market, research claim, impact claim, or venture assumption is credible. |

| Research | Systematic inquiry intended to generate knowledge, methods, evidence, datasets, publications, inventions, prototypes, policy insight, or applied solutions. |

| Publication | A research output made available through journal, conference, repository, thesis, report, technical note, or recognized publication channel. |

| Patent | A legal intellectual property claim for an invention. Patent lifecycle is separate from publication and project lifecycle. |

| Grant | Non-dilutive funding awarded for research, innovation, public interest, development, institutional support, or social impact. |

| Intellectual Property | Rights related to inventions, creative works, software, datasets, designs, methods, research outputs, or knowledge assets. |

| Startup | A new venture formed to build a scalable solution under uncertainty. Not every small business is a startup. |

| Founder | A person with explicit ownership, accountability, and leadership responsibility in a startup. |

| Incubation | Structured support that helps projects, innovations, or startups move toward validation, venture formation, funding readiness, and impact. |

| Cohort | A group of selected teams, startups, students, researchers, or innovators participating in the same program cycle. |

| Mentor | An experienced advisor who supports students, projects, researchers, founders, or startups without automatically holding approval authority. |

| Investor | An actor that evaluates and may provide capital. Includes angel investors, venture capital funds, corporate investors, diaspora investors, public funds, and impact investors. |

| Funding Round | A structured fundraising process such as pre-seed, seed, bridge, or growth round. |

| Due Diligence | Structured evidence review before funding, investment, grant approval, partnership, acquisition, or high-trust decision. |

| Company | An organization seeking talent, innovation, research collaboration, pilots, services, products, or investment opportunities. |

| Recruiter | A person or organization that sources and evaluates candidates for internships, jobs, fellowships, or contract roles. |

| Opportunity | Any opening for job, internship, challenge, grant, mentorship, partnership, research collaboration, incubation, or investment. |

| Challenge | A structured problem-solving opportunity with sponsor, eligibility, rules, deadline, evaluation criteria, and outcomes. |

| Hackathon | A time-bound event where participants build or propose solutions. In this ecosystem it should feed projects, portfolios, and incubation pathways. |

| Community | A group of practice around topic, region, institution, sector, technology, mission, alumni identity, founder stage, or social cause. |

| Innovation Marketplace | A structured discovery space for projects, research, startups, talent, mentors, challenges, funding, partnerships, and services. |

| Ranking | Contextual ordered recognition based on validated evidence and transparent criteria. |

| Innovation Score | Composite evaluation of innovation quality, novelty, feasibility, validation, collaboration, impact, and readiness. |

| Impact Score | Assessment of measurable economic, social, environmental, educational, institutional, or community impact. |

| Reputation | Trust accumulated through verified contribution, mentorship, delivery, ethical behavior, and ecosystem participation. |

| Evidence | Any artifact or confirmation supporting a business claim. Evidence is the currency of trust in the ecosystem. |

| Conflict of Interest | A situation where an actor has personal, financial, institutional, relational, or competitive interest that may bias a decision. |

# Appendix A: Event Storming Starter Map

```text
UserRegistered
→ StudentAffiliationVerified
→ LearningGoalDeclared
→ ProjectCreated
→ ProjectApproved
→ ProjectCompleted
→ ProjectMarkedInnovationCandidate
→ InnovationValidated
→ IncubationRecommended
→ IncubationApplicationAccepted
→ StartupCreated
→ MentorAssigned
→ IncubationMilestoneCompleted
→ StartupStageUpdated
→ FundingRequestSubmitted
→ InvestorInterestRegistered
→ FundingApproved
→ FundingDisbursed
→ StartupGraduated
→ EmploymentRecorded / InvestmentRoundClosed / ImpactReportGenerated
```

## Workshop Prompts

1. What event is missing between project completion and innovation validation?
2. Who has authority to approve each transition?
3. What evidence is required before high-trust decisions?
4. Which events should update portfolio?
5. Which events should influence ranking?
6. Which events should be private or confidential?
7. Which events should government see only in aggregate?
8. Which events create legal or intellectual property implications?
9. Which events currently happen informally in universities?
10. Which events are completely missing from Nepal's current innovation pipeline?

# Appendix B: Open Domain Questions

## Institution and Governance

1. Who verifies university, college, department, and program records?
2. Should universities self-manage structures or should a national council verify them?
3. How should affiliated colleges be modeled when they operate under different universities?
4. What minimum institutional information is required for participation?
5. How should suspended, merged, renamed, or discontinued programs be represented?

## Project and IP

1. Who owns final-year projects by default across different universities?
2. Should there be a standard project ownership declaration template?
3. How should sponsor-funded projects define IP and confidentiality?
4. How should team contribution disputes be resolved?
5. Can students continue university projects as independent startups after graduation?

## Innovation and Incubation

1. What minimum evidence qualifies a project as an innovation candidate?
2. Should each university have its own panel or should there be a shared evaluator pool?
3. How should discipline-specific innovation criteria differ?
4. Should social innovation and commercial startup tracks be separated?
5. What is the minimum demo day readiness standard?

## Funding and Investment

1. What evidence should be required before investor exposure?
2. How should investor verification work for Nepal, diaspora, and international investors?
3. How should grant funding and equity investment be clearly separated for founders?
4. What funding data should be visible to government?
5. What due diligence information should remain confidential?

## Ranking and Recognition

1. Should rankings be annual, semester-based, cohort-based, or continuous?
2. Which evidence types deserve the highest weight?
3. How should rankings avoid favoring wealthy urban institutions?
4. How should rural innovation and local impact be recognized?
5. Who hears appeals and how quickly?

## Privacy and Data Ownership

1. Who owns student portfolio data?
2. What can universities see after graduation?
3. What can recruiters see without direct student application?
4. What can investors see before startup permission is granted?
5. What should government see only in aggregate form?

---

# Closing Note

This domain model should be treated as a living business artifact. Its value is not in having a large list of nouns. Its value is in making the ecosystem's meaning, ownership, rules, lifecycles, events, and boundaries explicit before technical design begins. Once validated with stakeholders, it can become the foundation for SRS, PRD, UX research, event storming, microservice boundaries, CQRS models, event sourcing, AI governance, database design, and API design.