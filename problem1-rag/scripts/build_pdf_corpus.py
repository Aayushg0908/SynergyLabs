"""Generate the PDF part of the corpus (company policies + quarterly report).

The corpus describes a fictional company, Kestrel Labs, so the LLM cannot answer
from pre-training knowledge — every correct answer must come from retrieval.

Usage:  python scripts/build_pdf_corpus.py
Output: data/corpus/policies/*.pdf, data/corpus/finance/*.pdf
"""

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

ROOT = Path(__file__).resolve().parent.parent / "data" / "corpus"

# Each doc: (relative path, title, [(heading, [paragraphs...]), ...])
DOCS = [
    (
        "policies/travel-and-expense-policy.pdf",
        "Kestrel Labs Travel and Expense Policy (v3.2, effective 2026-01-01)",
        [
            ("1. Purpose and scope", [
                "This policy applies to all Kestrel Labs employees and contractors who travel or incur "
                "expenses on company business. Expenses outside this policy are not reimbursed unless "
                "approved in writing by the employee's department head before they are incurred.",
            ]),
            ("2. Booking travel", [
                "All flights, hotels and rental cars must be booked through the company travel portal, "
                "Navan. Bookings made outside the portal are reimbursed only if they are at least 10% "
                "cheaper than the cheapest comparable portal option, with a screenshot as proof.",
                "Flights must be booked at least 14 days in advance when possible. Economy class is the "
                "default. Premium economy is allowed for flights longer than 6 hours. Business class is "
                "allowed only for flights longer than 10 hours and requires VP approval.",
            ]),
            ("3. Hotels", [
                "The nightly hotel cap is $250 in standard cities and $375 in high-cost cities "
                "(New York, San Francisco, London, Zurich, Tokyo and Singapore), excluding taxes.",
            ]),
            ("4. Meals and per diem", [
                "Employees on domestic travel receive a meal per diem of $75 per day. International "
                "travel per diem is $95 per day. On the first and last day of travel, the per diem is "
                "75% of the daily amount. Alcohol is not reimbursable except at approved client dinners.",
                "Client entertainment is capped at $120 per person, including the employee, and the "
                "names and companies of all attendees must be recorded on the expense report.",
            ]),
            ("5. Ground transport and mileage", [
                "Taxis and ride-share are reimbursed for business trips. Personal car use is reimbursed "
                "at $0.67 per mile. Rental cars should be compact or intermediate class.",
            ]),
            ("6. Submitting expenses", [
                "Expense reports must be submitted in Navan within 30 days of the expense date. "
                "Itemized receipts are required for every expense over $25. Reports submitted more "
                "than 60 days late are not reimbursed.",
                "Expense reports are approved by the employee's direct manager. Any single expense "
                "above $5,000 also requires approval from Finance. Approved reimbursements are paid in "
                "the next payroll run.",
            ]),
            ("7. Corporate cards", [
                "Employees who travel more than four times per year are issued a corporate card. "
                "Personal purchases on a corporate card are prohibited; any accidental personal charge "
                "must be repaid within 15 days.",
            ]),
        ],
    ),
    (
        "policies/leave-policy.pdf",
        "Kestrel Labs Leave and Time-Off Policy (effective 2026-01-01)",
        [
            ("1. Paid time off (PTO)", [
                "Full-time employees accrue paid time off at a rate of 1.67 days per month, which is "
                "20 days per year. After 5 years of continuous service, the accrual increases to 25 "
                "days per year.",
                "Up to 5 unused PTO days may be carried over into the next calendar year. Carried-over "
                "days expire on March 31 if not used. Any remaining balance above the carry-over limit "
                "is forfeited on December 31.",
                "PTO requests of 3 days or more must be submitted at least 2 weeks in advance and "
                "approved by the employee's manager in Workday.",
            ]),
            ("2. Company holidays", [
                "Kestrel Labs observes 11 paid company holidays per year. The company also closes "
                "between December 24 and January 1 inclusive; these days do not count against PTO.",
            ]),
            ("3. Sick leave", [
                "Employees receive 10 days of paid sick leave per year, separate from PTO. Sick leave "
                "does not carry over. A doctor's note is required for an absence of more than 3 "
                "consecutive working days.",
            ]),
            ("4. Parental leave", [
                "Birth parents receive 20 weeks of fully paid parental leave. Non-birth parents, "
                "including adoptive and foster parents, receive 16 weeks of fully paid leave. Parental "
                "leave must be taken within 12 months of the birth or placement and may be split into "
                "up to two blocks.",
            ]),
            ("5. Sabbatical", [
                "After 7 years of continuous service, employees are eligible for a 4-week paid "
                "sabbatical, in addition to normal PTO. Sabbaticals must be taken within 2 years of "
                "becoming eligible and require 3 months' notice.",
            ]),
            ("6. Other leave", [
                "Bereavement leave is 5 paid days for an immediate family member and 2 paid days for "
                "an extended family member. Jury duty is fully paid for its duration. Employees "
                "receive 2 paid volunteer days per year.",
            ]),
        ],
    ),
    (
        "policies/information-security-policy.pdf",
        "Kestrel Labs Information Security Policy (v5.0)",
        [
            ("1. Data classification", [
                "All company data falls into one of four classes: Public, Internal, Confidential and "
                "Restricted. Customer telemetry and flight logs are classified as Confidential. "
                "Cryptographic keys, device signing keys and employee personal data are Restricted.",
                "Restricted data may only be stored in the approved secrets manager or the HR system. "
                "It must never be sent over email or chat, even internally.",
            ]),
            ("2. Passwords and authentication", [
                "Passwords must be at least 14 characters long. Multi-factor authentication is "
                "mandatory for every company system; hardware security keys are required for engineering "
                "and finance staff. Passwords are not rotated on a schedule, but must be changed "
                "immediately if a compromise is suspected.",
            ]),
            ("3. Devices", [
                "Company laptops must use full-disk encryption and the managed endpoint agent. Screens "
                "must lock automatically after 5 minutes of inactivity. Personal devices may access "
                "email and chat only, through the managed mobile app.",
            ]),
            ("4. Access reviews", [
                "Managers review their team's access to Confidential systems every quarter. Access to "
                "Restricted systems is reviewed every month by the security team. Access is revoked "
                "within 4 hours of an employee's departure.",
            ]),
            ("5. Reporting incidents", [
                "Suspected security incidents, including lost devices and phishing emails that were "
                "clicked, must be reported to security@kestrel-labs.example or the #security-incidents "
                "channel within 1 hour of discovery. Employees are never penalized for reporting in "
                "good faith.",
                "If a security incident involves customer data, the security team notifies affected "
                "customers within 72 hours of confirming the incident.",
            ]),
            ("6. Third-party software", [
                "New SaaS tools that will handle Internal data or above require a security review before "
                "purchase. Reviews take about 10 business days. Open-source libraries are allowed if "
                "their license is on the approved list (MIT, Apache 2.0, BSD).",
            ]),
        ],
    ),
    (
        "finance/q3-2026-quarterly-report.pdf",
        "Kestrel Labs Quarterly Business Report — Q3 2026 (internal)",
        [
            ("Summary", [
                "Q3 2026 revenue was $18.4 million, up 31% year over year and up 9% from Q2 2026. Gross "
                "margin was 58%, compared with 54% in Q3 2025. Operating loss narrowed to $1.2 million "
                "from $2.9 million a year earlier.",
            ]),
            ("Revenue by segment", [
                "Hardware (drones, gateways, payloads) generated $11.1 million. Cloud subscriptions "
                "generated $5.6 million, up 52% year over year. Services and Care+ plans generated "
                "$1.7 million.",
                "Annual recurring revenue (ARR) from cloud subscriptions reached $23.8 million at the end "
                "of the quarter. Net revenue retention for cloud customers was 118%.",
            ]),
            ("Customers and fleet", [
                "Kestrel ended the quarter with 1,240 paying customers, of which 86 are Enterprise "
                "customers. The active fleet reached 14,300 drones and 2,050 Vela gateways. The largest "
                "single deployment is 640 drones at an agricultural cooperative in Brazil.",
            ]),
            ("Operations", [
                "Average telemetry ingestion volume was 2.1 billion messages per day. There were two "
                "SEV-1 incidents in the quarter; neither caused customer data loss. Cloud platform uptime "
                "was 99.96%.",
                "Firmware 4.2 reached general availability in August and reduced crash reports by 23% "
                "compared with firmware 4.1.",
            ]),
            ("Headcount", [
                "Headcount was 412 at quarter end, including 188 in engineering. The company plans to "
                "hire 35 more engineers by the end of Q1 2027.",
            ]),
            ("Outlook", [
                "For Q4 2026, management expects revenue between $19.5 million and $20.5 million. The "
                "board approved a $6 million budget to open a European service center in Rotterdam in "
                "2027.",
            ]),
        ],
    ),
]


def build(path: Path, title: str, sections) -> None:
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"]), Spacer(1, 12)]
    for heading, paras in sections:
        story.append(Paragraph(heading, styles["Heading2"]))
        for p in paras:
            story.append(Paragraph(p, styles["BodyText"]))
            story.append(Spacer(1, 6))
    path.parent.mkdir(parents=True, exist_ok=True)
    # invariant=1 makes the output byte-identical across runs (stable hashes for idempotency tests)
    SimpleDocTemplate(str(path), pagesize=A4, title=title, author="Kestrel Labs", invariant=1).build(story)


if __name__ == "__main__":
    for rel, title, sections in DOCS:
        out = ROOT / rel
        build(out, title, sections)
        print("wrote", out.relative_to(ROOT.parent.parent))
