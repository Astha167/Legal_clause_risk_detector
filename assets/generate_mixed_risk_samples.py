import sys
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ss = getSampleStyleSheet()

# Create custom styles
title_style = ParagraphStyle(
    "ContractTitle",
    parent=ss["Title"],
    fontSize=16,
    leading=20,
    spaceAfter=14,
    alignment=1
)
heading_style = ParagraphStyle(
    "ClauseHeading",
    parent=ss["Heading3"],
    fontSize=11,
    leading=14,
    spaceBefore=8,
    spaceAfter=3,
    textColor="#1F2937"
)
body_style = ParagraphStyle(
    "ClauseBody",
    parent=ss["BodyText"],
    fontSize=9.5,
    leading=13,
    spaceAfter=6,
    textColor="#374151"
)

def create_pdf(text: str, filename: str, doc_title: str) -> Path:
    out_path = Path(__file__).resolve().parent / filename
    story = []
    
    blocks = [b.strip() for b in text.strip().split("\n\n") if b.strip()]
    
    # First block is the document title
    if blocks:
        story.append(Paragraph(blocks[0], title_style))
        story.append(Spacer(1, 8))
        blocks = blocks[1:]
        
    for block in blocks:
        lines = block.split("\n")
        # Check if first line is a clause header (e.g., "1. Term", "Section 2. Non-Compete")
        first = lines[0].strip()
        if first[:1].isdigit() or first.lower().startswith("section") or first.lower().startswith("article"):
            story.append(Paragraph(f"<b>{first}</b>", heading_style))
            rest = " ".join(ln.strip() for ln in lines[1:])
            if rest:
                story.append(Paragraph(rest, body_style))
        else:
            combined = " ".join(ln.strip() for ln in lines)
            story.append(Paragraph(combined, body_style))
        story.append(Spacer(1, 4))
        
    SimpleDocTemplate(str(out_path), pagesize=A4, title=doc_title, leftMargin=45, rightMargin=45, topMargin=45, bottomMargin=45).build(story)
    print(f"Generated {out_path}")
    return out_path


# 1. Software Development & Services Agreement (Balanced Mixed Risk)
# - High Risk: Uncapped client liability, Unilateral termination by client at will, Broad worldwide non-compete
# - Medium Risk: 1-year non-solicitation, 60-day auto-renewal, Acceptance deemed in 10 days
# - Low Risk: Standard payment terms, Delaware governing law, Mutual confidentiality, Severability, Entire agreement
SOFTWARE_DEV_AGREEMENT = """MASTER SOFTWARE DEVELOPMENT AGREEMENT

This Master Software Development Agreement (the "Agreement") is dated 12 January 2026 by and between Horizon Cloud Solutions Inc. ("Client") and StellarByte Interactive LLC ("Developer").

1. Scope of Work and Deliverables
Developer shall design, build, and deliver the customized enterprise software application according to the specifications set forth in Statement of Work No. 1. Developer warrants that all deliverables shall perform in material conformance with the agreed technical specifications.

2. Term and Auto-Renewal
This Agreement shall commence on the Effective Date and remain in effect for an initial period of twelve (12) months. Thereafter, this Agreement shall automatically renew for additional successive terms of one (1) year, unless either party delivers written notice of termination at least sixty (60) days prior to the expiration of the then-current term.

3. Payment Terms and Invoicing
Client shall pay Developer the milestone fees outlined in Schedule B within thirty (30) days of receiving an itemized invoice. Late undisputed payments shall accrue interest at a rate of 1.0% per month or the maximum statutory rate permitted by applicable law.

4. Acceptance and Inspection Period
Client shall have a period of ten (10) business days following receipt of each milestone deliverable to test and inspect the work. If Client fails to deliver written notice of rejection identifying specific defects within such ten-day period, the deliverable shall be deemed conclusively accepted.

5. Non-Compete and Market Restrictions
During the term of this Agreement and for a period of eighteen (18) months following termination, Developer shall not directly or indirectly develop, consult on, or deliver software applications that compete with Client's proprietary logistics platform anywhere in North America or the European Union.

6. Non-Solicitation of Personnel
For a period of twelve (12) months following completion of the project, neither party shall actively solicit, recruit, or hire any engineer, project manager, or key employee of the other party without prior written consent.

7. Unilateral Termination for Convenience
Client reserves the right to terminate this Agreement or any pending Statement of Work at any time, for any reason or no reason, in its sole discretion, upon seven (7) days written notice to Developer. In contrast, Developer may terminate only for uncured material breach by Client upon sixty (60) days written notice.

8. Limitation of Liability and Carveouts
Developer's total liability under this Agreement shall be capped at the aggregate fees paid by Client during the six (6) months preceding the incident. However, Developer's liability for breaches of confidentiality, intellectual property indemnity, and gross negligence shall be completely unlimited and uncapped.

9. Confidentiality and Trade Secrets
Each party agrees to safeguard the Confidential Information of the other party using the same degree of care it uses for its own sensitive data, and in no event less than reasonable care. Confidential Information shall not be disclosed to any third party without prior authorization.

10. Governing Law and Exclusive Jurisdiction
This Agreement shall be governed by, and construed in accordance with, the laws of the State of Delaware, United States, without giving effect to its conflict of laws principles. The parties submit to the exclusive jurisdiction of the state and federal courts located in New Castle County, Delaware.

11. Severability and Counterparts
If any provision of this Agreement is held to be invalid or unenforceable, such provision shall be severed and the remaining terms shall remain in full force and effect. This Agreement may be executed in counterparts.
"""


# 2. Commercial Vendor Services Agreement (Balanced Mixed Risk)
# - High Risk: Liquidated damages of $150k, Sole discretion termination by Buyer, Uncapped indemnity
# - Medium Risk: 90-day auto-renewal, Unannounced audit rights, Price freeze for 2 years
# - Low Risk: Standard warranties, New York governing law, Written notices, Force Majeure
COMMERCIAL_VENDOR_AGREEMENT = """COMMERCIAL VENDOR SERVICES AGREEMENT

This Commercial Vendor Agreement (the "Agreement") is entered into on 3 February 2026 between Omni Retail Enterprises ("Buyer") and Pacific Supply Distribution LLC ("Vendor").

1. Supply of Goods and Services
Vendor shall provide commercial inventory management and fulfillment services in accordance with purchase orders issued by Buyer from time to time. All goods supplied must comply with standard manufacturing specifications.

2. Term and Automatic Renewal
The initial term of this Agreement shall be two (2) years from the Effective Date. The Agreement will automatically renew for successive one (1) year terms unless Buyer or Vendor gives written notice of non-renewal at least ninety (90) days prior to the expiration of the current period.

3. Pricing and Payment
Prices for the initial two (2) years shall remain fixed as set forth in Exhibit A. Payment of approved invoices shall be due net forty-five (45) days following physical delivery and verification of items.

4. Right to Audit Books and Records
Buyer or its designated independent auditor shall have the right, upon forty-eight (48) hours notice, to enter Vendor's premises during business hours to inspect, review, and audit books, operational records, and quality control logs.

5. Liquidated Damages for Late Delivery
Timely delivery is of the essence. If Vendor fails to deliver shipments by the confirmed purchase order delivery date, Vendor shall pay Buyer liquidated damages of $150,000 for each week of delay, without prejudice to any other legal remedies Buyer may possess.

6. Unilateral Termination
Buyer may terminate this Agreement immediately without penalty upon written notice if Vendor experiences a material operational disruption or if Buyer determines in its sole discretion that continuation is economically undesirable.

7. Indemnification and Defense
Vendor agrees to indemnify, defend, and hold harmless Buyer, its officers, agents, and affiliates against any and all claims, property damage, third-party injuries, or product liabilities arising out of or resulting from the goods supplied under this Agreement.

8. Force Majeure
Neither party shall be liable for any failure or delay in performance resulting from acts of God, war, pandemic, government embargoes, or other events beyond the reasonable control of the impacted party, provided prompt written notice is given.

9. Governing Law
This Agreement and any dispute arising out of or related to it shall be governed by the laws of the State of New York, excluding its choice of law rules.
"""


# 3. Executive Employment & Non-Compete Agreement (Balanced Mixed Risk)
# - High Risk: 2-year post-employment non-compete, Immediate termination for cause without cure
# - Medium Risk: 60-day resignation notice, Intellectual property assignment of all concepts
# - Low Risk: Base salary and bonus schedule, Health insurance benefits, California governing law
EXECUTIVE_EMPLOYMENT_AGREEMENT = """EXECUTIVE EMPLOYMENT AGREEMENT

This Executive Employment Agreement (the "Agreement") is made effective 1 March 2026 between Pinnacle BioHealth Inc. (the "Company") and Dr. Elena Vance (the "Executive").

1. Title and Duties
The Company employs Executive as Senior Vice President of Research and Development. Executive shall devote her full business time, attention, and energies to the performance of duties and advancement of Company objectives.

2. Compensation and Bonus
Company shall pay Executive a base annual salary of $280,000, payable in semi-monthly installments. Executive is eligible to participate in the executive annual incentive bonus pool with a target bonus of 30% of base salary, based on milestone achievements.

3. Benefits and Paid Time Off
Executive shall be entitled to standard comprehensive medical, dental, and life insurance benefits, as well as four (4) weeks of paid vacation per calendar year in accordance with Company personnel policies.

4. Inventions and Intellectual Property Assignment
Executive hereby irrevocably assigns to Company all right, title, and interest in and to all inventions, algorithms, compounds, data discoveries, and copyrightable works created or conceived by Executive during the period of employment, whether during or outside normal working hours.

5. Termination by Company for Cause
The Company may terminate Executive's employment immediately for Cause without advance notice, severance, or opportunity to cure upon Executive's commission of gross misconduct, breach of fiduciary duty, or violation of Company policy.

6. Resignation and Notice Period
Executive may resign employment at any time by providing at least sixty (60) days advance written notice to the Board of Directors, during which period Company may require Executive to transition duties.

7. Restrictive Covenants and Non-Compete
During the employment term and for a period of two (2) years following separation of employment for any reason, Executive shall not directly or indirectly accept employment with, consult for, or hold an equity interest in any enterprise engaged in competing biomedical therapeutics.

8. Non-Solicitation of Talent and Clients
For eighteen (18) months following termination, Executive shall not recruit, solicit, or induce any employee, scientist, or consultant of the Company to terminate their employment relationship with the Company.

9. Governing Law
This Agreement shall be construed and interpreted in accordance with the laws of the State of California.
"""

if __name__ == "__main__":
    create_pdf(SOFTWARE_DEV_AGREEMENT, "mixed_risk_software_development_agreement.pdf", "Software Development Agreement")
    create_pdf(COMMERCIAL_VENDOR_AGREEMENT, "mixed_risk_commercial_vendor_agreement.pdf", "Commercial Vendor Agreement")
    create_pdf(EXECUTIVE_EMPLOYMENT_AGREEMENT, "mixed_risk_executive_employment_agreement.pdf", "Executive Employment Agreement")
