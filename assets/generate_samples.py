import sys
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from services.mock_data import DEMO_CONTRACT_TEXT, SAMPLE_RENTAL_TEXT

ss = getSampleStyleSheet()

def build_pdf(text: str, filename: str, title: str):
    path = Path(__file__).resolve().parent / filename
    story = []
    for block in text.strip().split("\n\n"):
        lines = block.strip().split("\n")
        if len(lines) == 1 and (lines[0].isupper() or "AGREEMENT" in lines[0]):
            story.append(Paragraph(lines[0], ss["Title"]))
            continue
        if lines[0][:1].isdigit():
            story.append(Paragraph(f"<b>{lines[0]}</b>", ss["Heading4"]))
            lines = lines[1:]
        for ln in lines:
            story.append(Paragraph(ln, ss["BodyText"]))
        story.append(Spacer(1, 6))
    SimpleDocTemplate(str(path), pagesize=A4, title=title).build(story)
    print(f"Created {path}")

NDA_TEXT = """MUTUAL NON-DISCLOSURE AGREEMENT

This Mutual Non-Disclosure Agreement (the "Agreement") is entered into on 15 January 2026 by and between Apex Cloud Inc. ("Discloser") and Beacon Health Ltd. ("Recipient").

1. Confidential Information
"Confidential Information" refers to all technical, financial, and business data disclosed by either party, whether orally or in tangible form, marked or reasonably understood to be confidential.

2. Non-Disclosure Obligations
Recipient agrees to hold Discloser's Confidential Information in strict confidence and shall not disclose such information to any third party without prior written authorization.

3. Exclusions from Confidentiality
Confidential Information does not include information that: (a) is or becomes publicly known through no breach of this Agreement; (b) was already known to Recipient prior to disclosure; or (c) is independently developed without reference to Discloser's Confidential Information.

4. Perpetual Term and Survival
The confidentiality obligations under this Agreement shall survive in perpetuity and shall never expire following termination of business relations between the parties.

5. Unilateral Indemnification
Recipient shall unconditionally indemnify, defend, and hold harmless Discloser and its affiliates against any and all losses, damages, or attorney fees arising out of any unauthorized disclosure or use of Confidential Information.

6. Injunctive Relief and Liquidated Damages
In the event of any breach, Discloser shall be entitled to seek immediate injunctive relief without the necessity of posting a bond. In addition, Recipient agrees to pay liquidated damages of $100,000 for each substantiated disclosure violation.

7. Non-Solicitation of Employees
For a period of two (2) years from the Effective Date, Recipient shall not directly or indirectly solicit, recruit, or hire any officer, employee, or independent contractor of Discloser.

8. Governing Law and Jurisdiction
This Agreement shall be construed and governed in accordance with the laws of the State of Delaware, United States, without regard to conflicts of law principles.
"""

SAAS_TEXT = """SOFTWARE AS A SERVICE (SAAS) AGREEMENT

This SaaS Agreement is made effective as of 10 February 2026 between CloudForge Systems Inc. ("Provider") and Global Retail Logistics LLC ("Customer").

1. License and Subscription Grant
Provider grants Customer a non-exclusive, non-transferable, revocable subscription to access and use the CloudForge Enterprise Platform during the Subscription Term solely for internal business operations.

2. Fees, Billing, and Price Increases
Customer agrees to pay the annual subscription fee of $45,000 in advance within thirty (30) days of invoice. Provider reserves the right to increase annual fees by up to 20% upon thirty (30) days notice prior to renewal.

3. Automatic Renewal
This Agreement shall automatically renew for successive terms of two (2) years unless Customer delivers written notice of termination at least one hundred and twenty (120) days prior to the expiration of the current term.

4. Unilateral Modification of Terms
Provider reserves the right to modify the features, functionality, and terms of service at any time without prior notice or consent of Customer.

5. Service Level Agreement and Disclaimers
The platform is provided "AS IS" and "AS AVAILABLE". Provider makes no warranties, express or implied, including warranties of merchantability, fitness for a particular purpose, or uninterrupted availability.

6. Limitation of Liability
Provider's total aggregate liability for all claims arising under this Agreement shall be strictly capped at $1,000. Customer agrees that Provider shall not be liable for any consequential, incidental, indirect, or lost profit damages.

7. Intellectual Property and Data Ownership
All data, analytics, enhancements, and models derived from Customer's use of the software shall become the exclusive intellectual property of Provider.

8. Termination for Convenience
Provider may terminate this Agreement at any time in its sole discretion upon seven (7) days written notice. Customer may terminate only upon 180 days advance written notice with full payment of the remaining term fees.

9. Governing Law
This Agreement shall be governed by the laws of the State of New York.
"""

if __name__ == "__main__":
    build_pdf(DEMO_CONTRACT_TEXT, "sample_consulting_agreement.pdf", "Consulting Services Agreement")
    build_pdf(NDA_TEXT, "sample_mutual_nda.pdf", "Mutual Non-Disclosure Agreement")
    build_pdf(SAAS_TEXT, "sample_saas_agreement.pdf", "SaaS Agreement")
