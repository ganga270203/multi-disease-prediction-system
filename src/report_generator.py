import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from datetime import datetime

def generate_pdf_report(patient_info, prediction_info, biomarkers):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    story = []
    styles = getSampleStyleSheet()

    # Title & Header
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=colors.HexColor('#1E3A8A'),
        spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#6B7280'),
        spaceAfter=15
    )

    story.append(Paragraph("OmniHealth AI — Clinical Diagnostic Report", title_style))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Confidential Medical Summary", subtitle_style))
    story.append(Spacer(1, 10))

    # Patient Details Table
    story.append(Paragraph("<b>Patient Demographics</b>", styles['Heading3']))
    patient_data = [
        ["Patient ID:", patient_info.get("id", "N/A"), "Full Name:", patient_info.get("name", "N/A")],
        ["Age:", str(patient_info.get("age", "N/A")), "Biological Sex:", patient_info.get("gender", "N/A")],
        ["Blood Group:", patient_info.get("blood_group", "N/A"), "Contact:", patient_info.get("contact", "N/A")]
    ]
    t_patient = Table(patient_data, colWidths=[100, 160, 100, 160])
    t_patient.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_patient)
    story.append(Spacer(1, 15))

    # Prediction Assessment Table
    story.append(Paragraph("<b>Diagnostic Evaluation</b>", styles['Heading3']))
    eval_color = colors.HexColor('#DC2626') if prediction_info['risk_level'] == 'High' else (colors.HexColor('#D97706') if prediction_info['risk_level'] == 'Moderate' else colors.HexColor('#16A34A'))
    eval_data = [
        ["Target Condition:", prediction_info.get("disease", "N/A")],
        ["Assessed Risk Probability:", f"{prediction_info.get('risk_score', 0):.1f}%"],
        ["Risk Classification:", prediction_info.get("risk_level", "N/A")],
        ["Clinical Finding:", prediction_info.get("result_text", "N/A")]
    ]
    t_eval = Table(eval_data, colWidths=[160, 360])
    t_eval.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#EFF6FF')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('TEXTCOLOR', (1, 2), (1, 2), eval_color),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_eval)
    story.append(Spacer(1, 15))

    # Biomarkers Submitted
    story.append(Paragraph("<b>Submitted Clinical Biomarkers</b>", styles['Heading3']))
    biomarker_rows = [["Biomarker Attribute", "Reported Value"]]
    for k, v in biomarkers.items():
        biomarker_rows.append([k.replace('_', ' ').title(), f"{v:.3f}"])
    
    t_bio = Table(biomarker_rows, colWidths=[260, 260])
    t_bio.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E293B')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_bio)
    story.append(Spacer(1, 20))

    # Sign-off footer
    disclaimer = Paragraph(
        "<i>Notice: This machine learning evaluation is an automated assistive screening tool and does not constitute a definitive medical diagnosis. Confirmatory clinical and laboratory follow-up is advised.</i>",
        styles['Italic']
    )
    story.append(disclaimer)

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()