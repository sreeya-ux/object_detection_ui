import os
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from io import BytesIO
import base64
from datetime import datetime

def generate_asset_pdf(asset_data):
    """
    Generates a professional PDF report for an asset.
    asset_data: dict containing 'id', 'worker_name', 'status', 'timestamp', 'asset_class', 'voltage', 'reason', 'images'
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=10,
        fontName='Helvetica-Bold'
    )
    
    label_style = ParagraphStyle(
        'LabelStyle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor("#64748b"),
        textTransform='uppercase',
        fontName='Helvetica-Bold'
    )

    story = []

    # --- Header Segment ---
    story.append(Paragraph("ASAKTA VISION AI", styles['Heading4']))
    story.append(Paragraph(f"INFRASTRUCTURE INSPECTION REPORT", title_style))
    story.append(Paragraph(f"Asset ID: {asset_data['id']}", styles['Normal']))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 20))

    # --- Summary Table ---
    summary_data = [
        [Paragraph("<b>WORKER</b>", label_style), asset_data['worker_name']],
        [Paragraph("<b>CLASSIFICATION</b>", label_style), asset_data['asset_class']],
        [Paragraph("<b>VOLTAGE</b>", label_style), asset_data['voltage']],
        [Paragraph("<b>TIMESTAMP</b>", label_style), asset_data['timestamp']],
        [Paragraph("<b>STATUS</b>", label_style), asset_data['status']]
    ]
    
    t = Table(summary_data, colWidths=[120, 350])
    t.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 0), (0, -1), colors.whitesmoke),
        ('PADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(t)
    story.append(Spacer(1, 20))

    # --- Reasoning ---
    story.append(Paragraph("<b>DIAGNOSTIC SUMMARY</b>", label_style))
    story.append(Paragraph(asset_data['reason'], styles['Italic']))
    story.append(Spacer(1, 30))

    # --- Detailed Image Analysis ---
    story.append(Paragraph("IMAGE EVIDENCE GALLERY", styles['Heading2']))
    story.append(Spacer(1, 10))

    for i, img in enumerate(asset_data['images']):
        story.append(Paragraph(f"Frame #{i+1} Analysis", styles['Heading3']))
        
        # Convert B64 to Temp Image
        img_data = base64.b64decode(img['image_b64'])
        img_buffer = BytesIO(img_data)
        
        # Add Image
        report_img = Image(img_buffer, width=450, height=300, kind='proportional')
        story.append(report_img)
        story.append(Spacer(1, 15))
        
        # Detections Table
        story.append(Paragraph("Components Identified:", label_style))
        det_rows = [["Label", "Confidence", "Type"]]
        for det in img['detections']:
            det_rows.append([
                det['label'], 
                f"{det.get('confidence', 1.0)*100:.1f}%", 
                "Manual" if det.get('manual') else "AI"
            ])
        
        dt = Table(det_rows, colWidths=[150, 100, 100])
        dt.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#3b82f6")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
        ]))
        story.append(dt)
        story.append(Spacer(1, 30))
        
        if (i + 1) % 2 == 0: # Page break every 2 images
             from reportlab.platypus import PageBreak
             story.append(PageBreak())

    doc.build(story)
    buffer.seek(0)
    return buffer

def generate_asset_excel(asset_data):
    """
    Generates a summary Excel report for an asset.
    """
    df_data = []
    for i, img in enumerate(asset_data['images']):
        for det in img['detections']:
            df_data.append({
                "Asset_ID": asset_data['id'],
                "Worker": asset_data['worker_name'],
                "Timestamp": asset_data['timestamp'],
                "Frame_No": i + 1,
                "Component": det['label'],
                "Confidence": det.get('confidence', 1.0),
                "Category": asset_data['asset_class'],
                "Voltage": asset_data['voltage']
            })
    
    df = pd.DataFrame(df_data)
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Detection_Log')
    
    buffer.seek(0)
    return buffer

def generate_global_pdf(assets_list):
    """
    Generates a global summary PDF for multiple assets.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'TitleStyle', parent=styles['Heading1'], fontSize=24,
        textColor=colors.HexColor("#1e293b"), spaceAfter=10, fontName='Helvetica-Bold'
    )
    
    story = []
    story.append(Paragraph("ASAKTA VISION AI", styles['Heading4']))
    story.append(Paragraph("GLOBAL INFRASTRUCTURE REPORT", title_style))
    story.append(Paragraph(f"Total Assets: {len(assets_list)}", styles['Normal']))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 20))

    if not assets_list:
        story.append(Paragraph("No assets found in the database.", styles['Normal']))
        doc.build(story)
        buffer.seek(0)
        return buffer

    # Table Header
    data = [["Asset ID", "Worker", "Status", "Class", "Timestamp"]]
    
    for a in assets_list:
        short_id = a['id'][:8] + "..." if len(a['id']) > 8 else a['id']
        data.append([
            short_id,
            a['worker_name'],
            a['status'],
            a['asset_class'] or "N/A",
            a['timestamp']
        ])
    
    t = Table(data, colWidths=[100, 100, 60, 100, 140])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#3b82f6")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
    ]))
    
    story.append(t)
    doc.build(story)
    buffer.seek(0)
    return buffer

def generate_global_excel(assets_list):
    """
    Generates a global detailed Excel report.
    """
    df_data = []
    for a in assets_list:
        if not a.get('images'):
            df_data.append({
                "Asset_ID": a['id'],
                "Worker": a['worker_name'],
                "Timestamp": a['timestamp'],
                "Status": a['status'],
                "Category": a['asset_class'],
                "Voltage": a['voltage'],
                "Component": "None",
                "Confidence": "",
                "Manual_Edit": ""
            })
            continue
            
        for i, img in enumerate(a['images']):
            for det in img.get('detections', []):
                df_data.append({
                    "Asset_ID": a['id'],
                    "Worker": a['worker_name'],
                    "Timestamp": a['timestamp'],
                    "Status": a['status'],
                    "Category": a['asset_class'],
                    "Voltage": a['voltage'],
                    "Frame_No": i + 1,
                    "Component": det.get('label', 'UNKNOWN'),
                    "Confidence": det.get('confidence', 1.0),
                    "Manual_Edit": det.get('manual', False)
                })
    
    if not df_data:
        df_data.append({"Info": "No data found"})
        
    df = pd.DataFrame(df_data)
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Global_Detection_Log')
    
    buffer.seek(0)
    return buffer
