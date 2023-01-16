from django.http import FileResponse
import io
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import letter

def pdfgen(pregnancy, glucose, bp, skin, insulin, bmi, dpf, age):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, bottomup=0)
    textob = c.beginText()
    textob.setTextOrigin(inch, inch)
    textob.setFont('Helvetica',14)
    lines = [
    "Pregnancy:"+str(pregnancy),
    "This is line 2",
    "This is line 3",
    ]
    for line in lines:
        textob.textLine(line)

    c.drawText(textob)
    c.showPage()
    c.save()
    buf.seek(0)
    # return HttpResponse("pdf")
    return FileResponse(buf, as_attachment=True, filename='venue.pdf')
