def generate_business_report(
    sentiment,
    issues
):

    report = ""


    report += "<b>Customer Sentiment:</b><br>"
    report += sentiment + "<br><br>"


    report += "<b>Detected Issues:</b><br>"

    for issue in issues:
        report += "• " + issue + "<br>"


    report += "<br><b>Business Impact:</b><br>"

    if "🚚 Delivery Issue" in issues:
        report += "Delivery problems may reduce customer satisfaction.<br>"


    if "☎️ Customer Support Issue" in issues:
        report += "Poor support experience may affect customer loyalty.<br>"


    if "📦 Product Quality Issue" in issues:
        report += "Quality problems may increase negative reviews.<br>"


    report += "<br><b>Recommended Actions:</b><br>"
    report += "• Improve customer experience<br>"
    report += "• Monitor repeated complaints<br>"
    report += "• Take corrective business actions<br>"


    return report