# ==========================================
# Customer Issue Detection
# ==========================================

def detect_customer_issues(review):

    review = review.lower()

    issues = []

    delivery_words = [
        "late",
        "delay",
        "delivery",
        "shipping",
        "arrived late",
        "not received"
    ]

    quality_words = [
        "bad quality",
        "poor quality",
        "broken",
        "damaged",
        "defective"
    ]

    support_words = [
        "support",
        "customer service",
        "response",
        "help"
    ]

    price_words = [
        "expensive",
        "costly",
        "price",
        "overpriced"
    ]

    payment_words = [
        "payment",
        "transaction",
        "refund"
    ]


    if any(word in review for word in delivery_words):
        issues.append("🚚 Delivery Issue")


    if any(word in review for word in quality_words):
        issues.append("📦 Product Quality Issue")


    if any(word in review for word in support_words):
        issues.append("☎️ Customer Support Issue")


    if any(word in review for word in price_words):
        issues.append("💰 Price Issue")


    if any(word in review for word in payment_words):
        issues.append("💳 Payment Issue")


    if not issues:
        issues.append("✅ No major issue detected")


    return issues