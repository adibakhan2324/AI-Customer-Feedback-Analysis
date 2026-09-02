def generate_recommendation(review, sentiment):

    review = review.lower()

    recommendations = []

    # ---------------------------------
    # Negative Review Recommendations
    # ---------------------------------

    if "Negative" in sentiment:

        if "battery" in review or "charging" in review:
            recommendations.append(
                "Improve battery performance and optimize power consumption"
            )

        if "slow" in review or "lag" in review:
            recommendations.append(
                "Improve system performance and reduce response time"
            )

        if "delivery" in review or "late" in review or "delay" in review:
            recommendations.append(
                "Optimize delivery process and logistics management"
            )

        if "price" in review or "expensive" in review:
            recommendations.append(
                "Review pricing strategy and provide better offers"
            )

        if "support" in review or "service" in review:
            recommendations.append(
                "Improve customer support response quality"
            )

        if "quality" in review or "broken" in review or "damaged" in review:
            recommendations.append(
                "Improve product quality and quality-control processes"
            )

        # If negative but no specific issue was detected
        if len(recommendations) == 0:
            recommendations.append(
                "Investigate the source of customer dissatisfaction and improve the overall customer experience"
            )

    # ---------------------------------
    # Positive Review Recommendations
    # ---------------------------------

    elif "Positive" in sentiment:

        recommendations.append(
            "Continue maintaining product quality and customer satisfaction"
        )

    # ---------------------------------
    # Neutral Review Recommendations
    # ---------------------------------

    else:

        recommendations.append(
            "Collect more customer feedback to identify improvement opportunities"
        )

    return recommendations