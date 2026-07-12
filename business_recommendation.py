def generate_recommendation(review):

    review = review.lower()

    recommendations = []

    if "battery" in review or "charging" in review:
        recommendations.append(
            "Improve battery performance and optimize power consumption"
        )

    if "slow" in review or "lag" in review:
        recommendations.append(
            "Improve system performance and reduce response time"
        )

    if "delivery" in review or "late" in review:
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

    if len(recommendations) == 0:
        recommendations.append(
            "Continue maintaining product quality and customer satisfaction"
        )

    return recommendations