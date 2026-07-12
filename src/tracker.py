def compare_visits(previous_area_pixels, current_area_pixels):
    previous_area_pixels = float(previous_area_pixels)
    current_area_pixels = float(current_area_pixels)

    if previous_area_pixels <= 0:
        return {
            "status": "NOT AVAILABLE",
            "message": "Previous wound area is not available or zero.",
            "change_pixels": None,
            "change_percent": None,
        }

    change_pixels = current_area_pixels - previous_area_pixels
    change_percent = (change_pixels / previous_area_pixels) * 100

    if change_percent < -10:
        status = "IMPROVING"
        message = "Predicted wound area decreased compared with the previous visit."
    elif change_percent > 10:
        status = "WORSENING"
        message = "Predicted wound area increased compared with the previous visit."
    else:
        status = "STABLE"
        message = "Predicted wound area changed only slightly compared with the previous visit."

    return {
        "status": status,
        "message": message,
        "change_pixels": int(change_pixels),
        "change_percent": round(change_percent, 2),
    }