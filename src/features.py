import cv2
import numpy as np


def extract_lesion_features(pred_mask, prob_map=None, min_area_pixels=20):
    mask = pred_mask.astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    lesions = []
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])

        if area < min_area_pixels:
            continue

        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[label_id]

        lesion_mask = labels == label_id

        confidence = None
        if prob_map is not None:
            confidence = float(prob_map[lesion_mask].mean())

        lesions.append({
            "lesion_id": len(lesions) + 1,
            "area_pixels": area,
            "bbox": [x, y, w, h],
            "centroid": [round(float(cx), 2), round(float(cy), 2)],
            "mean_confidence": round(confidence, 4) if confidence is not None else None,
        })

    total_area = int(sum(item["area_pixels"] for item in lesions))

    return {
        "number_of_lesions": len(lesions),
        "total_area_pixels": total_area,
        "lesions": lesions,
    }