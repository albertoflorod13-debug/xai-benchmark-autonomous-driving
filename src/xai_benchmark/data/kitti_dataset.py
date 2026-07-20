"""KITTI dataset utilities.
"""

KITTI_CLASSES = {
    0: "car",
    1: "van",
    2: "truck",
    3: "pedestrian",
    4: "person_sitting",
    5: "cyclist",
    6: "tram",
    7: "misc",
}


def load_kitti_yolo_bboxes(txt_file_path, img_height, img_width, target_class=None,
                            return_class=False, box_format="polygon"):
    """Load KITTI labels in standard YOLO format (class_id cx cy w h, normalized).

    box_format="polygon" (por defecto): formato de 4 esquinas que espera D-CRISP,
        [(x1,y1), (x2,y1), (x2,y2), (x1,y2)] en pixeles absolutos.
    box_format="xyxy": tupla simple (x1,y1,x2,y2), para emparejamiento por IoU.
    return_class=True antepone class_id a cada entrada: (class_id, *box) en vez
    de solo box -- necesario para comparar clase predicha vs. clase real
    (uncertainty/tta.py).
    """
    bboxes = []

    with open(txt_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        parts = list(map(float, line.strip().split()))
        if len(parts) < 5:
            continue
        class_id = int(parts[0])
        if target_class is not None and class_id != target_class:
            continue

        cx, cy, w, h = parts[1], parts[2], parts[3], parts[4]

        x1 = (cx - w / 2) * img_width
        y1 = (cy - h / 2) * img_height
        x2 = (cx + w / 2) * img_width
        y2 = (cy + h / 2) * img_height

        if box_format == "xyxy":
            box = (x1, y1, x2, y2)
        else:
            box = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

        bboxes.append((class_id, *box) if return_class else box)

    return bboxes
