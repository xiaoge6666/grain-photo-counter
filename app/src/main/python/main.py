import numpy as np
import cv2
import asw_core

MAX_SIDE = 720  # 降分辨率加速


def count_and_label(data):
    """计数并返回标注图。data 是 Java byte[] 转来的 bytes。

    返回 (粒数, 标注图JPEG字节)。粒数 < 0 表示出错。
    """
    try:
        buf = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            return (-1, b'')

        h, w = img.shape[:2]
        side = max(h, w)
        if side > MAX_SIDE:
            scale = MAX_SIDE / side
            img = cv2.resize(img, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)

        instances, info, total, pre = asw_core.label_image(img)

        # 画标注
        vis = img.copy()
        num = 0
        for lbl in sorted(info.keys()):
            meta = info[lbl]
            if meta["count"] == 0:
                continue
            mask = (instances == lbl)
            ys, xs = np.where(mask)
            if len(xs) == 0:
                continue
            xmin, xmax = xs.min(), xs.max()
            ymin, ymax = ys.min(), ys.max()
            cnt = meta["count"]
            if cnt == 1:
                num += 1
                color = (0, 255, 0)
                label = str(num)
            else:
                color = (0, 165, 255)
                label = "%d粒" % cnt
            cv2.rectangle(vis, (xmin, ymin), (xmax, ymax), color, 1)
            cx = (xmin + xmax) // 2
            cy = (ymin + ymax) // 2
            if cy > 5:
                cv2.putText(vis, label, (max(2, cx - 8), cy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        ok, encoded = cv2.imencode('.jpg', vis)
        if not ok:
            return (int(total), b'')
        return (int(total), encoded.tobytes())
    except Exception:
        return (-2, b'')
