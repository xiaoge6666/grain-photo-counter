# -*- coding: utf-8 -*-
"""
ASW-SPC 粘连米粒计数算法 —— numpy + opencv 纯实现（无 skimage/scipy 依赖）
来源: stars-spark/asw-spc-rice-counting (MIT License)
重写目的: 可在安卓 (Chaquopy) 上运行，去掉 skimage/scipy 交叉编译难题。
对外接口: count_rice(img_bgr) / label_image(img_bgr)
"""
import cv2
import numpy as np
import heapq

EPS = 1e-12
MAX_COVERAGE = 0.5
MIN_SINGLES = 3
MIN_AXIS_RATIO = 1.25
MAX_AXIS_RATIO = 8.0
MIN_SOLIDITY = 0.80
MAX_GRAIN_AREA_FRACTION = 0.01
MAX_GRAIN_EXTENT = 0.5
DEFAULT_CHANNELS = ("gray", "hsv_s")
SUBSTRATE_AREA_RATIO = 20.0
SPECK_AREA_RATIO = 0.3
MEDIAN_KSIZE = 1

NOISE_FLOOR_PX = 5
SHAPE_FLOOR_RATIO = 0.3
SINGLE_BAND = (0.65, 1.45)
AREA_MODE = "dominant"

BETA = 0.10
MIN_DEPTH_PX = 1.0
SEED_MERGE_RATIO = 0.6
TOUCH_AREA_RATIO = 1.2
TOUCH_SOLIDITY_MARGIN = 0.06
MIN_TRUSTED_MINOR_PX = 4.5
COARSE_TOUCH_EXCESS = 1.5
FOREIGN_AREA_RATIO = 3.0
FOREIGN_WIDTH_RATIO = 2.5

FRAGMENT_RATIO = 0.4
RESIDUAL_RATIO = 1.5
CONCAVITY_MARGIN = 0.03
SPECK_RATIO = 0.3
RESPLIT = True


# ==================== 工具（替代 skimage/scipy） ====================

def _label(mask):
    """连通域标记，等价 skimage.measure.label"""
    _, labels = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
    return labels


def multi_otsu(channel, classes=3):
    """多类 Otsu 阈值，等价 skimage.filters.threshold_multiotsu(classes=3)"""
    hist = cv2.calcHist([channel], [0], None, [256], [0, 256]).ravel().astype(np.float64)
    hist /= hist.sum() + EPS
    omega = np.cumsum(hist)
    mu_cum = np.cumsum(hist * np.arange(256))
    mu_total = mu_cum[-1]
    if classes == 3:
        best, best_t = 0.0, (0, 0)
        for t1 in range(1, 255):
            w1 = omega[t1 - 1]
            if w1 <= 0:
                continue
            mu1 = mu_cum[t1 - 1] / w1
            for t2 in range(t1 + 1, 256):
                w2 = omega[t2 - 1] - w1
                w3 = 1.0 - omega[t2 - 1]
                if w2 <= 0 or w3 <= 0:
                    continue
                mu2 = (mu_cum[t2 - 1] - mu_cum[t1 - 1]) / w2
                mu3 = (mu_total - mu_cum[t2 - 1]) / w3
                sb = w1 * (mu1 - mu_total) ** 2 + w2 * (mu2 - mu_total) ** 2 + w3 * (mu3 - mu_total) ** 2
                if sb > best:
                    best, best_t = sb, (t1, t2)
        return list(best_t)
    # fallback: 2类
    return [int(np.argmax((omega * (1 - omega)) * (mu_total * omega - mu_cum) ** 2 / (omega * (1 - omega) + EPS)))]


def clear_border(mask):
    """清除与图像边界连接的连通域，等价 skimage.segmentation.clear_border"""
    labels = _label(mask)
    border = set(labels[0, :]) | set(labels[-1, :]) | set(labels[:, 0]) | set(labels[:, -1])
    border.discard(0)
    out = (mask > 0).copy()
    for bl in border:
        out[labels == bl] = False
    return out


def _find_peaks(density):
    """局部极大值索引，等价 scipy.signal.find_peaks（简化版）"""
    peaks = []
    d = np.asarray(density)
    for i in range(1, len(d) - 1):
        if d[i] > d[i - 1] and d[i] >= d[i + 1]:
            peaks.append(i)
    return np.array(peaks, dtype=int)


def kde_density(log_areas, grid):
    """对数空间高斯 KDE，等价 scipy.stats.gaussian_kde"""
    n = len(log_areas)
    if n < 2:
        return np.ones_like(grid)
    bw = 1.06 * float(np.std(log_areas)) * n ** (-1.0 / 5.0) + EPS
    diff = grid[:, None] - log_areas[None, :]
    density = np.exp(-0.5 * (diff / bw) ** 2).sum(axis=1)
    density /= (n * bw * np.sqrt(2 * np.pi))
    return density


def _ws_watershed(dist, markers, mask):
    """priority-flood 分水岭，等价 skimage.segmentation.watershed(-dist, markers, mask)
    image = -dist，优先级升序（dist 峰值 = 谷物中心先处理，水向外蔓延）。"""
    image = -dist.astype(np.float64)
    h, w = dist.shape
    out = np.zeros((h, w), np.int32)
    pq = []
    rs, cs = np.nonzero(markers > 0)
    for r, c in zip(rs, cs):
        out[r, c] = int(markers[r, c])
        heapq.heappush(pq, (float(image[r, c]), int(r), int(c)))
    visited = (out > 0)
    while pq:
        val, r, c = heapq.heappop(pq)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                out[nr, nc] = out[r, c]
                visited[nr, nc] = True
                heapq.heappush(pq, (float(image[nr, nc]), nr, nc))
    return out


def h_maxima(dist, h):
    """h-maxima，严格等价 skimage.morphology.h_maxima"""
    image = dist.astype(np.float64)
    resolution = 2 * np.finfo(np.float32).resolution * np.abs(image)
    shifted = image - h - resolution
    kernel = np.ones((3, 3), np.uint8)
    out = shifted.copy()
    while True:
        prev = out.copy()
        out = np.minimum(cv2.dilate(out.astype(np.float32), kernel).astype(np.float64), image)
        if np.array_equal(out, prev):
            break
    residue = image - out
    return (residue >= h).astype(np.uint8)


def _fill_solidity(labels, rows, threshold=None):
    """计算各区域的 solidity = area / 凸包面积（替代 regionprops.solidity）"""
    for r in rows:
        if threshold is not None and r["area"] < threshold:
            continue
        mask = (labels == r["label"]).astype(np.uint8)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            r["solidity"] = float("nan")
            continue
        hull = cv2.convexHull(cnts[0])
        ha = cv2.contourArea(hull)
        r["solidity"] = float(r["area"] / ha) if ha > 0 else float("nan")


# ==================== preprocess ====================

def otsu_separability(channel):
    hist = cv2.calcHist([channel], [0], None, [256], [0, 256]).ravel()
    p = hist / (hist.sum() + EPS)
    levels = np.arange(256, dtype=np.float64)
    omega = np.cumsum(p)
    mu = np.cumsum(p * levels)
    mu_total = mu[-1]
    denom = omega * (1.0 - omega)
    sigma_b2 = np.where(denom > EPS, (mu_total * omega - mu) ** 2 / (denom + EPS), 0.0)
    sigma_total2 = float(np.sum(p * (levels - mu_total) ** 2))
    threshold = int(np.argmax(sigma_b2))
    eta = float(sigma_b2[threshold] / (sigma_total2 + EPS))
    return threshold, eta


def candidate_channels(img_bgr, median_ksize=None):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    channels = {
        "gray": cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY),
        "hsv_s": hsv[:, :, 1],
        "lab_a": lab[:, :, 1],
        "lab_b": lab[:, :, 2],
    }
    ksize = MEDIAN_KSIZE if median_ksize is None else median_ksize
    if ksize <= 1:
        return channels
    return {k: cv2.medianBlur(v, ksize) for k, v in channels.items()}


def clean_mask(mask, ksize=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)


def candidate_masks(img_bgr, channel_names=None, ksize=3, median_ksize=None):
    channels = {}
    for name, channel in candidate_channels(img_bgr, median_ksize=median_ksize).items():
        if channel_names is not None and name not in channel_names:
            continue
        channels[name] = channel

    out = []
    for name, ch in channels.items():
        t_otsu, eta = otsu_separability(ch)
        splits = {"otsu": [t_otsu]}
        try:
            splits["multiotsu"] = list(multi_otsu(ch, classes=3))
        except Exception:
            pass

        for method, thresholds in splits.items():
            bright = ch > thresholds[-1]
            dark = ch <= thresholds[0]
            for polarity, raw in (("bright", bright), ("dark", dark)):
                raw_mask = raw.astype(np.uint8) * 255
                out.append({
                    "channel_name": name, "channel": ch, "method": method,
                    "polarity": polarity, "thresholds": thresholds, "eta": eta,
                    "raw_mask": raw_mask, "mask": clean_mask(raw_mask, ksize=ksize),
                })
    return out


def local_contrast(channel, mask):
    fg = mask > 0
    if not fg.any():
        return 0.0
    inner = cv2.dilate(fg.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    ring = cv2.dilate(fg.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))).astype(bool) & ~inner.astype(bool)
    if not ring.any():
        return 0.0
    ch = channel.astype(np.float64)
    return float(abs(ch[fg].mean() - ch[ring].mean()) / (ch.std() + EPS))


def plausible_grain_population(calib, frame_shape, check_solidity=True):
    if calib["n_singles"] < MIN_SINGLES:
        return False
    if not MIN_AXIS_RATIO <= calib["axis_ratio0"] <= MAX_AXIS_RATIO:
        return False
    if check_solidity and calib["solidity0"] < MIN_SOLIDITY:
        return False
    height, width = frame_shape[:2]
    if calib["a0"] > MAX_GRAIN_AREA_FRACTION * height * width:
        return False
    if calib["major0"] > MAX_GRAIN_EXTENT * max(height, width):
        return False
    return True


def score_candidate(cand, frame_shape):
    mask = cand["mask"]
    interior = clear_border(mask > 0)
    coverage = float(interior.mean())
    if coverage > MAX_COVERAGE or coverage <= 0:
        return None
    try:
        calib = calibrate(mask, need_solidity=False)
    except ValueError:
        return None
    if not plausible_grain_population(calib, frame_shape, check_solidity=False):
        return None
    calib["coverage"] = coverage
    calib["contrast"] = local_contrast(cand["channel"], mask)
    calib["score"] = calib["n_singles"] * calib["a0"] * calib["contrast"]
    return calib


def drop_substrate(mask, a0):
    labels = _label(mask)
    out = (mask > 0).copy()
    border_labels = set(np.unique(labels[0, :])) | set(np.unique(labels[-1, :])) | \
                    set(np.unique(labels[:, 0])) | set(np.unique(labels[:, -1]))
    border_labels.discard(0)
    for bl in border_labels:
        if int(np.sum(labels == bl)) > SUBSTRATE_AREA_RATIO * a0:
            out[labels == bl] = False
    return out


def remove_specks(mask, a0):
    labels = _label(mask)
    out = (mask > 0).copy()
    for i in range(1, labels.max() + 1):
        if int(np.sum(labels == i)) < SPECK_AREA_RATIO * a0:
            out[labels == i] = False
    return out


def preprocess(img_bgr, channel_names=DEFAULT_CHANNELS, ksize=3, median_ksize=None):
    scored = []
    for cand in candidate_masks(img_bgr, channel_names=channel_names, ksize=ksize, median_ksize=median_ksize):
        calib = score_candidate(cand, img_bgr.shape)
        if calib is not None:
            scored.append((cand, calib))
    if not scored:
        raise ValueError("no usable binarisation candidate")

    chosen = None
    for cand, ranked in sorted(scored, key=lambda pair: -pair[1]["score"]):
        calib = calibrate(cand["mask"])
        if not plausible_grain_population(calib, img_bgr.shape):
            continue
        mask = drop_substrate(cand["mask"], calib["a0"])
        mask = remove_specks(mask, calib["a0"])
        try:
            final = calibrate(mask)
        except ValueError:
            continue
        if plausible_grain_population(final, img_bgr.shape):
            chosen = (cand, mask, final)
            break
    if chosen is None:
        raise ValueError("no binarisation candidate survived recalibration")
    cand, mask, calib = chosen
    return {
        "channel_name": cand["channel_name"], "channel": cand["channel"],
        "method": cand["method"], "polarity": cand["polarity"],
        "thresholds": cand["thresholds"], "eta": cand["eta"],
        "raw_mask": cand["raw_mask"], "mask": mask, "calib": calib,
    }


# ==================== calibrate ====================

def component_table(mask, shape_from=None, shape_sample=None, defer_solidity=False):
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return (labels, [], (lambda *a, **k: None)) if defer_solidity else (labels, [])

    index = np.arange(1, count)
    areas = stats[index, cv2.CC_STAT_AREA].astype(np.float64)
    boxes = np.stack([
        stats[index, cv2.CC_STAT_TOP],
        stats[index, cv2.CC_STAT_LEFT],
        stats[index, cv2.CC_STAT_TOP] + stats[index, cv2.CC_STAT_HEIGHT],
        stats[index, cv2.CC_STAT_LEFT] + stats[index, cv2.CC_STAT_WIDTH]], axis=1)

    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    flat_labels = labels.ravel()
    flat_dist = dist.ravel()
    order = np.argsort(flat_labels, kind='stable')
    sl = flat_labels[order]
    sd = flat_dist[order]
    change = np.where(sl[1:] != sl[:-1])[0] + 1
    starts = np.concatenate([[0], change])
    max_vals = np.maximum.reduceat(sd, starts)
    max_per_label = np.zeros(count, dtype=np.float64)
    max_per_label[sl[starts]] = max_vals
    widths = 2.0 * max_per_label[1:]

    rows_idx, cols_idx = np.nonzero(labels)
    flat = labels[rows_idx, cols_idx]
    n_px = np.bincount(flat, minlength=count).astype(np.float64)[1:]
    sum_x = np.bincount(flat, weights=cols_idx, minlength=count)[1:]
    sum_y = np.bincount(flat, weights=rows_idx, minlength=count)[1:]
    sum_xx = np.bincount(flat, weights=cols_idx * cols_idx, minlength=count)[1:]
    sum_yy = np.bincount(flat, weights=rows_idx * rows_idx, minlength=count)[1:]
    sum_xy = np.bincount(flat, weights=cols_idx * rows_idx, minlength=count)[1:]
    mean_x = sum_x / n_px
    mean_y = sum_y / n_px
    mu_xx = sum_xx / n_px - mean_x * mean_x
    mu_yy = sum_yy / n_px - mean_y * mean_y
    mu_xy = sum_xy / n_px - mean_x * mean_y
    spread = np.sqrt(np.maximum((mu_xx - mu_yy) ** 2 + 4.0 * mu_xy ** 2, 0.0))
    major = 4.0 * np.sqrt(np.maximum((mu_xx + mu_yy + spread) / 2.0, 0.0))
    minor = 4.0 * np.sqrt(np.maximum((mu_xx + mu_yy - spread) / 2.0, 0.0))

    rows = [
        {
            "label": int(index[i]),
            "area": float(areas[i]),
            "solidity": float("nan"),
            "major": float(major[i]),
            "minor": float(minor[i]),
            "width": float(widths[i]),
            "axis_ratio": float(major[i] / minor[i]) if minor[i] > 0 else np.inf,
            "bbox": tuple(int(v) for v in boxes[i]),
        }
        for i in range(index.size)
    ]

    def fill_solidity(threshold=None, sample=None):
        wanted = np.ones(areas.shape, bool) if threshold is None else areas >= threshold
        chosen = index[wanted]
        if sample is not None and chosen.size > sample:
            order = chosen[np.argsort(areas[wanted], kind="stable")]
            chosen = np.sort(order[np.linspace(0, order.size - 1, sample).astype(int)])
        for c in chosen:
            r = rows[int(c) - 1]
            mask_c = (labels == c).astype(np.uint8)
            cnts, _ = cv2.findContours(mask_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                hull = cv2.convexHull(cnts[0])
                hull_mask = np.zeros(mask_c.shape, np.uint8)
                cv2.fillPoly(hull_mask, [hull], 1)
                ha = int(hull_mask.sum())
                r["solidity"] = float(r["area"] / ha) if ha > 0 else float("nan")

    if defer_solidity:
        return labels, rows, fill_solidity
    fill_solidity(shape_from, shape_sample)
    return labels, rows


def estimate_single_area(areas, grid_size=512, mode=None):
    mode = AREA_MODE if mode is None else mode
    areas = np.asarray([a for a in areas if a >= NOISE_FLOOR_PX], dtype=np.float64)
    if areas.size == 0:
        raise ValueError("no foreground components above the noise floor")
    if areas.size < 5:
        return float(np.median(areas)), None

    log_areas = np.log(areas)
    grid = np.linspace(log_areas.min(), log_areas.max(), grid_size)
    density = kde_density(log_areas, grid)

    peak_idx = _find_peaks(density)
    if peak_idx.size == 0 or mode == "dominant":
        chosen = int(np.argmax(density))
    else:
        prominent = peak_idx[density[peak_idx] >= 0.3 * density.max()]
        chosen = prominent[0] if prominent.size else int(np.argmax(density))
    a0 = float(np.exp(grid[chosen]))

    band = areas[(areas >= SINGLE_BAND[0] * a0) & (areas <= SINGLE_BAND[1] * a0)]
    if band.size:
        a0 = float(band.mean())
    return a0, {"grid": grid, "density": density, "log_areas": log_areas}


def calibrate(mask, mode=None, need_solidity=True):
    labels, rows, fill_solidity = component_table(mask, defer_solidity=True)
    if not rows:
        raise ValueError("no foreground components above the noise floor")
    a0, kde_debug = estimate_single_area([r["area"] for r in rows], mode=mode)

    if need_solidity:
        fill_solidity(SHAPE_FLOOR_RATIO * a0)
    singles = [r for r in rows if SINGLE_BAND[0] * a0 <= r["area"] <= SINGLE_BAND[1] * a0]
    if not singles:
        if need_solidity:
            fill_solidity()
        singles = rows

    solidity0 = float(np.nanmedian([r["solidity"] for r in singles])) if need_solidity else float("nan")

    return {
        "labels": labels,
        "components": rows,
        "a0": a0,
        "r0": float(np.sqrt(a0 / np.pi)),
        "major0": float(np.median([r["major"] for r in singles])),
        "minor0": float(np.median([r["minor"] for r in singles])),
        "width0": float(np.median([r["width"] for r in singles])),
        "solidity0": solidity0,
        "axis_ratio0": float(np.median([r["axis_ratio"] for r in singles])),
        "n_components": len(rows),
        "n_singles": len(singles),
        "kde_debug": kde_debug,
    }


# ==================== segment ====================

def width_excess(component, calib):
    if calib["width0"] <= 0 or calib["a0"] <= 0:
        return 0.0
    reference = (component["width"] / calib["width0"]) ** 2
    return (component["area"] / calib["a0"]) / max(reference, 1e-6)


def distance_transform(mask):
    return cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)


def adaptive_markers(dist, minor0, beta=BETA):
    h = max(beta * minor0 / 2.0, MIN_DEPTH_PX)
    peaks = (h_maxima(dist, h) > 0).astype(np.uint8)
    radius = max(1, int(round(SEED_MERGE_RATIO * minor0 / 2.0)))
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    joined = _label(cv2.dilate(peaks, disc) > 0)
    return np.where(peaks > 0, joined, 0).astype(np.int32)


def is_foreign_object(component, calib):
    if component["area"] <= FOREIGN_AREA_RATIO * calib["a0"]:
        return False
    if calib["width0"] > 0 and component["width"] > FOREIGN_WIDTH_RATIO * calib["width0"]:
        return True
    return component["solidity"] >= calib["solidity0"]


def is_touching(component, calib):
    if component["area"] <= TOUCH_AREA_RATIO * calib["a0"]:
        return False
    if calib["minor0"] < MIN_TRUSTED_MINOR_PX:
        return width_excess(component, calib) > COARSE_TOUCH_EXCESS
    return component["solidity"] < calib["solidity0"] - TOUCH_SOLIDITY_MARGIN


def split_component(mask, calib, beta=None):
    beta = BETA if beta is None else beta
    dist = distance_transform(mask)
    markers = adaptive_markers(dist, calib["minor0"], beta=beta)
    if markers.max() <= 1:
        return (mask > 0).astype(np.int32), dist, markers
    return _ws_watershed(dist, markers, mask > 0), dist, markers


def split_by_count(mask, k, minor0):
    if k < 2:
        return None
    dist = distance_transform(mask)
    peaks = (dist == cv2.dilate(dist, np.ones((3, 3), np.uint8))) & (mask > 0)
    rows_p, cols_p = np.nonzero(peaks)
    order = np.argsort(-dist[rows_p, cols_p])
    spacing = SEED_MERGE_RATIO * minor0
    chosen = []
    for i in order:
        point = np.array([rows_p[i], cols_p[i]], dtype=np.float32)
        if all(np.linalg.norm(point - q) >= spacing for q in chosen):
            chosen.append(point)
            if len(chosen) == k:
                break
    if len(chosen) < k:
        return None
    markers = np.zeros(mask.shape, dtype=np.int32)
    for index, (r, c) in enumerate(chosen, start=1):
        markers[int(r), int(c)] = index
    return _ws_watershed(dist, markers, mask > 0)


# ==================== correct ====================

def _region_table(labels):
    """返回 [{label, area, solidity}]，直接按 label 唯一值遍历（替代 regionprops）"""
    regions = []
    for lab in np.unique(labels):
        if lab <= 0:
            continue
        area = float((labels == lab).sum())
        mask_i = (labels == lab).astype(np.uint8)
        solidity = float("nan")
        cnts, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            hull = cv2.convexHull(cnts[0])
            hull_mask = np.zeros(mask_i.shape, np.uint8)
            cv2.fillPoly(hull_mask, [hull], 1)
            ha = int(hull_mask.sum())
            solidity = area / ha if ha > 0 else float("nan")
        regions.append({"label": int(lab), "area": area, "solidity": solidity})
    return regions


def merge_fragments(labels, a0, fragment_ratio=FRAGMENT_RATIO):
    out = labels.copy()
    regions = {r["label"]: r for r in _region_table(out)}
    order = sorted(regions, key=lambda k: regions[k]["area"])

    for lab in order:
        area = int((out == lab).sum())
        if area == 0 or area >= fragment_ratio * a0:
            continue
        piece = out == lab
        ring = cv2.dilate(piece.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~piece
        neighbours = out[ring]
        neighbours = neighbours[neighbours > 0]
        if neighbours.size == 0:
            continue
        values, counts = np.unique(neighbours, return_counts=True)
        out[piece] = values[int(np.argmax(counts))]
    return out


def count_regions(labels, a0, solidity0=None, residual_ratio=RESIDUAL_RATIO,
                  fragment_ratio=FRAGMENT_RATIO, solidity_margin=CONCAVITY_MARGIN):
    total = 0
    details = []
    for region in _region_table(labels):
        ratio = region["area"] / a0
        if ratio < fragment_ratio:
            verdict, n = "fragment", 0
        elif ratio > residual_ratio and (solidity0 is None or region["solidity"] < solidity0 - solidity_margin):
            verdict, n = "residual", max(1, int(round(ratio)))
        else:
            verdict, n = "grain", 1
        total += n
        details.append({"label": region["label"], "ratio": ratio, "verdict": verdict, "n": n})
    return total, details


def correct_cluster(labels, a0, solidity0=None, residual_ratio=RESIDUAL_RATIO,
                    fragment_ratio=FRAGMENT_RATIO):
    merged = merge_fragments(labels, a0, fragment_ratio=fragment_ratio)
    count, details = count_regions(merged, a0, solidity0=solidity0,
                                   residual_ratio=residual_ratio, fragment_ratio=fragment_ratio)
    return count, merged, details


def resplit_residuals(merged, details, a0, minor0, solidity0=None,
                      residual_ratio=RESIDUAL_RATIO, fragment_ratio=FRAGMENT_RATIO):
    out = merged.copy()
    next_label = int(out.max()) + 1
    for detail in details:
        if detail["verdict"] != "residual":
            continue
        region = out == detail["label"]
        pieces = split_by_count(region.astype(np.uint8), detail["n"], minor0)
        if pieces is None or (region & (pieces == 0)).any():
            continue
        areas = [int((pieces == i).sum()) for i in range(1, detail["n"] + 1)]
        if min(areas) < fragment_ratio * a0 or max(areas) > residual_ratio * a0:
            continue
        for i in range(1, detail["n"] + 1):
            out[pieces == i] = next_label
            next_label += 1
    count, details = count_regions(out, a0, solidity0=solidity0,
                                   residual_ratio=residual_ratio, fragment_ratio=fragment_ratio)
    return count, out, details


# ==================== counter ====================

def count_rice(img_bgr, beta=None, residual_ratio=RESIDUAL_RATIO,
               fragment_ratio=FRAGMENT_RATIO, pre=None, return_debug=False):
    pre = pre if pre is not None else preprocess(img_bgr)
    calib = pre["calib"]
    labels, a0 = calib["labels"], calib["a0"]

    total = 0
    debug = {"clusters": [], "n_isolated": 0, "n_clusters": 0, "n_foreign": 0}

    for component in calib["components"]:
        if component["area"] < SPECK_RATIO * a0:
            continue
        if is_foreign_object(component, calib):
            debug["n_foreign"] += 1
            continue
        if not is_touching(component, calib):
            total += 1
            debug["n_isolated"] += 1
            continue

        r0, c0, r1, c1 = component["bbox"]
        pad = 2
        r0, c0 = max(0, r0 - pad), max(0, c0 - pad)
        r1, c1 = min(labels.shape[0], r1 + pad), min(labels.shape[1], c1 + pad)
        mask = (labels[r0:r1, c0:c1] == component["label"]).astype(np.uint8) * 255

        ws, dist, markers = split_component(mask, calib, beta=beta)
        count, merged, details = correct_cluster(ws, a0, solidity0=calib["solidity0"],
                                                 residual_ratio=residual_ratio,
                                                 fragment_ratio=fragment_ratio)
        if RESPLIT:
            count, merged, details = resplit_residuals(
                merged, details, a0, calib["minor0"], solidity0=calib["solidity0"],
                residual_ratio=residual_ratio, fragment_ratio=fragment_ratio)
        total += count
        debug["n_clusters"] += 1
        if return_debug:
            debug["clusters"].append({
                "component": component, "bbox": (r0, c0, r1, c1), "mask": mask,
                "dist": dist, "markers": markers, "watershed": ws, "merged": merged,
                "count": count, "details": details,
            })

    return (total, pre, debug) if return_debug else total


def label_image(img_bgr=None, pre=None, **kwargs):
    total, pre, debug = count_rice(img_bgr, pre=pre, return_debug=True, **kwargs)
    calib = pre["calib"]
    source = calib["labels"]
    instances = np.zeros(source.shape, dtype=np.int32)
    info = {}
    next_label = 1

    for component in calib["components"]:
        if component["area"] < SPECK_RATIO * calib["a0"]:
            continue
        if is_foreign_object(component, calib):
            instances[source == component["label"]] = next_label
            info[next_label] = {"kind": "foreign", "count": 0}
            next_label += 1
        elif not is_touching(component, calib):
            instances[source == component["label"]] = next_label
            info[next_label] = {"kind": "grain", "count": 1}
            next_label += 1

    for cluster in debug["clusters"]:
        r0, c0, r1, c1 = cluster["bbox"]
        merged = cluster["merged"]
        verdicts = {d["label"]: d for d in cluster["details"]}
        window = instances[r0:r1, c0:c1]
        for local_label in np.unique(merged):
            if local_label == 0:
                continue
            detail = verdicts.get(local_label, {"verdict": "grain", "n": 1})
            if detail["n"] == 0:
                continue
            window[merged == local_label] = next_label
            info[next_label] = {"kind": detail["verdict"], "count": detail["n"]}
            next_label += 1

    return instances, info, total, pre
