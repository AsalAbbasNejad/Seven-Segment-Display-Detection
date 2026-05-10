import sys
import cv2
import numpy as np


BROKEN_INDEX_BASE = 0
BASE_MAX_SIZE = 1200
EXTRA_MAX_SIZE = 1500

SEG_ORDER = "abcdefg"

SYMBOL_TO_SEG = {
    "0": "abcdef", "1": "bc", "2": "abdeg", "3": "abcdg",
    "4": "bcfg", "5": "acdfg", "6": "acdefg", "7": "abc",
    "8": "abcdefg", "9": "abcdfg", "A": "abcefg",
    "b": "cdefg", "C": "adef", "d": "bcdeg",
    "E": "adefg", "F": "aefg",
    "PLUS_L": "bcg", "MINUS": "g", "MUL": "a", "EQ": "dg",
}

SEG_TO_SYMBOL = {"".join(sorted(v)): k for k, v in SYMBOL_TO_SEG.items()}


def hex_value(ch):
    if ch in "0123456789":
        return int(ch)
    return {"A": 10, "b": 11, "C": 12, "d": 13, "E": 14, "F": 15}.get(ch)


def parse_number(tokens):
    if not tokens:
        return None
    value = 0
    for t in tokens:
        d = hex_value(t)
        if d is None:
            return None
        value = value * 16 + d
    return value


def convert_symbols(raw):
    out = []
    i = 0
    while i < len(raw):
        if raw[i] == "PLUS_L":
            if i + 1 < len(raw) and raw[i + 1] == "MINUS":
                out.append("+")
                i += 2
            else:
                return None
        elif raw[i] == "MINUS":
            out.append("-")
            i += 1
        elif raw[i] == "MUL":
            out.append("*")
            i += 1
        elif raw[i] == "EQ":
            out.append("=")
            i += 1
        else:
            out.append(raw[i])
            i += 1
    return out


def evaluate(tokens):
    if tokens is None or tokens.count("=") != 1:
        return False

    eq = tokens.index("=")
    left = tokens[:eq]
    right = tokens[eq + 1:]

    rhs = parse_number(right)
    if rhs is None:
        return False

    nums, ops, cur = [], [], []

    for t in left:
        if t in "+-*":
            n = parse_number(cur)
            if n is None:
                return False
            nums.append(n)
            ops.append(t)
            cur = []
        else:
            cur.append(t)

    n = parse_number(cur)
    if n is None:
        return False
    nums.append(n)

    i = 0
    while i < len(ops):
        if ops[i] == "*":
            nums[i] *= nums[i + 1]
            del nums[i + 1]
            del ops[i]
        else:
            i += 1

    result = nums[0]
    for op, val in zip(ops, nums[1:]):
        if op == "+":
            result += val
        elif op == "-":
            result -= val

    return result == rhs


def raw_symbols(observed):
    raw = []
    for s in observed:
        key = "".join(sorted(s))
        if key not in SEG_TO_SYMBOL:
            return None
        raw.append(SEG_TO_SYMBOL[key])
    return raw


def find_broken(observed):
    for i in range(len(observed)):
        original = set(observed[i])
        for seg in SEG_ORDER:
            fixed = set(original)
            if seg in fixed:
                fixed.remove(seg)
            else:
                fixed.add(seg)

            candidate = observed[:]
            candidate[i] = "".join(sorted(fixed))

            raw = raw_symbols(candidate)
            if raw is None:
                continue

            if evaluate(convert_symbols(raw)):
                return i + BROKEN_INDEX_BASE, seg, True

    return BROKEN_INDEX_BASE, "a", False


def candidate_score(observed, confidence=0.0):
    if not (1 <= len(observed) <= 16):
        return (-999,)

    known = sum(1 for s in observed if "".join(sorted(s)) in SEG_TO_SYMBOL)
    _, _, valid = find_broken(observed)

    raw = raw_symbols(observed)
    eq_bonus = 0
    plus_bonus = 0
    structure_bonus = 0

    if raw is not None:
        if raw.count("EQ") == 1:
            eq_bonus = 2

        for i in range(len(raw) - 1):
            if raw[i] == "PLUS_L" and raw[i + 1] == "MINUS":
                plus_bonus += 1

        tokens = convert_symbols(raw)
        if tokens is not None and tokens.count("=") == 1:
            eq = tokens.index("=")
            if 0 < eq < len(tokens) - 1:
                structure_bonus = 1

    return (
        100 if valid else 0,
        known,
        eq_bonus,
        structure_bonus,
        plus_bonus,
        len(observed),
        round(confidence, 5),
    )


def resize_image(img, max_size):
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_size:
        return img
    scale = max_size / m
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def background_color(img):
    sample = img[::10, ::10].reshape(-1, 3)
    q = (sample // 8).astype(np.int16)
    colors, counts = np.unique(q, axis=0, return_counts=True)
    return (colors[np.argmax(counts)] * 8 + 4).astype(np.float32)


def clean_mask(mask, min_area):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    clean = np.zeros_like(mask)
    areas = []

    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if a >= min_area:
            areas.append(a)

    if not areas:
        return clean

    med = np.median(areas)
    lo = max(min_area, med * 0.02)
    hi = med * 300

    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if lo <= a <= hi:
            clean[labels == i] = 255

    return clean


def make_masks(img):
    bg = background_color(img)
    diff = np.linalg.norm(img.astype(np.float32) - bg[None, None, :], axis=2)

    display_mask = (diff > 25).astype(np.uint8) * 255

    b, g, r = cv2.split(img)
    b = b.astype(np.int16)
    g = g.astype(np.int16)
    r = r.astype(np.int16)

    red = (r > 115) & (r > g + 40) & (r > b + 40)
    green = (g > 115) & (g > r + 40) & (g > b + 40)
    blue = (b > 115) & (b > r + 40) & (b > g + 40)

    active_mask = ((red | green | blue) & (diff > 40)).astype(np.uint8) * 255

    display_mask = cv2.morphologyEx(display_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    display_mask = cv2.morphologyEx(display_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    active_mask = cv2.morphologyEx(active_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    active_mask = cv2.morphologyEx(active_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    display_mask = clean_mask(display_mask, 3)
    active_mask = clean_mask(active_mask, 5)

    return display_mask, active_mask


def mask_points(mask):
    y, x = np.where(mask > 0)
    return np.column_stack([x, y]).astype(np.float32)


def pca_axes(points):
    center = points.mean(axis=0)
    q = points - center
    cov = np.cov(q.T)

    vals, vecs = np.linalg.eigh(cov)

    x_axis = vecs[:, np.argmax(vals)].astype(np.float32)
    x_axis /= np.linalg.norm(x_axis)

    y_axis = np.array([-x_axis[1], x_axis[0]], dtype=np.float32)

    return center.astype(np.float32), x_axis, y_axis


def project(points, center, x_axis, y_axis):
    q = points - center
    return q @ x_axis, q @ y_axis


def split_intervals(xvals, close_size):
    xmin = int(np.floor(xvals.min()))
    xmax = int(np.ceil(xvals.max()))
    w = xmax - xmin + 1

    if w <= 0:
        return []

    hist = np.zeros(w, dtype=np.uint8)
    xs = np.clip((xvals - xmin).astype(np.int32), 0, w - 1)
    hist[xs] = 255

    kernel = np.ones((1, close_size), np.uint8)
    closed = cv2.morphologyEx(hist.reshape(1, -1), cv2.MORPH_CLOSE, kernel).ravel()

    intervals = []
    inside = False
    start = 0

    for i, v in enumerate(closed):
        if v and not inside:
            start = i
            inside = True
        elif not v and inside:
            if i - start > 4:
                intervals.append((start + xmin, i + xmin))
            inside = False

    if inside:
        intervals.append((start + xmin, w + xmin))

    return intervals


def active_components(mask):
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    comps = []

    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 10:
            continue

        ys, xs = np.where(labels == i)
        pts = np.column_stack([xs, ys]).astype(np.float32)
        comps.append((pts, np.array(centroids[i], dtype=np.float32)))

    return comps


def classify_components(group, interval, gymin, gymax):
    active = set()
    conf = 0.0

    x1, x2 = interval
    W = max(x2 - x1, 1.0)
    H = max(gymax - gymin, 1.0)

    for cx, cy, rx, ry in group:
        nx = (cx - x1) / W
        ny = (cy - gymin) / H

        if rx >= ry * 1.20:
            if ny < 0.30:
                active.add("a")
                conf += abs(0.15 - ny)
            elif ny > 0.70:
                active.add("d")
                conf += abs(0.85 - ny)
            else:
                active.add("g")
                conf += abs(0.50 - ny)

        elif ry >= rx * 1.20:
            if nx >= 0.50 and ny < 0.50:
                active.add("b")
                conf += abs(nx - 0.85) + abs(ny - 0.25)
            elif nx >= 0.50 and ny >= 0.50:
                active.add("c")
                conf += abs(nx - 0.85) + abs(ny - 0.75)
            elif nx < 0.50 and ny >= 0.50:
                active.add("e")
                conf += abs(nx - 0.15) + abs(ny - 0.75)
            else:
                active.add("f")
                conf += abs(nx - 0.15) + abs(ny - 0.25)

        else:
            if ny < 0.25:
                active.add("a")
            elif ny > 0.75:
                active.add("d")
            else:
                active.add("g")

    return "".join(sorted(active)), -conf


def classify_pixels(active_xy, interval, gymin, gymax, threshold):
    if len(active_xy) < 8:
        return "", 0.0

    px = active_xy[:, 0]
    py = active_xy[:, 1]

    x1, x2 = interval
    W = max(x2 - x1, 1.0)
    H = max(gymax - gymin, 1.0)

    nx = (px - x1) / W
    ny = (py - gymin) / H

    regions = {
        "a": (0.16, 0.84, 0.00, 0.24),
        "b": (0.55, 1.00, 0.06, 0.54),
        "c": (0.55, 1.00, 0.46, 0.94),
        "d": (0.16, 0.84, 0.76, 1.00),
        "e": (0.00, 0.45, 0.46, 0.94),
        "f": (0.00, 0.45, 0.06, 0.54),
        "g": (0.16, 0.84, 0.36, 0.64),
    }

    active = []
    conf = 0.0

    for seg, (xa, xb, ya, yb) in regions.items():
        inside = (nx >= xa) & (nx <= xb) & (ny >= ya) & (ny <= yb)
        ratio = inside.sum() / len(nx)

        conf += abs(ratio - threshold)

        if ratio > threshold:
            active.append(seg)

    return "".join(sorted(active)), conf


def solve_component(img, close_sizes, thresholds):
    display_mask, active_mask = make_masks(img)

    display_pts = mask_points(display_mask)
    active_pts = mask_points(active_mask)

    if len(display_pts) < 10 or len(active_pts) < 8:
        return []

    center, x0, y0 = pca_axes(display_pts)

    candidates = []

    n, labels, stats, _ = cv2.connectedComponentsWithStats(display_mask, 8)
    display_boxes = []

    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 40:
            continue

        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        if w < 4 or h < 4:
            continue

        pts_y, pts_x = np.where(labels == i)
        pts = np.column_stack([pts_x, pts_y]).astype(np.float32)
        display_boxes.append(pts)

    comps = active_components(active_mask)

    for sx in [1, -1]:
        for sy in [1, -1]:
            x_axis = x0 * sx
            y_axis = y0 * sy

            dx, dy = project(display_pts, center, x_axis, y_axis)
            ax, ay = project(active_pts, center, x_axis, y_axis)
            active_projected = np.column_stack([ax, ay]).astype(np.float32)

            gymin, gymax = np.percentile(dy, [1, 99])

            comp_data = []

            for pts, cen in comps:
                xp, yp = project(pts, center, x_axis, y_axis)
                cp_x, cp_y = project(cen[None, :], center, x_axis, y_axis)

                comp_data.append(
                    (
                        float(cp_x[0]),
                        float(cp_y[0]),
                        float(xp.max() - xp.min()),
                        float(yp.max() - yp.min()),
                    )
                )

            all_interval_sets = []

            for close_size in close_sizes:
                all_interval_sets.append(split_intervals(dx, close_size))

            comp_intervals = []

            for pts in display_boxes:
                xp, yp = project(pts, center, x_axis, y_axis)
                comp_intervals.append((float(xp.min()), float(xp.max())))

            comp_intervals.sort(key=lambda z: z[0])

            if 1 <= len(comp_intervals) <= 16:
                all_interval_sets.append(comp_intervals)

            for intervals in all_interval_sets:
                observed_comp = []
                conf_comp = 0.0

                pixel_sets = [([], 0.0, t) for t in thresholds]

                for a, b in intervals:
                    group = [c for c in comp_data if a - 2 <= c[0] <= b + 2]

                    segs_c, cconf = classify_components(group, (a, b), gymin, gymax)
                    if segs_c:
                        observed_comp.append((a, segs_c))
                        conf_comp += cconf

                    idx = (active_projected[:, 0] >= a) & (active_projected[:, 0] <= b)

                    for k in range(len(pixel_sets)):
                        arr, conf, thr = pixel_sets[k]
                        segs_p, pconf = classify_pixels(active_projected[idx], (a, b), gymin, gymax, thr)
                        if segs_p:
                            arr.append((a, segs_p))
                            conf += pconf
                        pixel_sets[k] = (arr, conf, thr)

                for observed_pos, conf in [(observed_comp, conf_comp)] + [(x[0], x[1]) for x in pixel_sets]:
                    observed_pos.sort(key=lambda z: z[0])
                    obs = [s for _, s in observed_pos]

                    if 1 <= len(obs) <= 16:
                        candidates.append((candidate_score(obs, conf), obs))

    return candidates


def best_candidate(candidates):
    if not candidates:
        return None, None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0]


def solve(path):
    img0 = cv2.imread(path, cv2.IMREAD_COLOR)

    if img0 is None:
        print(0)
        print()
        print(f"{BROKEN_INDEX_BASE} a")
        return

    configs = [
        (
            BASE_MAX_SIZE,
            [5, 8, 10, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 100, 130, 160, 200],
            [0.025, 0.035, 0.050],
        ),
        (
            1400,
            [5, 8, 10, 12, 16, 20, 24, 28, 32, 36, 40, 48, 56, 64, 72, 80, 96, 100, 130, 160, 200],
            [0.022, 0.028, 0.035, 0.045, 0.055],
        ),
        (
            EXTRA_MAX_SIZE,
            [5, 8, 10, 12, 16, 20, 24, 28, 32, 36, 40, 48, 56, 64, 72, 80, 96, 100, 130, 160, 200],
            [0.020, 0.025, 0.030, 0.035, 0.040, 0.050],
        ),
    ]

    best_score = None
    best_obs = None

    for max_size, close_sizes, thresholds in configs:
        img = resize_image(img0, max_size)
        candidates = solve_component(img, close_sizes, thresholds)
        score, obs = best_candidate(candidates)

        if score is None:
            continue

        if best_score is None or score > best_score:
            best_score = score
            best_obs = obs

        if max_size == BASE_MAX_SIZE and score[0] >= 100:
            best_obs = obs
            break

        if score[0] >= 100 and score[1] == len(obs):
            best_obs = obs
            break

    if best_obs is None:
        print(0)
        print()
        print(f"{BROKEN_INDEX_BASE} a")
        return

    broken_display, broken_segment, _ = find_broken(best_obs)

    print(len(best_obs))
    print(" ".join(best_obs))
    print(broken_display, broken_segment)


def main():
    path = sys.stdin.readline().strip()
    solve(path)


if __name__ == "__main__":
    main()
