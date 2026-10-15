import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from src.braille_tables import decode_cells, is_known


def _load_rgb(image):
    if isinstance(image, str):
        image = Image.open(image)
    if isinstance(image, Image.Image):
        if image.mode in ("RGBA", "LA", "P"):
            image = image.convert("RGBA")
            bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(bg, image)
        image = np.array(image.convert("RGB"))
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[2] == 4:
        a = arr[..., 3:4].astype(np.float32) / 255.0
        arr = arr[..., :3].astype(np.float32) * a + 255.0 * (1 - a)
    return np.clip(arr[..., :3], 0, 255).astype(np.uint8)


def _resize(rgb, f):
    pil = Image.fromarray(rgb)
    method = Image.BILINEAR if f > 1 else Image.BOX
    pil = pil.resize((max(1, int(round(pil.width * f))), max(1, int(round(pil.height * f)))), method)
    return np.array(pil)


def _ink(rgb):
    rgbf = rgb.astype(np.float32)
    gray = 0.299 * rgbf[..., 0] + 0.587 * rgbf[..., 1] + 0.114 * rgbf[..., 2]
    mx = rgbf.max(axis=-1)
    mn = rgbf.min(axis=-1)
    sat = (mx - mn) / np.maximum(mx, 1.0)
    bg = float(np.median(gray))
    if bg < 110:
        gray = 255.0 - gray
        bg = 255.0 - bg
    return (gray < bg - 70) | ((sat > 0.35) & (gray < bg - 35))


def _prepare(image):
    rgb = _load_rgb(image)
    scale = 1.0
    if max(rgb.shape[:2]) > 3000:
        f = 3000.0 / max(rgb.shape[:2])
        rgb = _resize(rgb, f)
        scale *= f
    ink = _ink(rgb)
    pts, r = _find_dots(ink)
    if len(pts) and r < 3.5:
        f = min(4.0, 5.0 / r)
        if max(rgb.shape[:2]) * f <= 4000:
            rgb = _resize(rgb, f)
            scale *= f
            ink = _ink(rgb)
            pts, r = _find_dots(ink)
    return pts, r, scale


def _roundish(sl, comp):
    h = sl[0].stop - sl[0].start
    w = sl[1].stop - sl[1].start
    if h == 0 or w == 0:
        return False
    aspect = w / h
    fill = comp.sum() / float(h * w)
    return 0.55 <= aspect <= 1.8 and fill >= 0.45


def _find_dots(ink):
    lab, n = ndi.label(ink)
    if n == 0:
        return np.zeros((0, 2)), 1.0
    edt = ndi.distance_transform_edt(ink)
    slices = ndi.find_objects(lab)
    info = []
    for i, sl in enumerate(slices):
        if sl is None:
            continue
        comp = lab[sl] == (i + 1)
        area = int(comp.sum())
        if area < 2:
            continue
        info.append((i + 1, sl, comp, area, _roundish(sl, comp)))
    round_areas = [a for _, _, _, a, r in info if r and a >= 3]
    if not round_areas:
        round_areas = [a for _, _, _, a, _ in info]
    if not round_areas:
        return np.zeros((0, 2)), 1.0
    a_med = float(np.median(round_areas))
    r = max(1.0, float(np.sqrt(a_med / np.pi)))
    pts = []
    for label_id, sl, comp, area, rnd in info:
        if area < max(2.0, 0.3 * a_med):
            continue
        if rnd and area <= 1.8 * a_med:
            ys, xs = np.nonzero(comp)
            pts.append((ys.mean() + sl[0].start, xs.mean() + sl[1].start))
            continue
        if area > 12 * a_med:
            continue
        sub = np.where(comp, edt[sl], 0)
        size = max(3, int(round(1.2 * r)) * 2 + 1)
        peaks = (sub == ndi.maximum_filter(sub, size=size)) & (sub >= 0.6 * sub.max()) & comp
        plab, pn = ndi.label(peaks, structure=np.ones((3, 3)))
        if pn == 0:
            continue
        centers = ndi.center_of_mass(peaks, plab, range(1, pn + 1))
        merged = []
        for cy, cx in centers:
            if all((cy - my) ** 2 + (cx - mx) ** 2 > (1.2 * r) ** 2 for my, mx in merged):
                merged.append((cy, cx))
        expected = area / a_med
        if len(merged) == 1 and not rnd:
            continue
        if len(merged) > expected + 1.5:
            continue
        for cy, cx in merged:
            pts.append((cy + sl[0].start, cx + sl[1].start))
    return np.array(pts, dtype=np.float64).reshape(-1, 2), r


def _cluster_1d(values, thr):
    order = np.argsort(values)
    groups = [[order[0]]]
    for idx in order[1:]:
        if values[idx] - values[groups[-1][-1]] <= thr:
            groups[-1].append(idx)
        else:
            groups.append([idx])
    return groups


def _smallest_mode(gaps, min_count):
    if len(gaps) == 0:
        return None
    gaps = np.sort(np.asarray(gaps, dtype=np.float64))
    clusters = [[gaps[0]]]
    for g in gaps[1:]:
        if g <= clusters[-1][-1] * 1.18 + 1.0:
            clusters[-1].append(g)
        else:
            clusters.append([g])
    for c in clusters:
        if len(c) >= min_count:
            return float(np.median(c))
    return float(np.median(clusters[0]))


def _lattice_error(xs, x0, P, d):
    err = 0.0
    for x in xs:
        k = np.floor((x - x0) / P)
        best = min(abs(x - (x0 + kk * P + off)) for kk in (k - 1, k, k + 1) for off in (0.0, d))
        err += best
    return err


def _assign(x, x0, P, d):
    k = int(np.floor((x - x0) / P))
    best = None
    for kk in (k - 1, k, k + 1):
        for slot, off in ((0, 0.0), (1, d)):
            e = abs(x - (x0 + kk * P + off))
            if best is None or e < best[0]:
                best = (e, kk, slot)
    return best


def _unit_spacing(pts, r):
    if len(pts) < 2:
        return None
    tree = cKDTree(pts)
    k = min(9, len(pts))
    dist, idx = tree.query(pts, k=k)
    cands = []
    tol = max(1.5, 0.8 * r)
    for i in range(len(pts)):
        best = None
        for dd, j in zip(dist[i][1:], idx[i][1:]):
            if dd < 1.5 * r:
                continue
            dy = abs(pts[j, 0] - pts[i, 0])
            dx = abs(pts[j, 1] - pts[i, 1])
            if dy <= tol or dx <= tol:
                best = dd
                break
        if best is not None:
            cands.append(best)
    if not cands:
        return None
    return _smallest_mode(cands, 1 if len(cands) < 8 else max(2, int(0.05 * len(cands))))


def _line_quality(cells, resid):
    if not cells:
        return 0.0
    known = sum(1 for c in cells if is_known(c))
    return (known / len(cells)) * max(0.0, 1.0 - 2 * (float(np.mean(resid)) if resid else 1.0))


def _walk(xs, start_slot, P, d):
    cell = 0
    slot = start_slot
    anchor = xs[0]
    out = [(0, start_slot, 0.0)]
    total = 0.0
    for x in xs[1:]:
        base = anchor - slot * d
        best = None
        for ns in (0, 1):
            k = int(round((x - base - ns * d) / P))
            for kk in (k - 1, k, k + 1):
                if kk < 0 or (kk == 0 and ns <= slot):
                    continue
                e = abs(x - (base + kk * P + ns * d))
                if best is None or e < best[0]:
                    best = (e, kk, ns)
        if best is None:
            best = (abs(x - (base + P)), 1, 0)
        e, kk, ns = best
        cell += kk
        slot = ns
        anchor = x
        total += e
        out.append((cell, slot, e))
    return out, total


def _dominant_angle(pts, r):
    if len(pts) < 6:
        return 0.0
    tree = cKDTree(pts)
    k = min(7, len(pts))
    dist, idx = tree.query(pts, k=k)
    angles = []
    weights = []
    for i in range(len(pts)):
        for dd, j in zip(dist[i][1:], idx[i][1:]):
            if dd < 1.5 * r or dd > 12 * r:
                continue
            dy = pts[j, 0] - pts[i, 0]
            dx = pts[j, 1] - pts[i, 1]
            if dx < 0:
                dx, dy = -dx, -dy
            a = np.degrees(np.arctan2(dy, dx))
            if -35 <= a <= 35:
                angles.append(a)
                weights.append(1.0 / dd)
    if len(angles) < 5:
        return 0.0
    angles = np.array(angles)
    weights = np.array(weights)
    hist, edges = np.histogram(angles, bins=140, range=(-35, 35), weights=weights)
    hist = np.convolve(hist, np.ones(5) / 5, mode="same")
    peak = (edges[np.argmax(hist)] + edges[np.argmax(hist) + 1]) / 2
    near = np.abs(angles - peak) < 2.0
    return float(np.average(angles[near], weights=weights[near])) if near.any() else float(peak)


def _rotate(pts, center, deg):
    t = np.radians(deg)
    cy, cx = center
    y = pts[:, 0] - cy
    x = pts[:, 1] - cx
    return np.c_[cy + x * np.sin(t) + y * np.cos(t), cx + x * np.cos(t) - y * np.sin(t)]


def decode(image):
    pts, r, scale = _prepare(image)
    angle = _dominant_angle(pts, r) if len(pts) else 0.0
    center = pts.mean(axis=0) if len(pts) else np.zeros(2)
    if abs(angle) > 0.4:
        pts = _rotate(pts, center, -angle)
    empty = {"text": "", "detections": [], "quality": 0.0, "n_dots": int(len(pts)), "n_cells": 0}
    if len(pts) == 0:
        return empty
    thr = max(1.5, 0.8 * r)
    row_groups = _cluster_1d(pts[:, 0], thr)
    row_centers = np.array([pts[g, 0].mean() for g in row_groups])
    row_gaps = np.diff(row_centers)
    row_gaps = row_gaps[row_gaps >= 1.5 * r]
    u = _unit_spacing(pts, r)
    s = _smallest_mode(row_gaps, 1 if len(row_gaps) < 6 else 2)
    if u is not None:
        if s is None:
            s = u
        elif 1.7 <= s / u <= 2.4:
            s = s / 2.0
    if s is None or s > 6 * r:
        s = u if u is not None else 2.6 * r
    lines = []
    i = 0
    while i < len(row_centers):
        top = row_centers[i]
        members = [(i, 0)]
        j = i + 1
        while j < len(row_centers):
            k = (row_centers[j] - top) / s
            slot = int(round(k))
            if slot <= 2 and abs(k - slot) < 0.35:
                members.append((j, slot))
                j += 1
            else:
                break
        lines.append((top, members))
        i = j
    line_cols = []
    all_gaps = []
    for top, members in lines:
        idxs = []
        slots = []
        for gi, slot in members:
            for p in row_groups[gi]:
                idxs.append(p)
                slots.append(slot)
        idxs = np.array(idxs)
        slots = np.array(slots)
        col_groups = _cluster_1d(pts[idxs, 1], thr)
        cols = sorted([(pts[idxs[g], 1].mean(), [(idxs[q], slots[q]) for q in g]) for g in col_groups], key=lambda c: c[0])
        line_cols.append((top, cols))
        cx = np.array([c[0] for c in cols])
        if len(cx) > 1:
            all_gaps.extend(np.diff(cx).tolist())
    all_gaps = np.array(all_gaps)
    valid = all_gaps[all_gaps >= 1.5 * r] if len(all_gaps) else all_gaps
    d = _smallest_mode(valid, 1 if len(valid) < 6 else 2)
    if d is None or not (0.55 * s <= d <= 1.7 * s):
        d = u if (u is not None and 0.55 * s <= u <= 1.7 * s) else s
    larger = valid[valid > 1.2 * d] if len(valid) else valid
    g1 = _smallest_mode(larger, 1) if len(larger) else None
    pair_diffs = []
    for _, cols in line_cols:
        xs_all = [cc[0] for cc in cols]
        starts = []
        i = 0
        while i < len(xs_all) - 1:
            if abs((xs_all[i + 1] - xs_all[i]) - d) <= 0.22 * d:
                starts.append(xs_all[i])
                i += 2
            else:
                i += 1
        for a, b in zip(starts, starts[1:]):
            if 1.8 * d <= b - a <= 4.0 * d:
                pair_diffs.append(b - a)
    P = None
    if pair_diffs:
        P = _smallest_mode(pair_diffs, 1 if len(pair_diffs) < 6 else 2)
    if P is None:
        if g1 is None:
            P = 2.6 * d
        else:
            ratio = g1 / d
            if 1.05 <= ratio < 2.3:
                P = g1 + d
            elif 2.3 <= ratio <= 3.6:
                P = g1
            else:
                P = 2.6 * d
    detections = []
    text_lines = []
    known = 0
    total = 0
    resid = []
    for top, cols in line_cols:
        if not cols:
            continue
        line_dets = []
        runs = [[cols[0]]]
        for c in cols[1:]:
            if c[0] - runs[-1][-1][0] > 1.5 * P:
                runs.append([c])
            else:
                runs[-1].append(c)
        line_cells = []
        line_known = []
        line_resid = []
        prev_end = None
        for run in runs:
            xs = [c[0] for c in run]
            walk0, err0 = _walk(xs, 0, P, d)
            walk1, err1 = _walk(xs, 1, P, d)
            if err1 + 0.15 * d < err0:
                walk, start_slot = walk1, 1
            else:
                walk, start_slot = walk0, 0
            x0 = xs[0] - start_slot * d
            cells = {}
            for (cxv, dots), (k, slot, e) in zip(run, walk):
                line_resid.append(e / max(d, 1.0))
                cell = cells.setdefault(k, set())
                for _, rslot in dots:
                    cell.add(int(rslot) + 1 + 3 * slot)
            col1_pos = {}
            for (cxv, dots), (k, slot, e) in zip(run, walk):
                col1_pos.setdefault(k, cxv - slot * d)
            ks = sorted(cells)
            start_x = col1_pos[ks[0]]
            if prev_end is not None:
                gap_cells = int(round((start_x - prev_end) / P)) - 1
                line_cells.extend([None] * max(1, gap_cells))
            for k in range(ks[0], ks[-1] + 1):
                dots = cells.get(k)
                if dots is None:
                    line_cells.append(None)
                    continue
                line_known.append(tuple(sorted(dots)))
                line_cells.append(tuple(sorted(dots)))
                ch = decode_cells([tuple(sorted(dots))]) or "^"
                line_dets.append({
                    "x": float((col1_pos.get(k, x0 + k * P) + d / 2) / scale),
                    "y": float((top + s) / scale),
                    "width": float((d + 2.6 * r) / scale),
                    "height": float((2 * s + 2.6 * r) / scale),
                    "confidence": 1.0,
                    "class": ch,
                    "dots": tuple(sorted(dots)),
                    "source": "grade",
                })
            prev_end = col1_pos[ks[-1]]
        lq = _line_quality(line_known, line_resid)
        if lq < 0.45 and len(line_cols) > 1:
            continue
        sparse = len(runs) >= 0.6 * max(1, len(line_known)) and all(len(cc) <= 2 for cc in line_known)
        if len(line_cols) >= 3 and len(line_known) <= 3 and sparse:
            continue
        total += len(line_known)
        known += sum(1 for cc in line_known if is_known(cc))
        resid.extend(line_resid)
        detections.extend(line_dets)
        text_lines.append(decode_cells(line_cells).strip())
    mean_resid = float(np.mean(resid)) if resid else 1.0
    quality = (known / total if total else 0.0) * max(0.0, 1.0 - mean_resid * 2)
    for det in detections:
        det["confidence"] = round(quality, 3)
        if abs(angle) > 0.4:
            yy, xx = _rotate(np.array([[det["y"] * scale, det["x"] * scale]]), center, angle)[0]
            det["x"], det["y"] = float(xx / scale), float(yy / scale)
    return {
        "text": "\n".join(t for t in text_lines if t),
        "detections": detections,
        "quality": round(quality, 3),
        "n_dots": int(len(pts)),
        "n_cells": total,
        "params": {"r": round(r, 2), "s": round(s, 2), "d": round(d, 2), "P": round(P, 2), "angulo": round(angle, 2)},
    }
