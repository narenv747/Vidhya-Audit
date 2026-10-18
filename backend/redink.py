"""Computer-vision helper: find red examiner ink on a scanned OR phone-photographed page.
Tuned on real data: flat scans with faint pink ticks and printed red margin lines, and phone photos with warm lighting and orange-red ink."""
import cv2
import numpy as np

NORM_WIDTH = 1000
RED_SAT_MIN = 38      # saturation (after lighting correction) above which a red/orange pixel counts as ink
HEADER_FRAC = 0.14    # top part of printed-margin scans (header, page number, QR) is ignored


def load_page(path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    return cv2.resize(img, (NORM_WIDTH, int(h * NORM_WIDTH / w)))


def flatten(img):
    """Remove uneven lighting: divide each channel by a blurred 'paper colour' estimate."""
    out = np.empty_like(img)
    for i in range(3):
        ch = img[..., i]
        bg = cv2.GaussianBlur(cv2.dilate(ch, np.ones((25, 25), np.uint8)), (0, 0), 30).astype(np.float32)
        out[..., i] = np.clip(ch / np.maximum(bg, 1) * 255, 0, 255).astype(np.uint8)
    return out


def paper_mask(img):
    """The big bright region (the paper), shrunk a little so table/edge shadows are excluded."""
    g = cv2.GaussianBlur(cv2.dilate(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), np.ones((11, 11), np.uint8)), (0, 0), 6)  # dilate: thin ink must not punch holes in the mask
    t, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = (g > t * 0.8).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n > 1:
        m = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    return cv2.erode(m, np.ones((41, 41), np.uint8)) > 0


def red_mask(img_bgr):
    f = flatten(img_bgr)
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = (((h < 20) | (h > 160)) & (s > RED_SAT_MIN) & (v < 238) & paper_mask(img_bgr)).astype(np.uint8) * 255
    H = m.shape[0]
    col = np.count_nonzero(m, axis=0)
    margins = np.where(col > 0.25 * H)[0]
    if len(margins):                                   # printed red margin lines => printed answer-book scan
        for x in margins:
            m[:, max(0, x - 3): x + 4] = 0
        m[: int(H * HEADER_FRAC)] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)  # drop specks and thin horizontal ruled-line streaks
    keep = np.zeros_like(m)
    for i in range(1, n):
        x, y, w, hh, a = st[i]
        if a >= 20 and not (w >= 50 and hh <= 7):
            keep[lab == i] = 255
    return keep


def red_pixels(img, bbox, pad_y=0.02, full_width=True, mask=None):
    """Red pixels near a region. Examiners often mark in the margin, so by default we look across the full page width."""
    h, w = img.shape[:2]
    if mask is None:
        mask = red_mask(img)
    x, y, bw, bh = bbox
    x0, x1 = (0, w) if full_width else (max(0, int(x * w)), min(w, int((x + bw) * w)))
    y0, y1 = max(0, int((y - pad_y) * h)), min(h, int((y + bh + pad_y) * h))
    if y1 <= y0 or x1 <= x0:
        return 0
    return int(np.count_nonzero(mask[y0:y1, x0:x1]))
