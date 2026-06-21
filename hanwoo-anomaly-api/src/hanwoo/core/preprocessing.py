from __future__ import annotations

import os
import warnings
from pathlib import Path

from hanwoo.core.config import U2NET_HOME

os.environ.setdefault("U2NET_HOME", str(U2NET_HOME))
os.environ.setdefault("MODEL_CHECKSUM_DISABLED", "1")

warnings.filterwarnings("ignore", message=".*CUDAExecutionProvider.*")

import cv2
import numpy as np
from PIL import Image
from rembg import remove, new_session

_sessions: dict = {}


def get_session(model_name: str = "u2net"):
    if model_name not in _sessions:
        from rembg.sessions.base import BaseSession
        orig = BaseSession.checksum_disabled
        BaseSession.checksum_disabled = classmethod(lambda cls, *a, **kw: True)
        try:
            _sessions[model_name] = new_session(
                model_name=model_name,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
        finally:
            BaseSession.checksum_disabled = orig
    return _sessions[model_name]


def refine_background_mask(mask: Image.Image) -> Image.Image:
    mask_np = np.array(mask.convert("L"))
    _, binary = cv2.threshold(mask_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if not np.any(binary):
        return mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        largest = np.where(labels == largest_label, 255, 0).astype(np.uint8)
    else:
        largest = binary
    h, w = largest.shape
    contours, _ = cv2.findContours(largest, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    main_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(main_contour)
    center, size, angle = rect
    size = (size[0] * 1.04, size[1] * 1.04)
    box_pts = np.int32(cv2.boxPoints((center, size, angle)))
    rect_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(rect_mask, [box_pts], -1, 255, -1)
    kernel = np.ones((15, 15), np.uint8)
    rect_rounded = cv2.morphologyEx(rect_mask, cv2.MORPH_CLOSE, kernel)
    rect_rounded = cv2.morphologyEx(rect_rounded, cv2.MORPH_OPEN, kernel)
    refined = cv2.bitwise_and(largest, rect_rounded)
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    blurred = cv2.GaussianBlur(refined.astype(np.float32), (9, 9), 0)
    _, smoothed = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)
    return Image.fromarray(smoothed.astype(np.uint8), mode="L")


def _odd_kernel_size(value: int, minimum: int = 3, maximum: int = 151) -> int:
    value = max(minimum, min(maximum, int(value)))
    return value if value % 2 == 1 else value + 1


def _dark_tray_candidate(image_np: np.ndarray) -> np.ndarray:
    max_ch = image_np.max(axis=2)
    min_ch = image_np.min(axis=2)
    spread = max_ch - min_ch
    lum = 0.299*image_np[:,:,0] + 0.587*image_np[:,:,1] + 0.114*image_np[:,:,2]
    return ((max_ch < 125) & (min_ch < 110) & (spread < 90) & (lum < 120)) | (max_ch < 70)


def _dark_tray_retention(image_np, mask_np):
    dark = _dark_tray_candidate(image_np)
    dark_count = int(dark.sum())
    if dark_count < max(500, int(dark.size * 0.002)):
        return None
    return float((dark & (mask_np > 127)).sum() / dark_count)


def preserve_dark_tray_in_mask(image: Image.Image, mask: Image.Image, min_dark_retention: float = 0.4) -> Image.Image:
    image_np = np.array(image.convert("RGB"))
    mask_np = np.array(mask.convert("L"))
    foreground = mask_np > 127
    if not foreground.any():
        return mask
    retention = _dark_tray_retention(image_np, mask_np)
    if retention is None or retention >= min_dark_retention:
        return mask
    h, w = mask_np.shape
    ys, xs = np.where(foreground)
    left, right = int(xs.min()), int(xs.max())
    top, bottom = int(ys.min()), int(ys.max())
    fg_w, fg_h = right-left+1, bottom-top+1
    pad_x = max(80, int(fg_w*0.55), int(w*0.06))
    pad_y = max(80, int(fg_h*0.55), int(h*0.06))
    roi = np.zeros((h, w), dtype=bool)
    roi[max(0,top-pad_y):min(h,bottom+pad_y+1), max(0,left-pad_x):min(w,right+pad_x+1)] = True
    tray_pixels = (_dark_tray_candidate(image_np) & roi).astype(np.uint8) * 255
    if not np.any(tray_pixels):
        return mask
    cs  = _odd_kernel_size(max(31, int(max(fg_w,fg_h)*0.04)), maximum=101)
    os_ = _odd_kernel_size(max(5,  int(max(fg_w,fg_h)*0.01)), maximum=21)
    tray_pixels = cv2.morphologyEx(tray_pixels, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(cs,cs)))
    tray_pixels = cv2.morphologyEx(tray_pixels, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(os_,os_)))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(tray_pixels, connectivity=8)
    if num_labels <= 1:
        return mask
    fg_area = int(foreground.sum())
    candidates = [l for l in range(1, num_labels) if stats[l, cv2.CC_STAT_AREA] >= max(1000, int(fg_area*0.08))]
    if not candidates:
        return mask
    largest_label = max(candidates, key=lambda l: stats[l, cv2.CC_STAT_AREA])
    largest_comp = np.where(labels == largest_label, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(largest_comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    tray_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(tray_mask, [cv2.convexHull(max(contours, key=cv2.contourArea))], -1, 255, -1)
    tray_area = int((tray_mask > 0).sum())
    if float((foreground & (tray_mask > 0)).sum() / fg_area) < 0.5:
        return mask
    if tray_area < fg_area*0.5 or tray_area > h*w*0.6:
        return mask
    repaired = cv2.morphologyEx(cv2.bitwise_or(mask_np, tray_mask), cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11)))
    return Image.fromarray(repaired.astype(np.uint8), mode="L")


def apply_mask_to_rgb(image, mask, rembg_output=None, rembg_mask=None, background_color=(255,255,255)):
    original_np = np.array(image.convert("RGB"))
    mask_np = np.array(mask.convert("L")).astype(np.float32)
    alpha = np.clip(mask_np/255.0, 0.0, 1.0)[:,:,None]
    source_np = np.array(rembg_output.convert("RGBA"))[:,:,:3] if rembg_output is not None else original_np.copy()
    rembg_mask_np = (np.array(rembg_mask.convert("L")) if rembg_mask is not None
                     else np.array(rembg_output.split()[3]) if (rembg_output is not None and rembg_output.mode=="RGBA")
                     else mask_np)
    final_fg = mask_np > 10
    added_fg = final_fg & ~(rembg_mask_np > 10)
    added_dark = added_fg & _dark_tray_candidate(original_np)
    source_np = source_np.copy()
    source_np[added_dark] = original_np[added_dark]
    bg = np.full_like(source_np, background_color, dtype=np.uint8)
    source_np[added_fg & ~added_dark] = bg[added_fg & ~added_dark]
    rgb = (source_np.astype(np.float32)*alpha + bg.astype(np.float32)*(1-alpha)).round().astype(np.uint8)
    return Image.merge("RGBA", (*Image.fromarray(rgb, mode="RGB").split(), mask))


def remove_background(image: Image.Image, use_session: bool = True, return_mask: bool = False,
                      refine_mask: bool = True, model_name: str = "u2net", preserve_dark_tray: bool = True):
    if image.mode != "RGB":
        image = image.convert("RGB")
    session = get_session(model_name) if use_session else None
    output = remove(image, session=session) if session else remove(image, model_name=model_name)
    if refine_mask:
        mask = output.split()[3] if output.mode == "RGBA" else output.convert("RGBA").split()[3]
        refined_mask = refine_background_mask(mask)
        repaired_tray = False
        if preserve_dark_tray:
            before = np.array(refined_mask.convert("L"), dtype=np.int16)
            refined_mask = preserve_dark_tray_in_mask(image, refined_mask)
            after = np.array(refined_mask.convert("L"), dtype=np.int16)
            repaired_tray = bool(np.any(after > before + 10))
        if repaired_tray:
            output = apply_mask_to_rgb(image, refined_mask, rembg_output=output, rembg_mask=mask)
        else:
            if output.mode != "RGBA":
                output = output.convert("RGBA")
            output = Image.merge("RGBA", (*output.split()[:3], refined_mask))
    if return_mask:
        mask = output.split()[3] if output.mode == "RGBA" else output.convert("RGBA").split()[3]
        return output, mask
    return output


def _detect_top_horizontal_line(mask: Image.Image):
    mask_np = np.array(mask.convert("L"))
    blur = cv2.GaussianBlur(mask_np, (5,5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if not np.any(binary):
        return 0.0, None
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  np.ones((3,3), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
    edges = cv2.Canny(binary, 50, 150)
    h, w = binary.shape[:2]
    min_len = max(40, int(w*0.25))
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=min_len, maxLineGap=20)
    candidates = []
    for line in (lines if lines is not None else []):
        x1,y1,x2,y2 = line[0]
        dx,dy = x2-x1, y2-y1
        if dx == 0: continue
        angle = ((np.degrees(np.arctan2(dy,dx)) + 90) % 180) - 90
        length = float(np.hypot(dx,dy))
        if abs(angle) <= 30 and length >= min_len:
            candidates.append(((y1+y2)/2.0, -length, float(angle), (int(x1),int(y1),int(x2),int(y2))))
    if candidates:
        candidates.sort(key=lambda x: (x[0],x[1]))
        top_y = candidates[0][0]
        top_band = sorted([c for c in candidates if c[0] <= top_y+max(8,h*0.02)], key=lambda x: x[1])
        return top_band[0][2], top_band[0][3]
    fg = binary > 0
    if not np.any(fg):
        return 0.0, None
    top_points_x, top_points_y = [], []
    for x in range(w):
        ys = np.where(fg[:,x])[0]
        if ys.size > 0:
            top_points_x.append(x); top_points_y.append(int(ys[0]))
    if len(top_points_x) < max(10, int(w*0.15)):
        return 0.0, None
    x_arr = np.array(top_points_x, dtype=np.float32)
    y_arr = np.array(top_points_y, dtype=np.float32)
    q_lo, q_hi = np.percentile(x_arr, [15,85])
    keep = (x_arr>=q_lo)&(x_arr<=q_hi)
    x_fit = x_arr[keep] if keep.sum()>=8 else x_arr
    y_fit = y_arr[keep] if keep.sum()>=8 else y_arr
    m, b = np.polyfit(x_fit, y_fit, 1)
    angle = float(np.degrees(np.arctan(m)))
    x1,x2 = int(np.min(x_fit)), int(np.max(x_fit))
    y1 = int(np.clip(m*x1+b, 0, h-1)); y2 = int(np.clip(m*x2+b, 0, h-1))
    return angle, (x1,y1,x2,y2)


def detect_top_line_tilt_angle(mask: Image.Image) -> float:
    angle, _ = _detect_top_horizontal_line(mask)
    return float(angle)


def rotate_image_and_mask(image: Image.Image, mask: Image.Image, angle: float):
    if abs(angle) < 1e-3:
        return image, mask
    image_np = np.array(image)
    mask_np = np.array(mask.convert("L"))
    h, w = mask_np.shape[:2]
    matrix = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    if image.mode == "RGBA":
        rgb = image_np[:,:,:3]; alpha_ch = image_np[:,:,3]
        r_rgb   = cv2.warpAffine(rgb,      matrix, (w,h), flags=cv2.INTER_LINEAR,  borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
        r_alpha = cv2.warpAffine(alpha_ch, matrix, (w,h), flags=cv2.INTER_LINEAR,  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        rotated_image = Image.fromarray(np.dstack([r_rgb, r_alpha]).astype(np.uint8), mode="RGBA")
    else:
        r_rgb = cv2.warpAffine(image_np, matrix, (w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
        rotated_image = Image.fromarray(r_rgb.astype(np.uint8), mode="RGB")
    r_mask = cv2.warpAffine(mask_np, matrix, (w,h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return rotated_image, Image.fromarray(r_mask.astype(np.uint8), mode="L")


def crop_image_by_mask(image: Image.Image, mask: Image.Image, padding: int = 0):
    mask_np = np.array(mask.convert("L"))
    fg = mask_np > 10
    if not fg.any():
        return image, mask
    rows, cols = np.any(fg, axis=1), np.any(fg, axis=0)
    ri, ci = np.where(rows)[0], np.where(cols)[0]
    if len(ri)==0 or len(ci)==0:
        return image, mask
    top    = max(0, int(ri[0])  - padding)
    bottom = min(image.height, int(ri[-1]) + padding + 1)
    left   = max(0, int(ci[0])  - padding)
    right  = min(image.width,  int(ci[-1]) + padding + 1)
    return image.crop((left,top,right,bottom)), mask.crop((left,top,right,bottom))


def preprocess(image: Image.Image) -> Image.Image:
    """배경제거 → 기울기보정 → 크롭 (260331_beef.py와 동일)."""
    proc_img, bg_mask = remove_background(image, return_mask=True)
    detected_angle = detect_top_line_tilt_angle(bg_mask)
    img_pos, mask_pos = rotate_image_and_mask(proc_img, bg_mask,  detected_angle)
    img_neg, mask_neg = rotate_image_and_mask(proc_img, bg_mask, -detected_angle)
    if abs(detect_top_line_tilt_angle(mask_pos)) <= abs(detect_top_line_tilt_angle(mask_neg)):
        proc_img, bg_mask = img_pos, mask_pos
    else:
        proc_img, bg_mask = img_neg, mask_neg
    proc_img, bg_mask = crop_image_by_mask(proc_img, bg_mask, padding=0)
    proc_img, _ = crop_image_by_mask(proc_img, bg_mask, padding=0)
    return proc_img.convert("RGB")
