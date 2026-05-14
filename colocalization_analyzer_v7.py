# -*- coding: utf-8 -*-
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

"""
=============================================================================
  QuantaColoc v7
  - aicspylibczi read_mosaic 기반 동적 해상도 로딩
  - 확대 수준에 맞는 해상도만 읽어서 렉 최소화
  - 다각형 ROI, Threshold Preview, Watershed, 면적 분석
=============================================================================
"""

import os, re, struct, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import ndimage
from skimage import filters, measure, morphology
from skimage.transform import resize as sk_resize
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.widgets import Slider, Button
from datetime import datetime

warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
#  CZI Mosaic Reader (aicspylibczi 기반)
# ─────────────────────────────────────────────
class CziMosaicReader:
    """
    CZI 파일을 피라미드 구조로 읽는 클래스.
    확대 영역에 맞는 해상도만 읽어서 ZEN처럼 빠르게 동작.
    """
    def __init__(self, path):
        from aicspylibczi import CziFile
        import xml.etree.ElementTree as ET

        self.path   = path
        self.czi    = CziFile(path)
        self.name   = Path(path).stem

        # 전체 이미지 크기
        bbox        = self.czi.get_mosaic_bounding_box()
        self.W      = bbox.w
        self.H      = bbox.h
        self.x0     = bbox.x
        self.y0     = bbox.y

        # 채널 정보
        meta_str    = ET.tostring(self.czi.meta, encoding='unicode') if self.czi.meta is not None else ''
        fluors      = list(dict.fromkeys(re.findall(r'<Fluor>(.*?)</Fluor>', meta_str)))
        dims        = self.czi.get_dims_shape()
        n_channels  = dims[0].get('C', (0, 1))[1]
        if not fluors or len(fluors) < n_channels:
            fluors  = [f'Ch{i}' for i in range(n_channels)]
        self.channels  = fluors
        self.n_channels = n_channels

        print(f"  파일: {self.name}")
        print(f"  크기: {self.W} x {self.H} px")
        print(f"  채널: {self.channels}")

    def read_region(self, x0, y0, x1, y1, scale_factor=1.0):
        """
        원본 좌표 (x0,y0)~(x1,y1) 영역을 scale_factor 해상도로 읽기.
        Returns: np.ndarray (C, H, W) uint16
        """
        x0 = max(0, int(x0)); y0 = max(0, int(y0))
        x1 = min(self.W, int(x1)); y1 = min(self.H, int(y1))
        if x1 <= x0 or y1 <= y0:
            return np.zeros((self.n_channels, 1, 1), dtype=np.uint16)

        scale_factor = max(0.01, min(1.0, scale_factor))

        imgs = []
        for c in range(self.n_channels):
            try:
                region = (self.x0 + x0, self.y0 + y0, x1 - x0, y1 - y0)
                arr, _ = self.czi.read_mosaic(region=region,
                                              scale_factor=scale_factor,
                                              C=c)
                arr = arr.squeeze().astype(np.uint16)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                imgs.append(arr)
            except Exception as e:
                h = max(1, int((y1-y0)*scale_factor))
                w = max(1, int((x1-x0)*scale_factor))
                imgs.append(np.zeros((h, w), dtype=np.uint16))

        # 같은 크기로 맞추기
        min_h = min(a.shape[0] for a in imgs)
        min_w = min(a.shape[1] for a in imgs)
        imgs  = [a[:min_h, :min_w] for a in imgs]
        return np.stack(imgs, axis=0)

    def get_thumbnail(self, max_px=2048):
        """전체 이미지 썸네일 반환 (C, H, W)"""
        scale = min(max_px / max(self.W, 1), max_px / max(self.H, 1), 1.0)
        print(f"  썸네일 생성 중... (scale={scale:.3f})")
        return self.read_region(0, 0, self.W, self.H, scale_factor=scale), scale

    def read_full(self):
        """전체 원본 이미지 반환 (분석용)"""
        return self.read_region(0, 0, self.W, self.H, scale_factor=1.0)


def load_image(path):
    """CZI or TIFF 로드. CZI는 CziMosaicReader, TIFF는 tifffile."""
    ext = Path(path).suffix.lower()
    if ext == '.czi':
        return CziMosaicReader(path)
    elif ext in ('.tif', '.tiff'):
        return _load_tiff_as_reader(path)
    raise ValueError(f"지원하지 않는 형식: {ext}")


class TiffReader:
    """TIFF 파일을 CziMosaicReader와 동일한 인터페이스로 래핑"""
    def __init__(self, path):
        import tifffile
        self.name = Path(path).stem
        with tifffile.TiffFile(path) as tf:
            arr = tf.asarray()
        if arr.ndim == 2:   arr = arr[np.newaxis]
        elif arr.ndim == 3:
            if arr.shape[2] <= 8: arr = arr.transpose(2, 0, 1)
        elif arr.ndim == 4: arr = arr[0] if arr.shape[1] <= 8 else arr[:, 0]
        self._data      = arr.astype(np.uint16)
        self.n_channels = arr.shape[0]
        self.H          = arr.shape[1]
        self.W          = arr.shape[2]
        self.channels   = [f'Ch{i}' for i in range(self.n_channels)]
        print(f"  파일: {self.name}")
        print(f"  크기: {self.W} x {self.H} px,  채널: {self.channels}")

    def read_region(self, x0, y0, x1, y1, scale_factor=1.0):
        x0=max(0,int(x0)); y0=max(0,int(y0))
        x1=min(self.W,int(x1)); y1=min(self.H,int(y1))
        crop = self._data[:, y0:y1, x0:x1]
        if scale_factor < 1.0:
            th = max(1, int(crop.shape[1]*scale_factor))
            tw = max(1, int(crop.shape[2]*scale_factor))
            crop = np.stack([sk_resize(crop[c].astype(float),(th,tw),
                             anti_aliasing=True,preserve_range=True).astype(np.uint16)
                             for c in range(crop.shape[0])], axis=0)
        return crop

    def get_thumbnail(self, max_px=2048):
        scale = min(max_px/max(self.W,1), max_px/max(self.H,1), 1.0)
        return self.read_region(0,0,self.W,self.H,scale_factor=scale), scale

    def read_full(self):
        return self._data.copy()


def _load_tiff_as_reader(path):
    return TiffReader(path)


# ─────────────────────────────────────────────
#  Utility
# ─────────────────────────────────────────────
def norm8(arr):
    v = arr.astype(float)
    if v.max() == 0: return np.zeros_like(arr, dtype=np.uint8)
    p2, p98 = np.percentile(v[v > 0], [2, 98])
    return np.clip((v - p2) / (p98 - p2 + 1e-9) * 255, 0, 255).astype(np.uint8)

def build_composite(arr_a, arr_b, arr_d, show_a, show_b, show_d):
    h, w = arr_a.shape
    rgb  = np.zeros((h, w, 3), dtype=np.uint8)
    if show_b: rgb[..., 0] = arr_b
    if show_a: rgb[..., 1] = arr_a
    if show_d and arr_d is not None:
        rgb[..., 2] = np.clip(rgb[..., 2].astype(int) + arr_d, 0, 255).astype(np.uint8)
    return rgb


# ─────────────────────────────────────────────
#  ROI Selector (동적 해상도)
# ─────────────────────────────────────────────
def select_roi_polygon(reader, ch_a_idx, ch_b_idx):
    """
    동적 해상도 ROI 선택창.
    확대 수준에 따라 read_mosaic으로 해당 영역만 고해상도 로드.
    Returns: list of np.array([[x,y],...]) in original coords
    """
    W, H = reader.W, reader.H
    ch_a_name = reader.channels[ch_a_idx]
    ch_b_name = reader.channels[ch_b_idx]
    has_dapi  = reader.n_channels > 0

    # 초기 썸네일
    thumb_data, init_scale = reader.get_thumbnail(max_px=2048)
    tA = norm8(thumb_data[ch_a_idx])
    tB = norm8(thumb_data[ch_b_idx])
    tD = norm8(thumb_data[0]) if has_dapi else None

    ch_state = [True, True, True]   # [ch_a, ch_b, dapi]
    current_scale = [init_scale]
    current_region = [0, 0, W, H]   # x0,y0,x1,y1 in original coords

    fig, ax = plt.subplots(figsize=(13, 9))
    fig.patch.set_facecolor('#1a1a1a')
    ax.set_facecolor('#1a1a1a')
    composite = build_composite(tA, tB, tD, True, True, True)
    im_handle = ax.imshow(composite, extent=[0, W, H, 0], aspect='equal')
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.set_title(
        f"{reader.name}    Green={ch_a_name}   Red={ch_b_name}   Blue=DAPI\n"
        "Left-click: add vertex  |  Double-click / Enter: finish ROI  |  "
        "Scroll: zoom  |  Right-drag: pan  |  Backspace: undo",
        color='white', fontsize=8, pad=6
    )

    # ── 동적 해상도 로딩 ───────────────────────────────────
    reload_timer = [None]

    def reload_region():
        xlim = ax.get_xlim(); ylim = sorted(ax.get_ylim())
        x0,x1 = max(0,xlim[0]), min(W,xlim[1])
        y0,y1 = max(0,ylim[0]), min(H,ylim[1])
        view_w = x1 - x0; view_h = y1 - y0
        if view_w <= 0 or view_h <= 0: return

        # 화면 픽셀 대비 이미지 픽셀 비율로 scale 결정
        fig_w_px = fig.get_size_inches()[0] * fig.dpi
        scale = min(fig_w_px / max(view_w, 1), 1.0)
        scale = max(0.02, scale)

        current_region[:] = [x0, y0, x1, y1]
        current_scale[0]  = scale

        data = reader.read_region(x0, y0, x1, y1, scale_factor=scale)
        nA = norm8(data[ch_a_idx])
        nB = norm8(data[ch_b_idx])
        nD = norm8(data[0]) if has_dapi else None

        comp = build_composite(nA, nB, nD,
                               ch_state[0], ch_state[1], ch_state[2])
        im_handle.set_data(comp)
        im_handle.set_extent([x0, x1, y1, y0])
        fig.canvas.draw_idle()

    def schedule_reload(_=None):
        """스크롤/패닝 후 짧은 딜레이 후 리로드 (연속 이벤트 디바운싱)"""
        reload_region()

    # ── zoom / pan ──────────────────────────────────────
    panning   = [False]
    pan_start = [None]

    def on_scroll(event):
        if event.inaxes != ax: return
        xd, yd = event.xdata, event.ydata
        xlim = ax.get_xlim(); ylim = ax.get_ylim()
        f = 0.8 if event.button == 'up' else 1.25
        ax.set_xlim([xd+(x-xd)*f for x in xlim])
        ax.set_ylim([yd+(y-yd)*f for y in ylim])
        schedule_reload()

    def on_press(event):
        if event.button == 3:
            panning[0] = True
            pan_start[0] = (event.x, event.y, ax.get_xlim(), ax.get_ylim())

    def on_motion(event):
        if panning[0] and pan_start[0] and event.inaxes == ax:
            inv = ax.transData.inverted()
            p0  = inv.transform((pan_start[0][0], pan_start[0][1]))
            p1  = inv.transform((event.x, event.y))
            ddx = p0[0]-p1[0]; ddy = p0[1]-p1[1]
            xl0, yl0 = pan_start[0][2], pan_start[0][3]
            ax.set_xlim([xl0[0]+ddx, xl0[1]+ddx])
            ax.set_ylim([yl0[0]+ddy, yl0[1]+ddy])
            fig.canvas.draw_idle()

    def on_release(event):
        if event.button == 3:
            panning[0] = False
            schedule_reload()

    # ── ROI drawing ────────────────────────────────────
    current_pts   = []
    finished_rois = []
    temp_artists  = []
    roi_artists   = []
    confirmed     = [False]

    def redraw_current():
        for a in temp_artists: a.remove()
        temp_artists.clear()
        if not current_pts: return
        xs = [p[0] for p in current_pts]
        ys = [p[1] for p in current_pts]
        sc = ax.scatter(xs, ys, s=20, c='yellow', zorder=5)
        temp_artists.append(sc)
        if len(current_pts) > 1:
            ln, = ax.plot(xs, ys, '-', color='yellow', lw=1.5, zorder=4)
            temp_artists.append(ln)
        if len(current_pts) > 2:
            cl, = ax.plot([xs[-1],xs[0]], [ys[-1],ys[0]],
                          '--', color='yellow', lw=1, alpha=0.5, zorder=4)
            temp_artists.append(cl)
        fig.canvas.draw_idle()

    def finish_roi():
        if len(current_pts) < 3:
            print("  ROI는 꼭짓점이 3개 이상이어야 합니다.")
            return
        pts = np.array(current_pts)
        finished_rois.append(pts)
        idx = len(finished_rois)
        poly = MplPolygon(pts, closed=True, edgecolor='cyan',
                          facecolor='cyan', alpha=0.15, lw=2, zorder=3)
        ax.add_patch(poly)
        cx, cy = pts[:,0].mean(), pts[:,1].mean()
        txt = ax.text(cx, cy, f"ROI {idx}", color='cyan',
                      fontsize=9, ha='center', va='center',
                      fontweight='bold', zorder=6)
        roi_artists.extend([poly, txt])
        current_pts.clear()
        redraw_current()
        print(f"  ROI {idx} 완성")

    def undo_last_roi():
        if not finished_rois: return
        finished_rois.pop()
        for _ in range(2):
            if roi_artists: roi_artists.pop().remove()
        print(f"  ROI 제거 (남은: {len(finished_rois)}개)")
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax or panning[0]: return
        if event.button == 1:
            if event.dblclick:
                finish_roi()
            else:
                current_pts.append((event.xdata, event.ydata))
                redraw_current()

    def on_key(event):
        if event.key == 'backspace':
            if current_pts:
                current_pts.pop(); redraw_current()
            elif finished_rois:
                undo_last_roi()
        elif event.key in ('enter', 'return'):
            finish_roi()
        elif event.key == 'escape':
            current_pts.clear(); redraw_current()

    # ── 버튼 ──────────────────────────────────────────
    ax_ok   = fig.add_axes([0.80, 0.01, 0.15, 0.05])
    ax_full = fig.add_axes([0.58, 0.01, 0.20, 0.05])
    ax_undo = fig.add_axes([0.40, 0.01, 0.16, 0.05])
    ax_chA  = fig.add_axes([0.02, 0.01, 0.10, 0.05])
    ax_chB  = fig.add_axes([0.13, 0.01, 0.10, 0.05])
    ax_chD  = fig.add_axes([0.24, 0.01, 0.10, 0.05])

    btn_ok   = Button(ax_ok,   'Analyze',    color='#27ae60', hovercolor='#2ecc71')
    btn_full = Button(ax_full, 'Full Image', color='#2980b9', hovercolor='#3498db')
    btn_undo = Button(ax_undo, 'Undo ROI',   color='#c0392b', hovercolor='#e74c3c')
    btn_chA  = Button(ax_chA,  '[ON] Ch-A',  color='#1a6b1a', hovercolor='#145214')
    btn_chB  = Button(ax_chB,  '[ON] Ch-B',  color='#6b1a1a', hovercolor='#521414')
    btn_chD  = Button(ax_chD,  '[ON] DAPI',  color='#1a1a6b', hovercolor='#141452')
    if not has_dapi: ax_chD.set_visible(False)

    def on_confirm(_):
        if current_pts: finish_roi()
        if not finished_rois:
            print("  ROI를 최소 1개 그려주세요."); return
        confirmed[0] = True; plt.close(fig)

    def on_full(_):
        finished_rois.clear()
        finished_rois.append(np.array([[0,0],[W,0],[W,H],[0,H]], dtype=float))
        confirmed[0] = True; plt.close(fig)

    def on_undo(_):
        if current_pts: current_pts.clear(); redraw_current()
        else: undo_last_roi()

    def make_toggle(idx, btn, on_color):
        def toggle(_):
            ch_state[idx] = not ch_state[idx]
            s = 'ON' if ch_state[idx] else 'OFF'
            labels = ['Ch-A', 'Ch-B', 'DAPI']
            btn.label.set_text(f'[{s}] {labels[idx]}')
            btn.ax.set_facecolor(on_color if ch_state[idx] else '#3d3d3d')
            schedule_reload()
        return toggle

    btn_ok.on_clicked(on_confirm)
    btn_full.on_clicked(on_full)
    btn_undo.on_clicked(on_undo)
    btn_chA.on_clicked(make_toggle(0, btn_chA, '#1a6b1a'))
    btn_chB.on_clicked(make_toggle(1, btn_chB, '#6b1a1a'))
    btn_chD.on_clicked(make_toggle(2, btn_chD, '#1a1a6b'))

    fig.canvas.mpl_connect('scroll_event',         on_scroll)
    fig.canvas.mpl_connect('button_press_event',   on_press)
    fig.canvas.mpl_connect('button_press_event',   on_click)
    fig.canvas.mpl_connect('motion_notify_event',  on_motion)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('key_press_event',      on_key)

    print(f"\n  [ROI 선택창]  Left-click: vertex  |  Double-click/Enter: finish")
    print(f"  Scroll: zoom  |  Right-drag: pan  |  Backspace: undo")
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.show(block=True)

    if not confirmed[0]:
        return [np.array([[0,0],[W,0],[W,H],[0,H]], dtype=float)]

    result = []
    for pts in finished_rois:
        p = pts.copy()
        p[:,0] = np.clip(p[:,0], 0, W)
        p[:,1] = np.clip(p[:,1], 0, H)
        result.append(p)
    return result


# ─────────────────────────────────────────────
#  Puncta Detection + Watershed
# ─────────────────────────────────────────────
def detect_puncta(img2d, mask=None, method='otsu',
                  min_size=5, max_size=500, use_watershed=True):
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max
    from scipy.ndimage import distance_transform_edt

    img_f = img2d.astype(float)
    if mask is not None: img_f[~mask] = 0
    if img_f.max() == 0: return np.zeros_like(img2d, dtype=int)

    p2, p98 = np.percentile(img_f[img_f > 0], [2, 98])
    img_n   = np.clip((img_f - p2) / (p98 - p2 + 1e-9), 0, 1)

    if method == 'otsu':       thresh = filters.threshold_otsu(img_n)
    elif method == 'triangle': thresh = filters.threshold_triangle(img_n)
    else:                      thresh = float(method)

    binary = img_n > thresh
    if mask is not None: binary &= mask
    binary = morphology.remove_small_objects(binary, min_size=min_size)
    binary = morphology.remove_small_holes(binary, area_threshold=20)

    if use_watershed:
        dist   = distance_transform_edt(binary)
        coords = peak_local_max(dist, min_distance=3,
                                labels=binary, exclude_border=False)
        pm = np.zeros_like(dist, dtype=bool)
        if len(coords): pm[tuple(coords.T)] = True
        markers, _ = ndimage.label(pm)
        labelled = watershed(-dist, markers, mask=binary) if markers.max() > 0 \
                   else ndimage.label(binary)[0]
    else:
        labelled, _ = ndimage.label(binary,
                                    structure=ndimage.generate_binary_structure(2,2))

    for p in measure.regionprops(labelled):
        if p.area > max_size: labelled[labelled == p.label] = 0
    return labelled


# ─────────────────────────────────────────────
#  Polygon mask
# ─────────────────────────────────────────────
def polygon_to_mask(vertices, shape):
    from skimage.draw import polygon as sk_polygon
    H, W = shape
    xs = np.clip(vertices[:,0].astype(int), 0, W-1)
    ys = np.clip(vertices[:,1].astype(int), 0, H-1)
    rr, cc = sk_polygon(ys, xs, shape=(H, W))
    mask = np.zeros((H, W), dtype=bool)
    mask[rr, cc] = True
    return mask


# ─────────────────────────────────────────────
#  Threshold Preview
# ─────────────────────────────────────────────
def threshold_preview(img_a, img_b, mask, ch_a_name, ch_b_name):
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0: return 0.5, 0.5, 5, 500, 3.0

    r1,r2 = rows[0],rows[-1]+1
    c1,c2 = cols[0],cols[-1]+1
    crop_a    = img_a[r1:r2, c1:c2].astype(float)
    crop_b    = img_b[r1:r2, c1:c2].astype(float)
    crop_mask = mask[r1:r2, c1:c2]

    def norm_crop(arr, m):
        v = arr.copy(); v[~m] = 0
        if v.max() == 0: return np.zeros_like(v)
        p2,p98 = np.percentile(v[v>0],[2,98])
        return np.clip((v-p2)/(p98-p2+1e-9),0,1)

    norm_a = norm_crop(crop_a, crop_mask)
    norm_b = norm_crop(crop_b, crop_mask)

    MAX_PX = 600; H2,W2 = norm_a.shape
    sc = min(MAX_PX/max(W2,1), MAX_PX/max(H2,1), 1.0)
    Ph,Pw = max(1,int(H2*sc)), max(1,int(W2*sc))

    def thumb(arr):
        if sc < 1.0:
            return sk_resize(arr,(Ph,Pw),anti_aliasing=True,preserve_range=True)
        return arr.copy()

    init_ta = float(filters.threshold_otsu(norm_a[crop_mask])) if norm_a[crop_mask].std()>0 else 0.5
    init_tb = float(filters.threshold_otsu(norm_b[crop_mask])) if norm_b[crop_mask].std()>0 else 0.5

    params = {'ta':init_ta,'tb':init_tb,'mins':5,'maxs':500,'cd':3.0,'confirmed':False}

    fig = plt.figure(figsize=(14,9), facecolor='#1a1a1a')
    fig.suptitle('Threshold Preview  |  Adjust sliders -> Confirm & Analyze',
                 color='white', fontsize=11)

    ax_a  = fig.add_axes([0.02,0.25,0.30,0.65]); ax_a.set_facecolor('#1a1a1a'); ax_a.axis('off')
    ax_b  = fig.add_axes([0.35,0.25,0.30,0.65]); ax_b.set_facecolor('#1a1a1a'); ax_b.axis('off')
    ax_ov = fig.add_axes([0.68,0.25,0.30,0.65]); ax_ov.set_facecolor('#1a1a1a'); ax_ov.axis('off')

    ax_sa  = fig.add_axes([0.08,0.17,0.35,0.03],facecolor='#333')
    ax_sb  = fig.add_axes([0.08,0.12,0.35,0.03],facecolor='#333')
    ax_smin= fig.add_axes([0.55,0.17,0.35,0.03],facecolor='#333')
    ax_smax= fig.add_axes([0.55,0.12,0.35,0.03],facecolor='#333')
    ax_scd = fig.add_axes([0.55,0.07,0.35,0.03],facecolor='#333')

    sl_ta  = Slider(ax_sa,  f'{ch_a_name} Thresh',0.0,1.0,valinit=init_ta,color='#27ae60')
    sl_tb  = Slider(ax_sb,  f'{ch_b_name} Thresh',0.0,1.0,valinit=init_tb,color='#e74c3c')
    sl_min = Slider(ax_smin,'Min size (px)',1,50,valinit=5,valstep=1,color='#8e44ad')
    sl_max = Slider(ax_smax,'Max size (px)',50,2000,valinit=500,valstep=10,color='#8e44ad')
    sl_cd  = Slider(ax_scd, 'Coloc dist (px)',1.0,20.0,valinit=3.0,color='#2980b9')
    for sl in [sl_ta,sl_tb,sl_min,sl_max,sl_cd]:
        sl.label.set_color('white'); sl.valtext.set_color('yellow')

    im_a  = ax_a.imshow(thumb(norm_a),  cmap='Greens', vmin=0, vmax=1)
    im_b  = ax_b.imshow(thumb(norm_b),  cmap='Reds',   vmin=0, vmax=1)
    im_ov = ax_ov.imshow(np.zeros((Ph,Pw,3),dtype=np.uint8))
    cnt_a = ax_a.set_title(f'{ch_a_name}', color='#88ff88', fontsize=10)
    cnt_b = ax_b.set_title(f'{ch_b_name}', color='#ff8888', fontsize=10)
    ax_ov.set_title('Overlay (green/red/yellow=coloc)', color='white', fontsize=10)

    def get_lab(norm, thresh, mins, maxs, m):
        from skimage.segmentation import watershed
        from skimage.feature import peak_local_max
        from scipy.ndimage import distance_transform_edt
        binary = (norm > thresh) & m
        binary = morphology.remove_small_objects(binary, min_size=int(mins))
        dist   = distance_transform_edt(binary)
        coords = peak_local_max(dist, min_distance=3, labels=binary, exclude_border=False)
        pm = np.zeros_like(dist, dtype=bool)
        if len(coords): pm[tuple(coords.T)] = True
        markers, _ = ndimage.label(pm)
        lab = watershed(-dist, markers, mask=binary) if markers.max()>0 \
              else ndimage.label(binary)[0]
        for p in measure.regionprops(lab):
            if p.area > maxs: lab[lab==p.label] = 0
        return lab

    def update(_=None):
        ta,tb,mins,maxs,cd = sl_ta.val,sl_tb.val,sl_min.val,sl_max.val,sl_cd.val
        la = get_lab(norm_a,ta,mins,maxs,crop_mask)
        lb = get_lab(norm_b,tb,mins,maxs,crop_mask)
        na,nb = la.max(), lb.max()

        ov_a = np.stack([np.zeros((H2,W2)),norm_a,np.zeros((H2,W2))],axis=-1)
        ov_a[la>0] = [1,1,0]
        ov_b = np.stack([norm_b,np.zeros((H2,W2)),np.zeros((H2,W2))],axis=-1)
        ov_b[lb>0] = [1,1,0]
        im_a.set_data(thumb(ov_a)); im_a.set_clim(0,1)
        im_b.set_data(thumb(ov_b)); im_b.set_clim(0,1)

        pa_list = measure.regionprops(la); pb_list = measure.regionprops(lb)
        ca_arr = np.array([[p.centroid[0],p.centroid[1]] for p in pa_list]) if pa_list else np.empty((0,2))
        cb_arr = np.array([[p.centroid[0],p.centroid[1]] for p in pb_list]) if pb_list else np.empty((0,2))
        ids_a  = [p.label for p in pa_list]
        coloc_a= set()
        for i,ca in enumerate(ca_arr):
            if len(cb_arr):
                hits = np.where(np.sqrt(np.sum((cb_arr-ca)**2,axis=1))<=cd)[0]
                if len(hits): coloc_a.add(ids_a[i])

        rgb = np.zeros((H2,W2,3),dtype=np.uint8)
        base= (norm_a*50).astype(np.uint8)
        rgb[...,0]=base; rgb[...,1]=base; rgb[...,2]=base
        mask_ao = np.isin(la,[i for i in np.unique(la)[1:] if i not in coloc_a])
        mask_co = np.isin(la,list(coloc_a))
        rgb[mask_ao,1]=200; rgb[mask_co,0]=255; rgb[mask_co,1]=255
        im_ov.set_data(thumb(rgb.astype(float)/255))

        n_coloc = len(coloc_a)
        pct_a = round(100*n_coloc/na,1) if na else 0
        ax_a.set_title(f'{ch_a_name}  n={na}  thresh={ta:.2f}', color='#88ff88', fontsize=10)
        ax_b.set_title(f'{ch_b_name}  n={nb}  thresh={tb:.2f}', color='#ff8888', fontsize=10)
        ax_ov.set_title(f'Overlay  coloc={n_coloc}  A:{pct_a}%', color='white', fontsize=10)
        fig.canvas.draw_idle()

    for sl in [sl_ta,sl_tb,sl_min,sl_max,sl_cd]: sl.on_changed(update)

    ax_ok2  = fig.add_axes([0.80,0.01,0.15,0.05])
    ax_auto = fig.add_axes([0.60,0.01,0.18,0.05])
    btn_ok2  = Button(ax_ok2,  'Confirm & Analyze', color='#27ae60', hovercolor='#2ecc71')
    btn_auto = Button(ax_auto, 'Auto (Otsu)',        color='#2980b9', hovercolor='#3498db')

    def on_confirm2(_):
        params.update({'ta':sl_ta.val,'tb':sl_tb.val,'mins':int(sl_min.val),
                       'maxs':int(sl_max.val),'cd':sl_cd.val,'confirmed':True})
        plt.close(fig)

    def on_auto(_):
        if norm_a[crop_mask].std()>0: sl_ta.set_val(float(filters.threshold_otsu(norm_a[crop_mask])))
        if norm_b[crop_mask].std()>0: sl_tb.set_val(float(filters.threshold_otsu(norm_b[crop_mask])))
        update()

    btn_ok2.on_clicked(on_confirm2); btn_auto.on_clicked(on_auto)
    update()
    print("  [Threshold Preview] 슬라이더 조절 후 Confirm & Analyze 클릭")
    plt.show(block=True)
    return params['ta'], params['tb'], params['mins'], params['maxs'], params['cd']


# ─────────────────────────────────────────────
#  Colocalization
# ─────────────────────────────────────────────
def compute_colocalization(label_a, label_b, dist_thresh_px=3.0):
    props_a = measure.regionprops(label_a)
    props_b = measure.regionprops(label_b)
    if not props_a or not props_b:
        return {'n_a':len(props_a),'n_b':len(props_b),'n_coloc':0,
                'pct_a_coloc':0.0,'pct_b_coloc':0.0,
                'coloc_ids_a':[],'coloc_ids_b':[],
                'area_a_total_px':0,'area_b_total_px':0,
                'area_a_coloc_px':0,'area_b_coloc_px':0,
                'pct_area_a_coloc':0.0,'pct_area_b_coloc':0.0}

    cents_a = np.array([[p.centroid[0],p.centroid[1]] for p in props_a])
    cents_b = np.array([[p.centroid[0],p.centroid[1]] for p in props_b])
    ids_a   = [p.label for p in props_a]
    ids_b   = [p.label for p in props_b]
    coloc_a,coloc_b = set(),set()
    for i,ca in enumerate(cents_a):
        dists = np.sqrt(np.sum((cents_b-ca)**2,axis=1))
        hits  = np.where(dists<=dist_thresh_px)[0]
        if len(hits):
            coloc_a.add(ids_a[i])
            for h in hits: coloc_b.add(ids_b[h])

    area_a_total = sum(p.area for p in props_a)
    area_b_total = sum(p.area for p in props_b)
    area_a_coloc = sum(p.area for p in props_a if p.label in coloc_a)
    area_b_coloc = sum(p.area for p in props_b if p.label in coloc_b)

    return {
        'n_a':len(props_a),'n_b':len(props_b),
        'n_coloc':         len(coloc_a),
        'pct_a_coloc':     round(100*len(coloc_a)/len(props_a),2),
        'pct_b_coloc':     round(100*len(coloc_b)/len(props_b),2),
        'coloc_ids_a':     sorted(coloc_a),
        'coloc_ids_b':     sorted(coloc_b),
        'area_a_total_px': int(area_a_total),
        'area_b_total_px': int(area_b_total),
        'area_a_coloc_px': int(area_a_coloc),
        'area_b_coloc_px': int(area_b_coloc),
        'pct_area_a_coloc':round(100*area_a_coloc/area_a_total,2) if area_a_total else 0.0,
        'pct_area_b_coloc':round(100*area_b_coloc/area_b_total,2) if area_b_total else 0.0,
    }


# ─────────────────────────────────────────────
#  Overlay 저장
# ─────────────────────────────────────────────
def save_overlay(img_a, img_b, label_a, label_b, mask,
                 result, ch_a_name, ch_b_name, title, out_path):
    H,W = img_a.shape
    rgb = np.zeros((H,W,3),dtype=np.uint8)
    base= norm8(img_a)//5
    rgb[...,0]=base; rgb[...,1]=base; rgb[...,2]=base
    coloc_a = set(result['coloc_ids_a']); coloc_b = set(result['coloc_ids_b'])
    mask_ao = np.isin(label_a,[i for i in np.unique(label_a)[1:] if i not in coloc_a])
    mask_bo = np.isin(label_b,[i for i in np.unique(label_b)[1:] if i not in coloc_b])
    mask_co = np.isin(label_a,list(coloc_a))
    rgb[mask_ao,1]=np.clip(rgb[mask_ao,1].astype(int)+200,0,255)
    rgb[mask_bo,0]=np.clip(rgb[mask_bo,0].astype(int)+200,0,255)
    rgb[mask_co,0]=255; rgb[mask_co,1]=255
    if mask is not None:
        boundary = mask ^ morphology.binary_erosion(mask)
        rgb[boundary]=[0,255,255]

    fig,axes = plt.subplots(1,3,figsize=(18,6),facecolor='#1a1a1a')
    fig.suptitle(title,color='white',fontsize=12)
    for ax in axes: ax.set_facecolor('#1a1a1a'); ax.axis('off')
    axes[0].imshow(norm8(img_a),cmap='Greens')
    axes[0].set_title(f'{ch_a_name}  (n={result["n_a"]})',color='#88ff88',fontsize=11)
    axes[1].imshow(norm8(img_b),cmap='Reds')
    axes[1].set_title(f'{ch_b_name}  (n={result["n_b"]})',color='#ff8888',fontsize=11)
    axes[2].imshow(rgb)
    axes[2].set_title('Overlay (green/red/yellow=coloc/cyan=ROI)',color='white',fontsize=11)
    stat=(f"Colocalized: {result['n_coloc']}\n"
          f"{ch_a_name} coloc: {result['pct_a_coloc']}%\n"
          f"{ch_b_name} coloc: {result['pct_b_coloc']}%")
    axes[2].text(0.02,0.02,stat,transform=axes[2].transAxes,
                 color='white',fontsize=9,va='bottom',
                 bbox=dict(boxstyle='round',facecolor='#333',alpha=0.85))
    plt.tight_layout()
    plt.savefig(out_path,dpi=150,bbox_inches='tight',facecolor='#1a1a1a')
    plt.close()
    print(f"  저장: {out_path}")


# ─────────────────────────────────────────────
#  Main Pipeline
# ─────────────────────────────────────────────
def analyze_file(img_path, ch_a_idx=1, ch_b_idx=2, out_dir=None):
    name = Path(img_path).stem
    if out_dir is None:
        out_dir = str(Path(img_path).parent / 'coloc_results')
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}\n  파일: {name}\n{'='*60}")
    print("  [1/5] 파일 로드 중...")
    reader = load_image(img_path)
    ch_a_name = reader.channels[ch_a_idx]
    ch_b_name = reader.channels[ch_b_idx]

    # ROI 선택
    roi_polygons = select_roi_polygon(reader, ch_a_idx, ch_b_idx)
    print(f"\n  ROI {len(roi_polygons)}개 선택됨")

    # 원본 전체 이미지 로드 (분석용)
    print("  [2/5] 원본 이미지 로드 중...")
    full_data = reader.read_full()
    img_a_full = full_data[ch_a_idx]
    img_b_full = full_data[ch_b_idx]
    H, W = img_a_full.shape

    all_results = []
    for roi_idx, vertices in enumerate(roi_polygons):
        roi_label = f"ROI{roi_idx+1}"
        mask  = polygon_to_mask(vertices, (H, W))
        n_px  = int(mask.sum())
        print(f"\n  [{roi_label}] 면적: {n_px:,} px")

        print("  [3/5] Threshold 미리보기...")
        thresh_a, thresh_b, min_px, max_px, coloc_dist = threshold_preview(
            img_a_full, img_b_full, mask, ch_a_name, ch_b_name)
        print(f"  Thresh A={thresh_a:.3f}  B={thresh_b:.3f}  min={min_px}  max={max_px}  dist={coloc_dist}")

        print("  [4/5] Puncta 검출 중...")
        label_a = detect_puncta(img_a_full, mask=mask, method=thresh_a,
                                min_size=min_px, max_size=max_px)
        label_b = detect_puncta(img_b_full, mask=mask, method=thresh_b,
                                min_size=min_px, max_size=max_px)
        print(f"  {ch_a_name}: {label_a.max()} puncta")
        print(f"  {ch_b_name}: {label_b.max()} puncta")

        print("  [5/5] Colocalization 계산 중...")
        result = compute_colocalization(label_a, label_b, dist_thresh_px=coloc_dist)

        print(f"\n  -- 결과 --")
        print(f"  ROI 면적:              {n_px:,} px")
        print(f"  {ch_a_name} puncta:    {result['n_a']}  (면적: {result['area_a_total_px']:,} px)")
        print(f"  {ch_b_name} puncta:    {result['n_b']}  (면적: {result['area_b_total_px']:,} px)")
        print(f"  Colocalized:           {result['n_coloc']}")
        print(f"  {ch_a_name} coloc:     {result['pct_a_coloc']}%  (면적: {result['area_a_coloc_px']:,} px / {result['pct_area_a_coloc']}%)")
        print(f"  {ch_b_name} coloc:     {result['pct_b_coloc']}%  (면적: {result['area_b_coloc_px']:,} px / {result['pct_area_b_coloc']}%)")

        img_out = str(Path(out_dir) / f"{name}_{roi_label}_overlay.png")
        save_overlay(img_a_full, img_b_full, label_a, label_b, mask,
                     result, ch_a_name, ch_b_name,
                     title=f"{name} {roi_label}  |  {ch_a_name} vs {ch_b_name}",
                     out_path=img_out)

        result.update({'file':name,'roi':roi_idx+1,'roi_area_px':n_px,
                       'ch_a':ch_a_name,'ch_b':ch_b_name,
                       'thresh_a':thresh_a,'thresh_b':thresh_b,
                       'dist_px':coloc_dist})
        all_results.append(result)

    return all_results


def run_batch(img_files, ch_a_idx=1, ch_b_idx=2, out_dir='coloc_results'):
    all_results = []
    for f in img_files:
        try:
            results = analyze_file(f, ch_a_idx, ch_b_idx, out_dir)
            all_results.extend(results)
        except Exception as e:
            print(f"  [ERROR] {Path(f).name}: {e}")
            import traceback; traceback.print_exc()

    if all_results:
        df = pd.DataFrame([{
            'File':              r['file'],
            'ROI':               r['roi'],
            'ROI_area_px':       r['roi_area_px'],
            'Ch_A':              r['ch_a'],
            'Ch_B':              r['ch_b'],
            'n_ChA_puncta':      r['n_a'],
            'n_ChB_puncta':      r['n_b'],
            'n_coloc':           r['n_coloc'],
            'pct_ChA_coloc':     r['pct_a_coloc'],
            'pct_ChB_coloc':     r['pct_b_coloc'],
            'area_ChA_total_px': r['area_a_total_px'],
            'area_ChB_total_px': r['area_b_total_px'],
            'area_ChA_coloc_px': r['area_a_coloc_px'],
            'area_ChB_coloc_px': r['area_b_coloc_px'],
            'pct_area_ChA_coloc':r['pct_area_a_coloc'],
            'pct_area_ChB_coloc':r['pct_area_b_coloc'],
            'thresh_A':          r['thresh_a'],
            'thresh_B':          r['thresh_b'],
            'coloc_dist_px':     r['dist_px'],
            'analyzed_at':       datetime.now().strftime('%Y-%m-%d %H:%M'),
        } for r in all_results])

        csv_path = str(Path(out_dir) / 'colocalization_results.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n  CSV 저장: {csv_path}")
        print("\n" + "="*60)
        print(df[['File','ROI','n_ChA_puncta','n_ChB_puncta',
                  'n_coloc','pct_ChA_coloc','pct_ChB_coloc']].to_string(index=False))
        print("="*60)

    return all_results


# ─────────────────────────────────────────────
#  설정
# ─────────────────────────────────────────────
if __name__ == '__main__':

    IMG_FILES = [
        r'D:\CEH\E-NS-26-01\Ctbp2_Bassoon_260402\G1-1.czi',
        r'D:\CEH\E-NS-26-01\Ctbp2_Bassoon_260402\G2-1.czi',
    ]

    CH_A_INDEX = 1   # 0=DAPI  1=AF488  2=AF647
    CH_B_INDEX = 2

    OUT_DIR = r'D:\CEH\E-NS-26-01\Ctbp2_Bassoon_260402\coloc_results'

    valid = [f for f in IMG_FILES if os.path.exists(f)]
    if not valid:
        print("ERROR: 파일을 찾을 수 없습니다.")
        for f in IMG_FILES:
            print(f"  {f} -> {'OK' if os.path.exists(f) else 'MISSING'}")
        input("\nEnter to exit...")
        sys.exit(1)

    run_batch(img_files=valid, ch_a_idx=CH_A_INDEX,
              ch_b_idx=CH_B_INDEX, out_dir=OUT_DIR)

    print("\n[Done] Analysis complete!")
    input("Enter to exit...")
