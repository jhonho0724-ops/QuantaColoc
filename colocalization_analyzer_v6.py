# -*- coding: utf-8 -*-
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

"""
=============================================================================
  Colocalization Analyzer v4
  - 다각형 ROI (클릭으로 꼭짓점 추가, 더블클릭으로 완성)
  - 스크롤 휠 확대/축소, 우클릭 드래그 이동
  - 여러 ROI 추가 가능
=============================================================================
"""

import os, re, struct, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import ndimage
from skimage import filters, measure, morphology, draw as skdraw
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.widgets import Button
from datetime import datetime

warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
#  CZI / TIFF Loader (v3와 동일)
# ─────────────────────────────────────────────
def _read_metadata_xml(path):
    with open(path, 'rb') as f:
        pos = 0
        while True:
            f.seek(pos)
            hdr = f.read(32)
            if len(hdr) < 32: break
            sid = hdr[:16].rstrip(b'\x00').decode('ascii', 'ignore')
            alloc, _ = struct.unpack_from('<qq', hdr, 16)
            if sid == 'ZISRAWMETADATA':
                xml_size = struct.unpack('<i', f.read(4))[0]
                f.read(252)
                return f.read(xml_size).decode('utf-8', 'ignore')
            if alloc <= 0: break
            pos += 32 + alloc
    return ''

def _parse_directory(path):
    entries = []
    with open(path, 'rb') as f:
        pos = 0
        while True:
            f.seek(pos)
            hdr = f.read(32)
            if len(hdr) < 32: break
            sid = hdr[:16].rstrip(b'\x00').decode('ascii', 'ignore')
            alloc, _ = struct.unpack_from('<qq', hdr, 16)
            if sid == 'ZISRAWDIRECTORY':
                n = struct.unpack('<i', f.read(4))[0]
                f.read(124)
                for _ in range(n):
                    try:
                        f.read(2)
                        px_type = struct.unpack('<i', f.read(4))[0]
                        fpos    = struct.unpack('<q', f.read(8))[0]
                        f.read(4)
                        comp    = struct.unpack('<i', f.read(4))[0]
                        pyr     = struct.unpack('<B', f.read(1))[0]
                        f.read(5)
                        dc      = struct.unpack('<i', f.read(4))[0]
                        dims = {}
                        for _ in range(dc):
                            dn  = f.read(4).rstrip(b'\x00').decode('ascii', 'ignore')
                            ds  = struct.unpack('<i', f.read(4))[0]
                            dsz = struct.unpack('<I', f.read(4))[0]
                            f.read(8)
                            dims[dn] = {'start': ds, 'size': dsz}
                        entries.append({'fpos': fpos, 'comp': comp,
                                        'ptype': px_type, 'pyr': pyr, 'dims': dims})
                    except struct.error:
                        break
                break
            if alloc <= 0: break
            pos += 32 + alloc
    return entries

def _try_decode_jxr(raw_bytes):
    try:
        import imagecodecs
        return imagecodecs.jpegxr_decode(raw_bytes)
    except Exception:
        pass
    try:
        import subprocess, tempfile
        with tempfile.NamedTemporaryFile(suffix='.jxr', delete=False) as tf:
            tf.write(raw_bytes); tfname = tf.name
        out = tfname.replace('.jxr', '.tiff')
        r = subprocess.run(['convert', tfname, out], capture_output=True, timeout=30)
        os.unlink(tfname)
        if r.returncode == 0 and os.path.exists(out):
            from PIL import Image as PILImage
            arr = np.array(PILImage.open(out)); os.unlink(out)
            return arr
    except Exception:
        pass
    return None

def load_czi(path):
    xml    = _read_metadata_xml(path)
    fluors = list(dict.fromkeys(re.findall(r'<Fluor>(.*?)</Fluor>', xml)))
    sx     = int(re.search(r'<SizeX>(\d+)', xml).group(1))
    sy     = int(re.search(r'<SizeY>(\d+)', xml).group(1))
    sc     = int(re.search(r'<SizeC>(\d+)', xml).group(1))
    if not fluors: fluors = [f'Ch{i}' for i in range(sc)]
    print(f"  크기: {sx} x {sy} px,  채널: {fluors}")
    entries  = _parse_directory(path)
    full_res = [e for e in entries if e['pyr']==0 and 'X' in e['dims'] and 'Y' in e['dims']]
    print(f"  타일: {len(full_res)}개")
    canvas = np.zeros((sc, sy, sx), dtype=np.uint16)
    with open(path, 'rb') as f:
        for i, e in enumerate(full_res):
            try:
                f.seek(e['fpos'])
                f.read(16); f.read(16)
                meta_sz = struct.unpack('<i', f.read(4))[0]
                att_sz  = struct.unpack('<i', f.read(4))[0]
                data_sz = struct.unpack('<q', f.read(8))[0]
                f.read(2); f.read(4); f.read(8); f.read(4); f.read(4); f.read(1); f.read(5)
                dc = struct.unpack('<i', f.read(4))[0]
                dims = {}
                for _ in range(dc):
                    dn  = f.read(4).rstrip(b'\x00').decode('ascii', 'ignore')
                    ds  = struct.unpack('<i', f.read(4))[0]
                    dsz = struct.unpack('<I', f.read(4))[0]
                    f.read(8)
                    dims[dn] = {'start': ds, 'size': dsz}
                f.read(meta_sz); f.read(att_sz)
                raw = f.read(data_sz)
                jxr_idx = raw.find(b'\x49\x49\xBC')
                if jxr_idx >= 0: raw = raw[jxr_idx:]
                tile = _try_decode_jxr(raw)
                if tile is None: continue
                c  = dims.get('C',{}).get('start',0)
                xs = dims.get('X',{}).get('start',0)
                ys = dims.get('Y',{}).get('start',0)
                w  = dims.get('X',{}).get('size', tile.shape[1] if tile.ndim>1 else 1)
                h  = dims.get('Y',{}).get('size', tile.shape[0])
                if tile.ndim == 3: tile = tile[..., 0]
                tile = tile.astype(np.uint16)
                ye = min(ys+h, sy); xe = min(xs+w, sx)
                canvas[c, ys:ye, xs:xe] = tile[:ye-ys, :xe-xs]
                if (i+1) % 10 == 0: print(f"  타일 {i+1}/{len(full_res)} ...")
            except Exception:
                continue
    return {'channels': fluors, 'images': canvas}

def load_tiff(path):
    import tifffile
    with tifffile.TiffFile(path) as tf:
        arr = tf.asarray()
    if arr.ndim == 2:   arr = arr[np.newaxis]
    elif arr.ndim == 3:
        if arr.shape[2] <= 8: arr = arr.transpose(2, 0, 1)
    elif arr.ndim == 4: arr = arr[0] if arr.shape[1] <= 8 else arr[:, 0]
    return {'channels': [f'Ch{i}' for i in range(arr.shape[0])],
            'images': arr.astype(np.uint16)}

def load_image(path):
    ext = Path(path).suffix.lower()
    if ext == '.czi': return load_czi(path)
    if ext in ('.tif', '.tiff'): return load_tiff(path)
    raise ValueError(f"지원하지 않는 형식: {ext}")


# ─────────────────────────────────────────────
#  Polygon ROI Selector GUI
# ─────────────────────────────────────────────
def select_roi_polygon(img_a, img_b, ch_a_name, ch_b_name, file_name):
    """
    Returns: list of polygon vertex arrays in ORIGINAL image coordinates.
    Preview is downscaled for performance; coordinates are rescaled back.
    """

    def norm8(arr):
        v = arr.astype(float)
        if v.max() == 0: return np.zeros_like(arr, dtype=np.uint8)
        p2, p98 = np.percentile(v[v > 0], [2, 98])
        return np.clip((v - p2) / (p98 - p2 + 1e-9) * 255, 0, 255).astype(np.uint8)

    H, W = img_a.shape

    # ── 미리보기 축소 (최대 2048px, 렉 방지) ─────────────────
    MAX_PX = 2048
    scale  = min(MAX_PX / max(W, 1), MAX_PX / max(H, 1), 1.0)
    Ph, Pw = max(1, int(H * scale)), max(1, int(W * scale))

    from skimage.transform import resize as sk_resize
    def make_thumb(arr):
        n = norm8(arr)
        if scale < 1.0:
            return (sk_resize(n.astype(float), (Ph, Pw),
                              anti_aliasing=True, preserve_range=True)).astype(np.uint8)
        return n

    print(f"  미리보기 생성 중... (원본 {W}x{H} -> 표시 {Pw}x{Ph}, 비율 {scale:.3f})")
    thumb_a = make_thumb(img_a)
    thumb_b = make_thumb(img_b)

    preview = np.zeros((Ph, Pw, 3), dtype=np.uint8)
    preview[..., 1] = thumb_a
    preview[..., 0] = thumb_b

    fig, ax = plt.subplots(figsize=(13, 9))
    fig.patch.set_facecolor('#1a1a1a')
    ax.set_facecolor('#1a1a1a')
    ax.imshow(preview, aspect='equal')
    ax.set_title(
        f"{file_name}    Green={ch_a_name}   Red={ch_b_name}\n"
        "좌클릭: 꼭짓점 추가  |  더블클릭: ROI 완성  |  "
        "스크롤: 확대/축소  |  우클릭 드래그: 이동  |  Backspace: 마지막 꼭짓점 삭제",
        color='white', fontsize=8, pad=6
    )
    ax.tick_params(colors='#555')
    for spine in ax.spines.values():
        spine.set_edgecolor('#555')

    # ── state ──────────────────────────────────────────
    current_pts  = []          # 현재 그리는 중인 꼭짓점 [(x,y),...]
    finished_rois = []         # 완성된 ROI vertex arrays
    temp_artists  = []         # 현재 다각형 임시 그리기 객체
    roi_artists   = []         # 완성된 ROI 그리기 객체
    confirmed     = [False]
    panning       = [False]
    pan_start     = [None]

    # ── zoom / pan ──────────────────────────────────────
    def on_scroll(event):
        if event.inaxes != ax: return
        xdata, ydata = event.xdata, event.ydata
        xlim = ax.get_xlim(); ylim = ax.get_ylim()
        factor = 0.85 if event.button == 'up' else 1.15
        ax.set_xlim([xdata + (x - xdata) * factor for x in xlim])
        ax.set_ylim([ydata + (y - ydata) * factor for y in ylim])
        fig.canvas.draw_idle()

    def on_press(event):
        if event.button == 3:          # 우클릭 드래그 시작
            panning[0] = True
            pan_start[0] = (event.x, event.y,
                            ax.get_xlim(), ax.get_ylim())

    def on_motion(event):
        if panning[0] and pan_start[0] and event.inaxes == ax:
            dx = event.x - pan_start[0][0]
            dy = event.y - pan_start[0][1]
            # pixel -> data coords
            inv = ax.transData.inverted()
            p0  = inv.transform((pan_start[0][0], pan_start[0][1]))
            p1  = inv.transform((event.x, event.y))
            ddx = p0[0] - p1[0]
            ddy = p0[1] - p1[1]
            xlim0, ylim0 = pan_start[0][2], pan_start[0][3]
            ax.set_xlim([xlim0[0]+ddx, xlim0[1]+ddx])
            ax.set_ylim([ylim0[0]+ddy, ylim0[1]+ddy])
            fig.canvas.draw_idle()

    def on_release(event):
        if event.button == 3:
            panning[0] = False

    # ── drawing helpers ────────────────────────────────
    def redraw_current():
        for a in temp_artists: a.remove()
        temp_artists.clear()
        if not current_pts: return
        xs = [p[0] for p in current_pts]
        ys = [p[1] for p in current_pts]
        # dots
        sc = ax.scatter(xs, ys, s=20, c='yellow', zorder=5)
        temp_artists.append(sc)
        # lines
        if len(current_pts) > 1:
            ln, = ax.plot(xs, ys, '-', color='yellow', lw=1.5, zorder=4)
            temp_artists.append(ln)
        # dashed closing line preview
        if len(current_pts) > 2:
            cl, = ax.plot([xs[-1], xs[0]], [ys[-1], ys[0]],
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
        # draw filled polygon
        poly = MplPolygon(pts, closed=True,
                          edgecolor='cyan', facecolor='cyan',
                          alpha=0.15, lw=2, zorder=3)
        ax.add_patch(poly)
        cx, cy = pts[:,0].mean(), pts[:,1].mean()
        txt = ax.text(cx, cy, f"ROI {idx}",
                      color='cyan', fontsize=9, ha='center', va='center',
                      fontweight='bold', zorder=6)
        roi_artists.extend([poly, txt])
        current_pts.clear()
        redraw_current()
        print(f"  ROI {idx} 완성 ({len(pts)} 꼭짓점)")

    def undo_last_roi():
        if not finished_rois: return
        finished_rois.pop()
        # remove last two artists (patch + text)
        for _ in range(2):
            if roi_artists:
                roi_artists.pop().remove()
        print(f"  마지막 ROI 제거 (남은 ROI: {len(finished_rois)}개)")
        fig.canvas.draw_idle()

    # ── click handler ──────────────────────────────────
    def on_click(event):
        if event.inaxes != ax: return
        if panning[0]: return
        if event.button == 1:                  # 좌클릭
            if event.dblclick:                 # 더블클릭 → 완성
                finish_roi()
            else:                              # 단클릭 → 꼭짓점 추가
                current_pts.append((event.xdata, event.ydata))
                redraw_current()
        elif event.button == 3 and not panning[0]:  # 우클릭(비드래그) → ROI 제거
            pass   # 드래그 기능이 우선

    # ── keyboard ──────────────────────────────────────
    def on_key(event):
        if event.key == 'backspace':
            if current_pts:
                current_pts.pop()
                redraw_current()
                print("  마지막 꼭짓점 삭제")
            elif finished_rois:
                undo_last_roi()
        elif event.key == 'escape':
            current_pts.clear()
            redraw_current()
            print("  현재 ROI 취소")
        elif event.key == 'enter':
            finish_roi()

    # ── buttons ───────────────────────────────────────
    ax_ok   = fig.add_axes([0.80, 0.01, 0.15, 0.05])
    ax_full = fig.add_axes([0.58, 0.01, 0.20, 0.05])
    ax_undo = fig.add_axes([0.40, 0.01, 0.16, 0.05])

    # 채널 토글 버튼
    ax_ch_a  = fig.add_axes([0.12, 0.01, 0.12, 0.05])
    ax_ch_b  = fig.add_axes([0.25, 0.01, 0.12, 0.05])

    btn_ok   = Button(ax_ok,   'Analyze', color='#27ae60', hovercolor='#2ecc71')
    btn_full = Button(ax_full, 'Full Image', color='#2980b9', hovercolor='#3498db')
    btn_undo = Button(ax_undo, 'Undo ROI',     color='#c0392b', hovercolor='#e74c3c')
    btn_ch_a = Button(ax_ch_a, f'[ON] Ch-A', color='#1a6b1a', hovercolor='#145214')
    btn_ch_b = Button(ax_ch_b, f'[ON] Ch-B', color='#6b1a1a', hovercolor='#521414')

    # 채널 표시 상태
    ch_state  = [True, True]   # [ch_a_on, ch_b_on]
    img_disp  = [None]         # 현재 imshow 객체

    def update_display():
        r = thumb_b if ch_state[1] else np.zeros((Ph, Pw), dtype=np.uint8)
        g = thumb_a if ch_state[0] else np.zeros((Ph, Pw), dtype=np.uint8)
        b = np.zeros((Ph, Pw), dtype=np.uint8)
        composite = np.stack([r, g, b], axis=-1)
        if img_disp[0] is not None:
            img_disp[0].set_data(composite)
        fig.canvas.draw_idle()

    img_disp[0] = ax.get_images()[0]   # 기존 imshow 객체 참조

    def toggle_ch_a(event):
        ch_state[0] = not ch_state[0]
        state_str = 'ON' if ch_state[0] else 'OFF'
        btn_ch_a.label.set_text(f'[{state_str}] Ch-A')
        btn_ch_a.ax.set_facecolor('#1a6b1a' if ch_state[0] else '#3d3d3d')
        update_display()

    def toggle_ch_b(event):
        ch_state[1] = not ch_state[1]
        state_str = 'ON' if ch_state[1] else 'OFF'
        btn_ch_b.label.set_text(f'[{state_str}] Ch-B')
        btn_ch_b.ax.set_facecolor('#6b1a1a' if ch_state[1] else '#3d3d3d')
        update_display()

    btn_ch_a.on_clicked(toggle_ch_a)
    btn_ch_b.on_clicked(toggle_ch_b)

    def on_confirm(event):
        if current_pts:          # 완성 안 된 ROI 자동 완성
            finish_roi()
        if not finished_rois:
            print("  ROI를 최소 1개 그려주세요.")
            return
        confirmed[0] = True
        plt.close(fig)

    def on_full(event):
        finished_rois.clear()
        finished_rois.append(np.array([[0,0],[Pw,0],[Pw,Ph],[0,Ph]], dtype=float))
        confirmed[0] = True
        plt.close(fig)

    def on_undo(event):
        if current_pts:
            current_pts.clear()
            redraw_current()
        else:
            undo_last_roi()

    btn_ok.on_clicked(on_confirm)
    btn_full.on_clicked(on_full)
    btn_undo.on_clicked(on_undo)
    btn_ch_a.on_clicked(toggle_ch_a)
    btn_ch_b.on_clicked(toggle_ch_b)

    fig.canvas.mpl_connect('scroll_event',        on_scroll)
    fig.canvas.mpl_connect('button_press_event',  on_press)
    fig.canvas.mpl_connect('button_press_event',  on_click)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_release_event',on_release)
    fig.canvas.mpl_connect('key_press_event',     on_key)

    print(f"\n  [ROI 선택창]")
    print(f"  좌클릭: 꼭짓점 추가  /  더블클릭 or Enter: ROI 완성")
    print(f"  스크롤 휠: 확대/축소  /  우클릭 드래그: 이동")
    print(f"  Backspace: 꼭짓점/Undo ROI  /  Esc: 현재 ROI 취소")

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.show(block=True)

    if not confirmed[0]:
        print("  창 닫힘. 전체 이미지로 분석합니다.")
        return [np.array([[0,0],[W,0],[W,H],[0,H]], dtype=float)]  # original coords

    # preview 좌표 -> 원본 좌표로 변환
    scaled_rois = []
    for pts in finished_rois:
        orig_pts = pts / scale   # (x/scale, y/scale)
        orig_pts[:, 0] = np.clip(orig_pts[:, 0], 0, W)
        orig_pts[:, 1] = np.clip(orig_pts[:, 1], 0, H)
        scaled_rois.append(orig_pts)
    return scaled_rois


# ─────────────────────────────────────────────
#  Polygon mask
# ─────────────────────────────────────────────
def polygon_to_mask(vertices, shape):
    """꼭짓점 배열 -> bool 마스크 (shape: H x W)"""
    from skimage.draw import polygon as sk_polygon
    H, W = shape
    xs = np.clip(vertices[:, 0].astype(int), 0, W-1)
    ys = np.clip(vertices[:, 1].astype(int), 0, H-1)
    rr, cc = sk_polygon(ys, xs, shape=(H, W))
    mask = np.zeros((H, W), dtype=bool)
    mask[rr, cc] = True
    return mask


# ─────────────────────────────────────────────
#  Puncta Detection
# ─────────────────────────────────────────────
def detect_puncta(img2d, mask=None, method='otsu',
                  min_size=5, max_size=500):
    img_f = img2d.astype(float)
    if mask is not None:
        img_f[~mask] = 0
    if img_f.max() == 0:
        return np.zeros_like(img2d, dtype=int)
    p2, p98 = np.percentile(img_f[img_f > 0], [2, 98])
    img_n   = np.clip((img_f - p2) / (p98 - p2 + 1e-9), 0, 1)
    if method == 'otsu':       thresh = filters.threshold_otsu(img_n)
    elif method == 'triangle': thresh = filters.threshold_triangle(img_n)
    else:                      thresh = float(method)
    binary = img_n > thresh
    if mask is not None: binary &= mask
    binary = morphology.remove_small_objects(binary, min_size=min_size)
    binary = morphology.remove_small_holes(binary, area_threshold=20)
    struct_el = ndimage.generate_binary_structure(2, 2)
    labelled, _ = ndimage.label(binary, structure=struct_el)
    for p in measure.regionprops(labelled):
        if p.area > max_size:
            labelled[labelled == p.label] = 0
    return labelled


# ─────────────────────────────────────────────
#  Colocalization
# ─────────────────────────────────────────────
def compute_colocalization(label_a, label_b, dist_thresh_px=3.0):
    props_a = measure.regionprops(label_a)
    props_b = measure.regionprops(label_b)
    if not props_a or not props_b:
        return {'n_a': len(props_a), 'n_b': len(props_b), 'n_coloc': 0,
                'pct_a_coloc': 0.0, 'pct_b_coloc': 0.0,
                'coloc_ids_a': [], 'coloc_ids_b': []}
    cents_a = np.array([[p.centroid[0], p.centroid[1]] for p in props_a])
    cents_b = np.array([[p.centroid[0], p.centroid[1]] for p in props_b])
    ids_a   = [p.label for p in props_a]
    ids_b   = [p.label for p in props_b]
    coloc_a, coloc_b = set(), set()
    for i, ca in enumerate(cents_a):
        dists = np.sqrt(np.sum((cents_b - ca)**2, axis=1))
        hits  = np.where(dists <= dist_thresh_px)[0]
        if len(hits):
            coloc_a.add(ids_a[i])
            for h in hits: coloc_b.add(ids_b[h])
    return {
        'n_a': len(props_a), 'n_b': len(props_b),
        'n_coloc':     len(coloc_a),
        'pct_a_coloc': round(100 * len(coloc_a) / len(props_a), 2),
        'pct_b_coloc': round(100 * len(coloc_b) / len(props_b), 2),
        'coloc_ids_a': sorted(coloc_a),
        'coloc_ids_b': sorted(coloc_b),
    }


# ─────────────────────────────────────────────
#  Overlay 저장
# ─────────────────────────────────────────────
def save_overlay(img_a, img_b, label_a, label_b, mask,
                 result, ch_a_name, ch_b_name, title, out_path):
    def norm8(arr):
        v = arr.astype(float)
        if v.max() == 0: return np.zeros_like(arr, dtype=np.uint8)
        p2, p98 = np.percentile(v[v > 0], [2, 98])
        return np.clip((v - p2) / (p98 - p2 + 1e-9) * 255, 0, 255).astype(np.uint8)

    H, W  = img_a.shape
    rgb   = np.zeros((H, W, 3), dtype=np.uint8)
    base  = norm8(img_a) // 5
    rgb[...,0]=base; rgb[...,1]=base; rgb[...,2]=base

    coloc_a = set(result['coloc_ids_a'])
    coloc_b = set(result['coloc_ids_b'])
    mask_ao = np.isin(label_a, [i for i in np.unique(label_a)[1:] if i not in coloc_a])
    mask_bo = np.isin(label_b, [i for i in np.unique(label_b)[1:] if i not in coloc_b])
    mask_co = np.isin(label_a, list(coloc_a))
    rgb[mask_ao,1] = np.clip(rgb[mask_ao,1].astype(int)+200,0,255)
    rgb[mask_bo,0] = np.clip(rgb[mask_bo,0].astype(int)+200,0,255)
    rgb[mask_co,0] = 255; rgb[mask_co,1] = 255

    # ROI 경계 표시
    if mask is not None:
        boundary = mask ^ morphology.binary_erosion(mask)
        rgb[boundary] = [0, 255, 255]

    fig, axes = plt.subplots(1, 3, figsize=(18,6), facecolor='#1a1a1a')
    fig.suptitle(title, color='white', fontsize=12)
    for ax in axes: ax.set_facecolor('#1a1a1a'); ax.axis('off')
    axes[0].imshow(norm8(img_a), cmap='Greens')
    axes[0].set_title(f'{ch_a_name}  (n={result["n_a"]})', color='#88ff88', fontsize=11)
    axes[1].imshow(norm8(img_b), cmap='Reds')
    axes[1].set_title(f'{ch_b_name}  (n={result["n_b"]})', color='#ff8888', fontsize=11)
    axes[2].imshow(rgb)
    axes[2].set_title('Overlay  (green/red/yellow=coloc / cyan=ROI)', color='white', fontsize=11)
    stat = (f"Colocalized: {result['n_coloc']}\n"
            f"{ch_a_name} coloc: {result['pct_a_coloc']}%\n"
            f"{ch_b_name} coloc: {result['pct_b_coloc']}%")
    axes[2].text(0.02, 0.02, stat, transform=axes[2].transAxes,
                 color='white', fontsize=9, va='bottom',
                 bbox=dict(boxstyle='round', facecolor='#333', alpha=0.85))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
    plt.close()
    print(f"  저장: {out_path}")



# ─────────────────────────────────────────────
#  Threshold Preview GUI
# ─────────────────────────────────────────────
def threshold_preview(img_a, img_b, mask, ch_a_name, ch_b_name):
    """
    ROI 내에서 threshold 슬라이더로 puncta 검출 미리보기.
    Returns: (thresh_a, thresh_b, min_size, max_size, coloc_dist)
    """
    from matplotlib.widgets import Slider, Button, RadioButtons
    from skimage.transform import resize as sk_resize

    # ROI 영역만 잘라서 미리보기
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return 0.5, 0.5, 5, 500, 3.0

    r1, r2 = rows[0], rows[-1]+1
    c1, c2 = cols[0], cols[-1]+1
    crop_a    = img_a[r1:r2, c1:c2].astype(float)
    crop_b    = img_b[r1:r2, c1:c2].astype(float)
    crop_mask = mask[r1:r2, c1:c2]

    def norm_crop(arr, m):
        v = arr.copy(); v[~m] = 0
        if v.max() == 0: return np.zeros_like(v)
        p2, p98 = np.percentile(v[v>0], [2,98])
        return np.clip((v - p2) / (p98 - p2 + 1e-9), 0, 1)

    norm_a = norm_crop(crop_a, crop_mask)
    norm_b = norm_crop(crop_b, crop_mask)

    # 미리보기 축소
    MAX_PX = 600
    H, W = norm_a.shape
    sc = min(MAX_PX/max(W,1), MAX_PX/max(H,1), 1.0)
    Ph, Pw = max(1,int(H*sc)), max(1,int(W*sc))

    def thumb(arr):
        if sc < 1.0:
            from skimage.transform import resize as sk_resize
            return sk_resize(arr, (Ph, Pw), anti_aliasing=True, preserve_range=True)
        return arr.copy()

    t_norm_a = thumb(norm_a)
    t_norm_b = thumb(norm_b)

    # 초기 otsu threshold
    init_ta = float(filters.threshold_otsu(norm_a[crop_mask])) if norm_a[crop_mask].std() > 0 else 0.5
    init_tb = float(filters.threshold_otsu(norm_b[crop_mask])) if norm_b[crop_mask].std() > 0 else 0.5

    params = {
        'thresh_a':   init_ta,
        'thresh_b':   init_tb,
        'min_size':   5,
        'max_size':   500,
        'coloc_dist': 3.0,
        'confirmed':  False,
    }

    fig = plt.figure(figsize=(14, 9), facecolor='#1a1a1a')
    fig.suptitle('Threshold Preview  |  Adjust sliders -> confirm to analyze',
                 color='white', fontsize=11)

    # axes layout
    ax_a   = fig.add_axes([0.02, 0.25, 0.30, 0.65])
    ax_b   = fig.add_axes([0.35, 0.25, 0.30, 0.65])
    ax_ov  = fig.add_axes([0.68, 0.25, 0.30, 0.65])

    for ax in [ax_a, ax_b, ax_ov]:
        ax.set_facecolor('#1a1a1a'); ax.axis('off')

    # sliders
    ax_sa  = fig.add_axes([0.08, 0.17, 0.35, 0.03], facecolor='#333')
    ax_sb  = fig.add_axes([0.08, 0.12, 0.35, 0.03], facecolor='#333')
    ax_smin= fig.add_axes([0.55, 0.17, 0.35, 0.03], facecolor='#333')
    ax_smax= fig.add_axes([0.55, 0.12, 0.35, 0.03], facecolor='#333')
    ax_scd = fig.add_axes([0.55, 0.07, 0.35, 0.03], facecolor='#333')

    sl_ta  = Slider(ax_sa,  f'{ch_a_name} Thresh', 0.0, 1.0, valinit=init_ta, color='#27ae60')
    sl_tb  = Slider(ax_sb,  f'{ch_b_name} Thresh', 0.0, 1.0, valinit=init_tb, color='#e74c3c')
    sl_min = Slider(ax_smin,'Min size (px)',        1,   50,  valinit=5,       valstep=1, color='#8e44ad')
    sl_max = Slider(ax_smax,'Max size (px)',        50,  2000,valinit=500,     valstep=10,color='#8e44ad')
    sl_cd  = Slider(ax_scd, 'Coloc dist (px)',      1.0, 20.0,valinit=3.0,    color='#2980b9')

    for sl in [sl_ta, sl_tb, sl_min, sl_max, sl_cd]:
        sl.label.set_color('white')
        sl.valtext.set_color('yellow')

    im_a  = ax_a.imshow(t_norm_a, cmap='Greens', vmin=0, vmax=1)
    im_b  = ax_b.imshow(t_norm_b, cmap='Reds',   vmin=0, vmax=1)
    im_ov = ax_ov.imshow(np.zeros((Ph, Pw, 3), dtype=np.uint8))

    cnt_a = ax_a.set_title(f'{ch_a_name}', color='#88ff88', fontsize=10)
    cnt_b = ax_b.set_title(f'{ch_b_name}', color='#ff8888', fontsize=10)
    ax_ov.set_title('Overlay (green/red/yellow=coloc)', color='white', fontsize=10)

    def get_binary(norm, thresh, min_s, max_s, m):
        binary = norm > thresh
        binary &= m if m.shape == norm.shape else True
        binary = morphology.remove_small_objects(binary, min_size=int(min_s))
        struct_el = ndimage.generate_binary_structure(2, 2)
        lab, _ = ndimage.label(binary, structure=struct_el)
        for p in measure.regionprops(lab):
            if p.area > max_s:
                lab[lab == p.label] = 0
        return lab

    def update(_=None):
        ta   = sl_ta.val
        tb   = sl_tb.val
        mins = int(sl_min.val)
        maxs = int(sl_max.val)
        cd   = sl_cd.val

        lab_a = get_binary(norm_a, ta, mins, maxs, crop_mask)
        lab_b = get_binary(norm_b, tb, mins, maxs, crop_mask)

        # thumb overlays
        ov_a = np.stack([np.zeros((H,W)), norm_a, np.zeros((H,W))], axis=-1)
        ov_a[lab_a > 0] = [1, 1, 0]   # yellow = detected
        ov_b = np.stack([norm_b, np.zeros((H,W)), np.zeros((H,W))], axis=-1)
        ov_b[lab_b > 0] = [1, 1, 0]

        im_a.set_data(thumb(ov_a))
        im_a.set_clim(0, 1)
        im_b.set_data(thumb(ov_b))
        im_b.set_clim(0, 1)

        # coloc overlay
        na = lab_a.max(); nb = lab_b.max()
        cents_a = np.array([[p.centroid[0], p.centroid[1]]
                            for p in measure.regionprops(lab_a)]) if na else np.empty((0,2))
        cents_b = np.array([[p.centroid[0], p.centroid[1]]
                            for p in measure.regionprops(lab_b)]) if nb else np.empty((0,2))
        ids_a = [p.label for p in measure.regionprops(lab_a)]
        ids_b = [p.label for p in measure.regionprops(lab_b)]
        coloc_a = set(); coloc_b = set()
        for i, ca in enumerate(cents_a):
            if len(cents_b):
                dists = np.sqrt(np.sum((cents_b - ca)**2, axis=1))
                hits  = np.where(dists <= cd)[0]
                if len(hits):
                    coloc_a.add(ids_a[i])
                    for h in hits: coloc_b.add(ids_b[h])

        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        base = (norm_a * 50).astype(np.uint8)
        rgb[...,0]=base; rgb[...,1]=base; rgb[...,2]=base
        mask_ao = np.isin(lab_a, [i for i in np.unique(lab_a)[1:] if i not in coloc_a])
        mask_bo = np.isin(lab_b, [i for i in np.unique(lab_b)[1:] if i not in coloc_b])
        mask_co = np.isin(lab_a, list(coloc_a))
        rgb[mask_ao,1] = 200
        rgb[mask_bo,0] = 200
        rgb[mask_co,0] = 255; rgb[mask_co,1] = 255
        im_ov.set_data(thumb(rgb.astype(float)/255))

        n_coloc = len(coloc_a)
        pct_a = round(100*n_coloc/na, 1) if na else 0
        pct_b = round(100*len(coloc_b)/nb, 1) if nb else 0
        cnt_a.set_text(f'{ch_a_name}  n={na}  thresh={ta:.2f}')
        cnt_b.set_text(f'{ch_b_name}  n={nb}  thresh={tb:.2f}')
        ax_ov.set_title(f'Overlay  coloc={n_coloc}  A:{pct_a}%  B:{pct_b}%',
                        color='white', fontsize=10)
        fig.canvas.draw_idle()

    sl_ta.on_changed(update)
    sl_tb.on_changed(update)
    sl_min.on_changed(update)
    sl_max.on_changed(update)
    sl_cd.on_changed(update)

    # buttons
    ax_ok   = fig.add_axes([0.80, 0.01, 0.15, 0.05])
    ax_auto = fig.add_axes([0.60, 0.01, 0.18, 0.05])
    btn_ok   = Button(ax_ok,   'Confirm & Analyze', color='#27ae60', hovercolor='#2ecc71')
    btn_auto = Button(ax_auto, 'Auto (Otsu)',       color='#2980b9', hovercolor='#3498db')

    def on_confirm(_):
        params['thresh_a']   = sl_ta.val
        params['thresh_b']   = sl_tb.val
        params['min_size']   = int(sl_min.val)
        params['max_size']   = int(sl_max.val)
        params['coloc_dist'] = sl_cd.val
        params['confirmed']  = True
        plt.close(fig)

    def on_auto(_):
        if norm_a[crop_mask].std() > 0:
            sl_ta.set_val(float(filters.threshold_otsu(norm_a[crop_mask])))
        if norm_b[crop_mask].std() > 0:
            sl_tb.set_val(float(filters.threshold_otsu(norm_b[crop_mask])))
        update()

    btn_ok.on_clicked(on_confirm)
    btn_auto.on_clicked(on_auto)

    update()
    print(f"  [Threshold Preview] 슬라이더로 조절 후 Confirm & Analyze 클릭")
    plt.show(block=True)

    return (params['thresh_a'], params['thresh_b'],
            params['min_size'], params['max_size'],
            params['coloc_dist'])

# ─────────────────────────────────────────────
#  Main Pipeline
# ─────────────────────────────────────────────
def analyze_file(img_path, ch_a_idx=1, ch_b_idx=2,
                 threshold_method='otsu', min_puncta_px=5,
                 max_puncta_px=500, coloc_dist_px=3.0, out_dir=None):

    name = Path(img_path).stem
    if out_dir is None:
        out_dir = str(Path(img_path).parent / 'coloc_results')
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  파일: {name}")
    print(f"{'='*60}")
    print("  [1/4] 파일 로드 중...")
    data = load_image(img_path)
    imgs = data['images']
    chs  = data['channels']
    ch_a_name = chs[ch_a_idx] if ch_a_idx < len(chs) else f'Ch{ch_a_idx}'
    ch_b_name = chs[ch_b_idx] if ch_b_idx < len(chs) else f'Ch{ch_b_idx}'

    img_a_full = imgs[ch_a_idx]
    img_b_full = imgs[ch_b_idx]
    H, W = img_a_full.shape

    # ROI 선택
    roi_polygons = select_roi_polygon(img_a_full, img_b_full,
                                      ch_a_name, ch_b_name, name)
    print(f"\n  ROI {len(roi_polygons)}개 선택됨")

    all_results = []

    for roi_idx, vertices in enumerate(roi_polygons):
        roi_label = f"ROI{roi_idx+1}" if len(roi_polygons) > 1 else "ROI1"
        mask = polygon_to_mask(vertices, (H, W))
        n_px = mask.sum()
        print(f"\n  [{roi_label}] 면적: {n_px:,} px")

        print(f"  [2/4] Threshold 미리보기...")
        thresh_a, thresh_b, min_puncta_px, max_puncta_px, coloc_dist_px = threshold_preview(
            img_a_full, img_b_full, mask, ch_a_name, ch_b_name
        )
        print(f"  Threshold - {ch_a_name}: {thresh_a:.3f}, {ch_b_name}: {thresh_b:.3f}")
        print(f"  Min size: {min_puncta_px}px, Max size: {max_puncta_px}px, Coloc dist: {coloc_dist_px}px")

        print(f"  [3/4] Puncta 검출 중...")
        label_a = detect_puncta(img_a_full, mask=mask, method=thresh_a,
                                min_size=min_puncta_px, max_size=max_puncta_px)
        label_b = detect_puncta(img_b_full, mask=mask, method=thresh_b,
                                min_size=min_puncta_px, max_size=max_puncta_px)
        print(f"  {ch_a_name}: {label_a.max()} puncta")
        print(f"  {ch_b_name}: {label_b.max()} puncta")

        print(f"  [4/4] Colocalization 계산 중...")
        result = compute_colocalization(label_a, label_b, dist_thresh_px=coloc_dist_px)

        print(f"\n  -- 결과 --")
        print(f"  {ch_a_name} puncta:     {result['n_a']}")
        print(f"  {ch_b_name} puncta:     {result['n_b']}")
        print(f"  Colocalized:        {result['n_coloc']}")
        print(f"  {ch_a_name} coloc:      {result['pct_a_coloc']}%")
        print(f"  {ch_b_name} coloc:      {result['pct_b_coloc']}%")

        img_out = str(Path(out_dir) / f"{name}_{roi_label}_overlay.png")
        save_overlay(img_a_full, img_b_full, label_a, label_b, mask,
                     result, ch_a_name, ch_b_name,
                     title=f"{name} {roi_label}  |  {ch_a_name} vs {ch_b_name}",
                     out_path=img_out)

        result.update({
            'file': name, 'roi': roi_idx+1,
            'roi_vertices': vertices.tolist(),
            'roi_area_px': int(n_px),
            'ch_a': ch_a_name, 'ch_b': ch_b_name,
            'threshold': threshold_method, 'dist_px': coloc_dist_px,
        })
        all_results.append(result)

    return all_results


def run_batch(img_files, ch_a_idx=1, ch_b_idx=2,
              threshold_method='otsu', min_puncta_px=5,
              max_puncta_px=500, coloc_dist_px=3.0,
              out_dir='coloc_results'):
    all_results = []
    for f in img_files:
        try:
            results = analyze_file(f, ch_a_idx, ch_b_idx, threshold_method,
                                   min_puncta_px, max_puncta_px, coloc_dist_px, out_dir)
            all_results.extend(results)
        except Exception as e:
            print(f"  [ERROR] {Path(f).name}: {e}")
            import traceback; traceback.print_exc()

    if all_results:
        df = pd.DataFrame([{
            'File':             r['file'],
            'ROI':              r['roi'],
            'ROI_area_px':      r['roi_area_px'],
            'Ch_A':             r['ch_a'],
            'Ch_B':             r['ch_b'],
            'n_ChA_puncta':     r['n_a'],
            'n_ChB_puncta':     r['n_b'],
            'n_coloc':          r['n_coloc'],
            'pct_ChA_coloc':    r['pct_a_coloc'],
            'pct_ChB_coloc':    r['pct_b_coloc'],
            'threshold_method': r['threshold'],
            'coloc_dist_px':    r['dist_px'],
            'analyzed_at':      datetime.now().strftime('%Y-%m-%d %H:%M'),
        } for r in all_results])
        csv_path = str(Path(out_dir) / 'colocalization_results.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n  CSV 저장: {csv_path}")
        print("\n" + "="*60)
        print(df[['File','ROI','roi_area_px','n_ChA_puncta','n_ChB_puncta',
                  'n_coloc','pct_ChA_coloc','pct_ChB_coloc']].to_string(index=False))
        print("="*60)
    return all_results


# ─────────────────────────────────────────────
#  설정
# ─────────────────────────────────────────────
if __name__ == '__main__':

    IMG_FILES = [
        r'C:\Users\user\Desktop\G1-1.czi',
        r'C:\Users\user\Desktop\G2-1.czi',
    ]

    CH_A_INDEX    = 1      # 0=DAPI  1=AF488  2=AF647
    CH_B_INDEX    = 2

    THRESHOLD     = 'otsu'
    MIN_PUNCTA_PX = 5
    MAX_PUNCTA_PX = 500
    COLOC_DIST_PX = 3.0

    OUT_DIR = r'C:\Users\user\Desktop\coloc_results'

    valid = [f for f in IMG_FILES if os.path.exists(f)]
    if not valid:
        print("ERROR: 파일을 찾을 수 없습니다.")
        for f in IMG_FILES:
            print(f"  {f} -> {'OK' if os.path.exists(f) else '없음'}")
        input("\n엔터를 누르면 종료합니다...")
        sys.exit(1)

    run_batch(img_files=valid, ch_a_idx=CH_A_INDEX, ch_b_idx=CH_B_INDEX,
              threshold_method=THRESHOLD, min_puncta_px=MIN_PUNCTA_PX,
              max_puncta_px=MAX_PUNCTA_PX, coloc_dist_px=COLOC_DIST_PX,
              out_dir=OUT_DIR)

    print("\n[완료] 분석 완료!")
    input("엔터를 누르면 종료합니다...")
