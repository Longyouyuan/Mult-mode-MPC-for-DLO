import cv2
import numpy as np
import os

# ======== 在这里改参数 ========
# video_path = "./m2pc/m2pc_sin.mp4"  # 视频路径
# video_path = "./m2pc/m2pc_eight.mp4"  # 视频路径
video_path = "./m2pc/m2pc_continuous_eight.mp4"  # 视频路径
# video_path = "my_ctr_rope.mp4"  # 视频路径


if video_path == "./m2pc/m2pc_sin.mp4":
    # 单段模式原参数：保留不变；如果 use_frame_ranges=False，则仍按原来的方式运行
    n = 9 * 12  # 目标帧索引（0 基） 150/330/2s/5s/10s
    m = 5  # 向前每隔 m 帧取一次
    k = 23  # 总共取 k 帧（含第 n 帧）
    n = m * k + 15

    # 多段并排模式参数
    use_frame_ranges = True
    start_frame = 23
    end_frame = 173
    middle_frames = [start_frame + 50, start_frame + 100]
    include_range_start = True  # 如果按 m 递减没有刚好踩到区间起点，是否额外补上区间起点帧

    mode = "max"  # "exp" | "linear" | "max" | "color" | "trail"
    exp_lambda = 0.00  # 指数衰减的 lambda（仅 exp/color 用）
    show = True
    save = True
    output_path = "time_lapse_ranges_" + os.path.splitext(os.path.basename(video_path))[0] + ".png"
    label_time_fps = 60.0

    # 裁剪参数；设为 None 表示使用整张图
    crop_x_min = None
    crop_x_max = 1250
    crop_y_min = None
    crop_y_max = 1000  # 1000

if video_path == "./m2pc/m2pc_eight.mp4":
    # 单段模式原参数：保留不变；如果 use_frame_ranges=False，则仍按原来的方式运行
    n = 9 * 12  # 目标帧索引（0 基） 150/330/2s/5s/10s
    m = 5  # 向前每隔 m 帧取一次
    k = 23  # 总共取 k 帧（含第 n 帧）
    n = m * k + 15

    # 多段并排模式参数
    use_frame_ranges = True
    start_frame = 25
    end_frame = start_frame + 4*60  # 25 + 4*60  272
    middle_frames = [start_frame + 80, start_frame + 160]
    include_range_start = True  # 如果按 m 递减没有刚好踩到区间起点，是否额外补上区间起点帧

    mode = "max"  # "exp" | "linear" | "max" | "color" | "trail"
    exp_lambda = 0.00  # 指数衰减的 lambda（仅 exp/color 用）
    show = True
    save = True
    output_path = "time_lapse_ranges_" + os.path.splitext(os.path.basename(video_path))[0] + ".png"
    label_time_fps = 60.0

    # 裁剪参数；设为 None 表示使用整张图
    crop_x_min = None
    crop_x_max = 1250
    crop_y_min = None
    crop_y_max = 1000  # 1000

if video_path == "./m2pc/m2pc_continuous_eight.mp4":
    # 单段模式原参数：保留不变；如果 use_frame_ranges=False，则仍按原来的方式运行
    n = 9 * 12  # 目标帧索引（0 基） 150/330/2s/5s/10s
    m = 5  # 向前每隔 m 帧取一次
    k = 23  # 总共取 k 帧（含第 n 帧）
    n = m * k + 15

    # 多段并排模式参数
    use_frame_ranges = True
    start_frame = 32 + 4*60 -15
    end_frame = start_frame + 4*60
    middle_frames = [start_frame + 80, start_frame + 160]
    include_range_start = True  # 如果按 m 递减没有刚好踩到区间起点，是否额外补上区间起点帧

    mode = "max"  # "exp" | "linear" | "max" | "color" | "trail"
    exp_lambda = 0.00  # 指数衰减的 lambda（仅 exp/color 用）
    show = True
    save = True
    output_path = "time_lapse_ranges_" + os.path.splitext(os.path.basename(video_path))[0] + ".png"
    label_time_fps = 60.0

    # 裁剪参数；设为 None 表示使用整张图
    crop_x_min = None
    crop_x_max = 1250
    crop_y_min = None
    crop_y_max = 1000  # 1000
# ============================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(SCRIPT_DIR, path)


def get_video_info(video_path):
    video_path = resolve_path(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    info = {
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fourcc": int(cap.get(cv2.CAP_PROP_FOURCC)),
    }
    cap.release()

    if info["fps"] > 0:
        info["duration_sec"] = info["frame_count"] / info["fps"]
    else:
        info["duration_sec"] = None

    return info


def format_fourcc(fourcc_value):
    chars = [chr((fourcc_value >> (8 * i)) & 0xFF) for i in range(4)]
    code = "".join(chars).strip()
    return code or "unknown"


def print_video_info(video_path):
    info = get_video_info(video_path)
    print(f"Video: {resolve_path(video_path)}")
    print(f"Frames: {info['frame_count']}")
    print(f"FPS: {info['fps']:.3f}")
    print(f"Resolution: {info['width']} x {info['height']} pixels")
    if info["duration_sec"] is None:
        print("Duration: unknown (fps <= 0)")
    else:
        print(f"Duration: {info['duration_sec']:.3f} s")
    print(f"Codec: {format_fourcc(info['fourcc'])}")


def crop_image(img, x_min=None, x_max=None, y_min=None, y_max=None):
    h, w = img.shape[:2]

    x0 = 0 if x_min is None else int(x_min)
    x1 = w if x_max is None else int(x_max)
    y0 = 0 if y_min is None else int(y_min)
    y1 = h if y_max is None else int(y_max)

    if not (0 <= x0 < x1 <= w):
        raise ValueError(f"Invalid x crop range [{x0}, {x1}) for width {w}")
    if not (0 <= y0 < y1 <= h):
        raise ValueError(f"Invalid y crop range [{y0}, {y1}) for height {h}")

    return img[y0:y1, x0:x1]


def read_frame(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok:
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 阈值，比如亮度低于 60 就认为是黑布
    mask = gray < 0  # 75
    # 把对应像素设为纯黑
    frame[mask] = (0, 0, 0)

    return frame.astype(np.float32) / 255.0


def gather_frames(video_path, n, m, k):
    video_path = resolve_path(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n >= total:
        raise ValueError(f"n ({n}) out of range, total frames = {total}")

    frames, indices = [], []
    for i in range(k):
        idx = n - i * m
        if idx < 0:
            break
        f = read_frame(cap, idx)
        if f is None:
            break
        frames.append(f)     # frames[0] 最新，后面更旧
        indices.append(idx)
    cap.release()
    if not frames:
        raise RuntimeError("No frames collected.")
    return frames, indices


def gather_frames_between(video_path, start_idx, end_idx, m, include_start=True):
    """
    收集一个区间内的帧：从 end_idx 开始，以 m 为间隔向前取，直到 start_idx。
    例如 start=10, end=80, m=7 -> 80, 73, 66, ...
    如果 include_start=True 且最后没有刚好取到 10，则额外补上 10。

    返回的 frames 顺序保持和原 gather_frames 一致：frames[0] 是该区间最新/结束帧。
    """
    if start_idx > end_idx:
        raise ValueError(f"start_idx ({start_idx}) must be <= end_idx ({end_idx})")
    if m <= 0:
        raise ValueError("m must be positive")

    video_path = resolve_path(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end_idx >= total:
        cap.release()
        raise ValueError(f"end_idx ({end_idx}) out of range, total frames = {total}")
    if start_idx < 0:
        cap.release()
        raise ValueError("start_idx must be >= 0")

    indices = list(range(end_idx, start_idx - 1, -m))
    if include_start and indices[-1] != start_idx:
        indices.append(start_idx)

    frames = []
    used = []
    for idx in indices:
        f = read_frame(cap, idx)
        if f is None:
            break
        frames.append(f)
        used.append(idx)

    cap.release()
    if not frames:
        raise RuntimeError(f"No frames collected for range {start_idx}-{end_idx}.")
    return frames, used


def make_segments(start_idx, end_idx, middle_indices):
    """
    start=10, middle=[80,160], end=200 -> [(10,80), (80,160), (160,200)]
    """
    points = [start_idx] + sorted(middle_indices) + [end_idx]
    if len(set(points)) != len(points):
        raise ValueError("start_frame, middle_frames, end_frame must not contain duplicates")
    if points[0] != start_idx or points[-1] != end_idx:
        raise ValueError("All middle_frames must be between start_frame and end_frame")
    if any(points[i] >= points[i + 1] for i in range(len(points) - 1)):
        raise ValueError("Frames must be strictly increasing: start < middle... < end")
    return [(points[i], points[i + 1]) for i in range(len(points) - 1)]


def get_weights(mode, k, exp_lambda):
    if mode == "linear":
        w = np.linspace(1.0, 0.1, k, dtype=np.float32)
        w /= w.sum()
        return w
    if mode in ("exp", "color"):
        w = np.exp(-exp_lambda * np.arange(k, dtype=np.float32))
        w /= w.sum()
        return w
    if mode in ("max", "trail"):
        return np.ones((k,), dtype=np.float32)  # 占位
    raise ValueError("Unknown mode")


def temporal_color(i, k):
    # 当前(0)红 -> 最旧(k-1)蓝
    t = i / max(1, k - 1)
    r = 1.0 - t
    g = 0.0
    b = t
    return np.array([r, g, b], dtype=np.float32)


def compose(frames, mode="exp", exp_lambda=0.3):
    h, w, c = frames[0].shape
    k = len(frames)
    weights = get_weights(mode, k, exp_lambda)

    if mode in ("linear", "exp"):
        out = np.zeros((h, w, c), dtype=np.float32)
        for i, f in enumerate(frames):
            out += f * weights[i]
        return np.clip(out, 0.0, 1.0)

    if mode == "trail":
        # 不用最新帧做底，否则旧机械臂会被盖掉
        out = np.zeros_like(frames[-1])

        # 旧 -> 新
        for i, f in enumerate(frames[::-1]):
            t = i / max(1, k - 1)  # 0=最旧, 1=最新

            # 机械臂残影：旧的也保留，新的一定程度更明显
            ghost_alpha = 0.2 + 0.0 * t

            # 整帧透明叠加，机械臂历史也能出现
            out = out * (1 - ghost_alpha) + f * ghost_alpha

            # 绳子/亮轨迹额外增强
            gray = cv2.cvtColor((f * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
            mask = gray > 75
            mask = mask.astype(np.float32)
            mask = cv2.GaussianBlur(mask, (7, 7), 0)
            mask = mask[..., None]

            rope_alpha = 0.25 + 0.60 * t
            out = out * (1 - mask * rope_alpha) + f * (mask * rope_alpha)

        # 适当提亮，因为透明叠加后整体可能偏暗
        out = out / max(out.max(), 1e-6)

        return np.clip(out, 0.0, 1.0)

    if mode == "max":
        out = np.zeros((h, w, c), dtype=np.float32)
        alphas = np.linspace(1.0, 0.80, k, dtype=np.float32)
        for i, f in enumerate(frames):
            out = np.maximum(out, f * alphas[i])
        return np.clip(out, 0.0, 1.0)

    if mode == "color":
        out = np.zeros((h, w, c), dtype=np.float32)
        for i, f in enumerate(frames):
            color = temporal_color(i, k)  # (r,g,b)
            colored = f * color[None, None, :]
            out += colored * weights[i]
        maxv = out.max()
        if maxv > 0:
            out = out / maxv
        return np.clip(out, 0.0, 1.0)

    raise ValueError("Unknown mode")


def edit_and_show_image(frames, out, show=True, window_name="Composite", label_text=None):
    # 假设 frame_n 是第 n 帧（frames[0]）
    frame_n = np.clip(frames[0] * 255, 0, 255).astype(np.uint8)
    out_disp = np.clip(out * 255, 0, 255).astype(np.uint8)

    frame_n = crop_image(
        frame_n,
        x_min=crop_x_min,
        x_max=crop_x_max,
        y_min=crop_y_min,
        y_max=crop_y_max,
    )
    out_disp = crop_image(
        out_disp,
        x_min=crop_x_min,
        x_max=crop_x_max,
        y_min=crop_y_min,
        y_max=crop_y_max,
    )

    # 调整显示大小（例如宽度缩小到 50%）
    scale = 0.4
    h, w = out_disp.shape[:2]
    new_size = (int(w * scale), int(h * scale))

    frame_n_small = cv2.resize(frame_n, new_size)
    out_small = cv2.resize(out_disp, new_size)

    if label_text is not None:
        out_small = draw_label(out_small, label_text)

    # 并排拼接：这里保留原变量，但最终仍返回合成图 out_small
    combined = np.vstack((frame_n_small, out_small))

    # 显示
    if show:
        cv2.imshow(window_name, out_small)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return out_small


def draw_label(img, text):
    labeled = img.copy()
    cv2.rectangle(labeled, (8, 8), (160, 42), (0, 0, 0), thickness=-1)
    cv2.putText(
        labeled,
        text,
        (16, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return labeled


def format_latest_time(frame_idx, origin_frame, fps=60.0):
    if fps <= 0:
        raise ValueError("fps must be positive")
    seconds = (frame_idx - origin_frame) / fps
    return f"t={seconds:.2f}s"


def make_range_panel(video_path, start_idx, end_idx, middle_indices, m, mode, exp_lambda):
    panels = []
    all_used = []
    segments = make_segments(start_idx, end_idx, middle_indices)

    for seg_start, seg_end in segments:
        frames, used = gather_frames_between(
            video_path,
            seg_start,
            seg_end,
            m,
            include_start=include_range_start,
        )
        out = compose(frames, mode=mode, exp_lambda=exp_lambda)
        out_small = edit_and_show_image(
            frames,
            out,
            show=False,
            label_text=format_latest_time(seg_end, start_idx, fps=label_time_fps),
        )
        panels.append(out_small)
        all_used.append((seg_start, seg_end, used))

    return np.hstack(panels), all_used


def save_png(img, out_path):
    # 确保目录存在
    d = os.path.dirname(out_path)
    if d:
        os.makedirs(d, exist_ok=True)
    cv2.imwrite(out_path, img)


if __name__ == "__main__":
    print_video_info(video_path)

    if use_frame_ranges:
        out, used_by_segment = make_range_panel(
            video_path,
            start_frame,
            end_frame,
            middle_frames,
            m,
            mode,
            exp_lambda,
        )
        if show:
            cv2.imshow("Range composites", out)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if save:
            save_png(out, output_path)
        print("Saved:", output_path)
        for seg_start, seg_end, used in used_by_segment:
            print(f"Range {seg_start}-{seg_end}, frames used latest -> oldest:", used)
    else:
        frames, used = gather_frames(video_path, n, m, k)
        out = compose(frames, mode=mode, exp_lambda=exp_lambda)
        out = edit_and_show_image(frames, out, show=show)
        if save:
            save_png(out, output_path)
        print("Saved:", output_path)
        print("Frames used (latest -> oldest):", used)
