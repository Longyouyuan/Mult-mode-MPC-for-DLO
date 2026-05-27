import cv2
import numpy as np
import os

# ======== 在这里改参数 ========
# video_path = "./m2pc/m2pc_sin.mp4"  # 视频路径
# video_path = "./m2pc/m2pc_eight.mp4"  # 视频路径
# video_path = "./m2pc/m2pc_continuous_eight.mp4"  # 视频路径
# video_path = "my_ctr_rope.mp4"  # 视频路径


# video_path = "./m2pc/first-obs/m2pc_obs_first_5.mp4"  # 视频路径  380
# video_path = "./m2pc/first-obs/m2pc_obs_first_14.mp4"  # 视频路径  8*13
video_path = "./m2pc/first-obs/m2pc_obs_first_14_real.mp4"  # 视频路径  8*13
# video_path = "./m2pc/first-obs/m2pc_obs_first_18.mp4"  # 视频路径 806
# video_path = "./m2pc/second-obs/m2pc_obs_second_3.mp4"  # 视频路径  681
# video_path = "./m2pc/second-obs/m2pc_obs_second_4.mp4"  # 视频路径  538
# video_path = "./m2pc/second-obs/m2pc_obs_second_5.mp4"  # 视频路径  604


m = 8          # 向前每隔 m 帧取一次
k = 11        # 总共取 k 帧（含第 n 帧）
n = 788
mode = "max"   # "exp" | "linear" | "max" | "color" | "trail"
exp_lambda = 0.00   # 指数衰减的 λ（仅 exp/color 用）
save_output = True
output_path = None  # None 表示保存到原视频同目录，文件名与原视频相同，仅扩展名改为 .png

# 裁剪参数；设为 None 表示使用整张图
crop_x_min = None
crop_x_max = 1100
crop_y_min = None
crop_y_max = 950
# ============================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(SCRIPT_DIR, path)


def build_output_path(video_path, output_path=None):
    if output_path is not None:
        return resolve_path(output_path)

    video_abs_path = resolve_path(video_path)
    video_dir = os.path.dirname(video_abs_path)
    video_name = os.path.splitext(os.path.basename(video_abs_path))[0]
    return os.path.join(video_dir, f"{video_name}.png")


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


def get_video_metadata(video_path):
    video_path = resolve_path(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s = (frame_count / fps) if fps > 1e-9 else None
    cap.release()

    return {
        "path": video_path,
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_s": duration_s,
    }


def print_video_metadata(video_path):
    metadata = get_video_metadata(video_path)
    print("Video:", metadata["path"])
    print("FPS:", f"{metadata['fps']:.3f}")
    print("Total frames:", metadata["frame_count"])
    print("Resolution:", f"{metadata['width']} x {metadata['height']}")
    if metadata["duration_s"] is None:
        print("Duration:", "unknown")
    else:
        print("Duration [s]:", f"{metadata['duration_s']:.3f}")
    print("Sampling config:", f"n={n}, m={m}, k={k}, mode={mode}")

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
        if idx < 0: break
        f = read_frame(cap, idx)
        if f is None: break
        frames.append(f)     # frames[0] 最新，后面更旧
        indices.append(idx)
    cap.release()
    if not frames:
        raise RuntimeError("No frames collected.")
    return frames, indices

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
        if maxv > 0: out = out / maxv
        return np.clip(out, 0.0, 1.0)

    raise ValueError("Unknown mode")

def edit_and_show_image(frames, out, show=True):
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

    # 并排拼接
    combined = np.vstack((frame_n_small, out_small))

    # 显示
    if show:
        cv2.imshow("Original (left) vs Composite (right)", out_small)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return out_small

def save_png(img, out_path):

    # 确保目录存在
    d = os.path.dirname(out_path)
    if d: os.makedirs(d, exist_ok=True)
    cv2.imwrite(out_path, img)

if __name__ == "__main__":
    resolved_output_path = build_output_path(video_path, output_path=output_path)
    print_video_metadata(video_path)
    frames, used = gather_frames(video_path, n, m, k)
    out = compose(frames, mode=mode, exp_lambda=exp_lambda)
    out = edit_and_show_image(frames, out, show=True)
    if save_output:
        save_png(out, resolved_output_path)
        print("Saved:", resolved_output_path)
    else:
        print("Save disabled. Target path:", resolved_output_path)
    print("Frames used (latest -> oldest):", used)
