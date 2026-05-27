import cv2
import numpy as np
import os

# ================= 参数区：只需要改这里 =================

video_path = "./m2pc/m2pc_sin.mp4"   # 视频路径

start_time_s = 0.0                  # 从第几秒开始画
count = 12                          # time lapse 图片数量/叠加帧数
interval_s = 0.1                    # 每个 lapse 之间间隔多少秒

mode = "max"                        # "exp" | "linear" | "max" | "color"
exp_lambda = 0.35                   # 指数衰减参数，仅 exp/color 用

output_path = "time_lapse_output.png"

show = True                         # 是否弹窗显示
save = True                         # 是否保存 png

# 裁剪参数，和你原始代码保持类似逻辑
crop_y = 80
crop_x = 500

scale = 0.4                         # 显示/保存缩放比例

# =======================================================


def print_video_info(cap):
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    duration_s = total_frames / fps if fps > 0 else 0

    print("========== Video Info ==========")
    print(f"Video path   : {video_path}")
    print(f"Resolution   : {width} x {height}")
    print(f"FPS          : {fps:.3f}")
    print(f"Total frames : {total_frames}")
    print(f"Duration     : {duration_s:.3f} s")
    print("================================")
    print()

    return fps, total_frames, width, height, duration_s


def read_frame(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok:
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 亮度低于 60 的区域认为是黑布，设为纯黑
    mask = gray < 60
    frame[mask] = (0, 0, 0)

    return frame.astype(np.float32) / 255.0


def gather_frames_by_time(video_path, start_time_s, count, interval_s):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps, total_frames, width, height, duration_s = print_video_info(cap)

    frames = []
    indices = []
    times = []

    for i in range(count):
        t = start_time_s + i * interval_s
        idx = int(round(t * fps))

        if idx < 0:
            print(f"[Skip] time={t:.3f}s -> frame={idx}, negative frame index")
            continue

        if idx >= total_frames:
            print(f"[Stop] time={t:.3f}s -> frame={idx}, out of range")
            break

        frame = read_frame(cap, idx)
        if frame is None:
            print(f"[Stop] failed to read frame {idx}")
            break

        frames.append(frame)
        indices.append(idx)
        times.append(t)

    cap.release()

    if not frames:
        raise RuntimeError("No frames collected. Please check start_time_s/count/interval_s.")

    print("========== Frames Used =========")
    for i, (t, idx) in enumerate(zip(times, indices)):
        print(f"{i:02d}: time={t:.3f}s, frame={idx}")
    print("================================")
    print()

    return frames, indices, times


def get_weights(mode, k, exp_lambda):
    if mode == "linear":
        w = np.linspace(1.0, 0.1, k, dtype=np.float32)
        w /= w.sum()
        return w

    if mode in ("exp", "color"):
        w = np.exp(-exp_lambda * np.arange(k, dtype=np.float32))
        w /= w.sum()
        return w

    if mode == "max":
        return np.ones((k,), dtype=np.float32)

    raise ValueError('Unknown mode. Use "exp", "linear", "max", or "color".')


def temporal_color(i, k):
    # 当前帧红，越往后越蓝
    t = i / max(1, k - 1)
    r = 1.0 - t
    g = 0.0
    b = t
    return np.array([r, g, b], dtype=np.float32)


def compose(frames, mode="max", exp_lambda=0.35):
    h, w, c = frames[0].shape
    k = len(frames)
    weights = get_weights(mode, k, exp_lambda)

    if mode in ("linear", "exp"):
        out = np.zeros((h, w, c), dtype=np.float32)
        for i, f in enumerate(frames):
            out += f * weights[i]
        return np.clip(out, 0.0, 1.0)

    if mode == "max":
        out = np.zeros((h, w, c), dtype=np.float32)
        alphas = np.linspace(1.0, 0.1, k, dtype=np.float32)
        for i, f in enumerate(frames):
            out = np.maximum(out, f * alphas[i])
        return np.clip(out, 0.0, 1.0)

    if mode == "color":
        out = np.zeros((h, w, c), dtype=np.float32)
        for i, f in enumerate(frames):
            color = temporal_color(i, k)
            colored = f * color[None, None, :]
            out += colored * weights[i]

        maxv = out.max()
        if maxv > 0:
            out = out / maxv

        return np.clip(out, 0.0, 1.0)

    raise ValueError('Unknown mode. Use "exp", "linear", "max", or "color".')


def crop_and_resize(img):
    img_u8 = np.clip(img * 255, 0, 255).astype(np.uint8)

    h, w = img_u8.shape[:2]

    y1 = max(0, crop_y - 20)
    y2 = min(h, h - crop_y)
    x1 = max(0, crop_x)
    x2 = min(w, w - crop_x)

    img_crop = img_u8[y1:y2, x1:x2]

    if img_crop.size == 0:
        raise RuntimeError("Crop result is empty. Please reduce crop_x/crop_y.")

    new_size = (
        max(1, int(img_crop.shape[1] * scale)),
        max(1, int(img_crop.shape[0] * scale)),
    )

    return cv2.resize(img_crop, new_size)


def save_png(img, out_path):
    d = os.path.dirname(out_path)
    if d:
        os.makedirs(d, exist_ok=True)
    cv2.imwrite(out_path, img)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    frames, used_indices, used_times = gather_frames_by_time(
        video_path=video_path,
        start_time_s=start_time_s,
        count=count,
        interval_s=interval_s,
    )

    out = compose(frames, mode=mode, exp_lambda=exp_lambda)
    out_disp = crop_and_resize(out)

    if save:
        save_png(out_disp, output_path)

    if show:
        cv2.imshow("Time Lapse", out_disp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
