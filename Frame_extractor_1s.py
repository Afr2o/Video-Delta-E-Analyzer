import argparse
import math
import sys
from pathlib import Path

try:
    import cv2
except ImportError as exc:
    sys.exit(
        f"无法加载 OpenCV：{exc}\n"
        "请先运行：py -m pip install opencv-python"
    )


def timestamp(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}_{minutes:02d}_{seconds:02d}"


def create_output_folder(video):
    """创建独立文件夹；已有同名文件夹时添加序号。"""
    number = 1
    while True:
        name = video.stem if number == 1 else f"{video.stem}_{number}"
        folder = video.parent / name
        try:
            folder.mkdir()
            return folder
        except FileExistsError:
            number += 1


def save_png(frame, destination):
    """先编码，再由 Python 写入，支持中文截图保存路径。"""
    success, encoded = cv2.imencode(".png", frame)
    if not success:
        raise RuntimeError(f"PNG 编码失败：{destination.name}")

    temporary = destination.with_name(destination.name + ".part")
    try:
        with temporary.open("xb") as file:
            file.write(encoded.tobytes())
        temporary.rename(destination)
    finally:
        try:
            temporary.unlink()
        except OSError:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="每隔 1 秒从 AVI 视频中保存一张 PNG 截图。"
    )
    parser.add_argument("video", nargs="?", help="AVI 视频完整路径")
    args = parser.parse_args()

    capture = None
    output_folder = None
    saved = 0

    try:
        raw_path = (
            args.video
            if args.video is not None
            else input("请输入 AVI 视频完整路径：")
        ).strip()

        # 兼容从 Windows“复制文件地址”得到的带引号路径。
        if (
            len(raw_path) >= 2
            and raw_path[0] == raw_path[-1]
            and raw_path[0] in "\"'"
        ):
            raw_path = raw_path[1:-1]

        if not raw_path:
            raise ValueError("视频路径不能为空。")

        video = Path(raw_path).expanduser()
        if not video.is_absolute():
            raise ValueError(r"请输入完整路径，例如 D:\视频\实验.avi")
        if video.suffix.lower() != ".avi":
            raise ValueError("请选择 .avi 格式的视频。")
        if not video.is_file():
            raise FileNotFoundError(f"视频文件不存在或无法访问：{video}")

        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise RuntimeError(
                "无法打开视频。请检查文件是否损坏、编码是否受支持；"
                "也可将视频移到纯英文路径后重试。"
            )

        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise RuntimeError("无法获得有效帧率，不能按秒准确截图。")

        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        total_frames = (
            int(round(frame_count))
            if math.isfinite(frame_count) and frame_count > 0
            else 0
        )

        print(f"视频帧率：{fps:.3f} FPS")
        print("正在处理，按 Ctrl+C 可以停止。")

        frame_index = 0
        next_second = 0

        while True:
            success, frame = capture.read()
            if not success or frame is None or frame.size == 0:
                if frame_index == 0:
                    raise RuntimeError("无法读取第一帧，视频可能为空或无法解码。")
                if total_frames and frame_index < total_frames:
                    raise RuntimeError(
                        f"视频提前停止读取：已读 {frame_index}/{total_frames} 帧。"
                        "文件可能损坏，或视频帧数信息不准确。"
                    )
                break

            if output_folder is None:
                output_folder = create_output_folder(video)
                print(f"截图文件夹：{output_folder}")

            # 根据帧率选取每个整秒播放时对应的帧。
            # 使用原始小数帧率，避免取整帧率造成长期时间偏移。
            while frame_index >= int(next_second * fps + 1e-9):
                filename = f"{timestamp(next_second)}.png"
                save_png(frame, output_folder / filename)
                saved += 1
                next_second += 1
                print(f"已保存：{filename}")

            frame_index += 1

        print("处理完成。")
        return 0

    except KeyboardInterrupt:
        print("\n已停止，已完成的截图会保留。", file=sys.stderr)
        return 130
    except EOFError:
        print("未收到视频路径，请在终端中运行程序。", file=sys.stderr)
        return 1
    except (OSError, ValueError, RuntimeError, cv2.error) as exc:
        print(f"\n处理失败：{exc}", file=sys.stderr)
        print("请检查文件权限、剩余磁盘空间及视频文件。", file=sys.stderr)
        return 1
    finally:
        if capture is not None:
            capture.release()
        if output_folder is not None:
            print(f"已保存 {saved} 张截图，位置：{output_folder}")


if __name__ == "__main__":
    sys.exit(main())