import argparse
import sys

import cv2
from skimage.metrics import structural_similarity as compare_ssim
from moviepy import VideoFileClip, concatenate_videoclips

def seconds_to_minutes(seconds):
    minutes = seconds // 60
    remaining_seconds = seconds % 60
    return f"{int(round(minutes, 0))} минут, {int(round(remaining_seconds, 0))} секунд"

def format_pairs(lst):
    pairs = []
    for i in range(0, len(lst), 2):
        if i+1 < len(lst):
            pairs.append(f"{lst[i]}->{lst[i+1]}")
        else:
            pairs.append(str(lst[i]))
    return ", ".join(pairs)

parser = argparse.ArgumentParser(description="Process Zoom lecture video")
parser.add_argument("video", help="Input video file")
parser.add_argument("output", help="Output video file")
parser.add_argument(
    "--use-cuda",
    action="store_true",
    help="Enable CUDA acceleration if available",
)
args = parser.parse_args()

video_name = args.video
final_name = args.output

use_cuda = False
if args.use_cuda:
    if hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0:
        use_cuda = True
    else:
        print("CUDA requested but no compatible device found. Falling back to CPU.")

#video_name = "Short1.mp4"
video_capture = cv2.VideoCapture(video_name)
video_movie = VideoFileClip(video_name)
rate = int(round(video_movie.fps , 0))
frame_len = video_movie.reader.n_frames
seconds_in_video = frame_len/rate
 
print("Видео: " + video_name)
print("FPS: " + str(rate))
print("Продолжительность: " + str(seconds_to_minutes(seconds_in_video)))

image1 = cv2.imread("ex1.png")

frame_num = 0
empty_flag = False
empty_list = []

if use_cuda:
    template_gpu = cv2.cuda_GpuMat()
    template_gpu.upload(cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY))
    matcher = cv2.cuda.createTemplateMatching(cv2.CV_8UC1, cv2.TM_CCOEFF_NORMED)
while video_capture.isOpened():
    # Захват кадра из видео
    ret, frame = video_capture.read()
    if not ret:
        break

    resized_gray1 = cv2.resize(image1, (frame.shape[1], frame.shape[0]))

    # Считываем кадр только раз в секунду
    if frame_num % rate == 0:
        if use_cuda:
            frame_gpu = cv2.cuda_GpuMat()
            frame_gpu.upload(frame)
            gray_gpu = cv2.cuda.cvtColor(frame_gpu, cv2.COLOR_BGR2GRAY)
            template_gpu.upload(cv2.cvtColor(resized_gray1, cv2.COLOR_BGR2GRAY))
            res = matcher.match(gray_gpu, template_gpu)
            result = res.download()
            _ , max_val, _ , _ = cv2.minMaxLoc(result)
            is_empty = max_val > 0.7
        else:
            gray1 = cv2.cvtColor(resized_gray1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ssim_index = compare_ssim(gray1, gray2)
            is_empty = ssim_index > 0.7

        if is_empty:
            if not empty_flag:
                empty_flag = True
                empty_list.append(frame_num / rate)
        else:
            if empty_flag:
                empty_flag = False
                empty_list.append(frame_num / rate)

        percentage = (frame_num / frame_len) * 100
        sys.stdout.write(f"\rПрогресс анализа видео: {percentage:.2f}%")
        sys.stdout.flush()

        #print("SSIM:", ssim_index)

    frame_num += 1

sys.stdout.write(f"\rПрогресс анализа видео: 100% \n")
sys.stdout.flush()

subclips = []
full_time = 0
for ptr_time in empty_list:
    if ptr_time == seconds_in_video:
        subclips.append(float(ptr_time))
    if ptr_time in empty_list:
        subclips.append(float(ptr_time))
    full_time += 1
if subclips[0] == 0:
    subclips = subclips[1:]
    subclips.append(float(seconds_in_video))

full_subclips = []
modified_list = list(zip(subclips[::2], subclips[1::2]))
for start, end in modified_list:
    #print(start, end)
    full_subclips.append(video_movie.subclipped(start, end))

if len(empty_list) % 2 != 0:
    empty_list.append(round(seconds_in_video , 0))
formatted_pairs = format_pairs(empty_list)
print("Вырезанные куски: " + formatted_pairs)

final_clip = concatenate_videoclips(full_subclips)
final_clip.write_videofile(final_name)

video_capture.release()
cv2.destroyAllWindows()
