import cv2
import numpy as np
from moviepy.editor import *
import sys

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

if len(sys.argv) >= 3:
    video_name = sys.argv[1]
    final_name = sys.argv[2]
else:
    print("Недостаточно аргументов командной строки. Пожалуйста, укажите имена файлов")

#video_name = "Short1.mp4"
video_capture = cv2.VideoCapture(video_name)
video_movie = VideoFileClip(video_name)
rate = int(round(video_movie.fps , 0))
frame_len = video_movie.reader.nframes
seconds_in_video = frame_len/rate
 
print("Видео: " + video_name)
print("FPS: " + str(rate))
print("Продолжительность: " + str(seconds_to_minutes(seconds_in_video)))

template = cv2.imread("ex1.png", 0)

frame_num = 0
threshold = 0.8
empty_num = 0
empty_flag = False
empty_list = []
while video_capture.isOpened():
    # Захват кадра из видео
    ret, frame = video_capture.read()
    if not ret:
        break
    # Считываем кадр только раз в секунду
    if frame_num % rate == 0:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
        _ , max_val, _ , _ = cv2.minMaxLoc(result)
        if max_val > threshold:
            if not empty_flag:
                empty_flag = True
                empty_list.append(frame_num/rate)
        else:
            if empty_flag:
                empty_flag = False
                empty_list.append(frame_num/rate)

        percentage = (frame_num / frame_len) * 100
        sys.stdout.write(f"\rПрогресс анализа видео: {percentage:.2f}%")
        sys.stdout.flush()

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
    full_subclips.append(video_movie.subclip(start, end))

if len(empty_list) % 2 != 0:
    empty_list.append(round(seconds_in_video , 0))
formatted_pairs = format_pairs(empty_list)
print("Вырезанные куски: " + formatted_pairs)

final_clip = concatenate_videoclips(full_subclips)
final_clip.write_videofile(final_name)

video_capture.release()
cv2.destroyAllWindows()
