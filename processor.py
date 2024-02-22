import matplotlib.pyplot as plt
import numpy as np
import torch

from moviepy.editor import ColorClip, concatenate_videoclips, AudioFileClip
import demucs.api
import librosa

import yaml
from pytube import YouTube
import moviepy.editor as mp


def merge_nearby_points(timestamps, min_distance):
    """
    Merges points that are within the `min_distance` of each other.

    :param timestamps: A numpy array of timestamps.
    :param min_distance: The minimum distance between timestamps to be considered separate events.
    :return: A numpy array of merged timestamps.
    """
    merged_timestamps = [timestamps[0]]  # Start with the first timestamp
    for time in timestamps[1:]:
        if time - merged_timestamps[-1] < min_distance:
            # If the current timestamp is within min_distance of the last
            # added timestamp, merge it (i.e., do nothing)
            continue
        else:
            # If it's further away than min_distance, add it as a separate event
            merged_timestamps.append(time)
    return np.array(merged_timestamps)


def moving_stats(signal, window_size):
    # Calculate moving average
    moving_avg = np.convolve(signal, np.ones(window_size) / window_size, mode='same')
    
    # Calculate moving standard deviation
    moving_std = np.sqrt(np.convolve(np.square(signal - moving_avg), np.ones(window_size) / window_size, mode='same'))
    
    return moving_avg, moving_std


class Processor():
    def __init__(self, yt_url="", start = 0, end = 10, focus='bass') -> None:
        self.yt_url = yt_url
        self.separator = demucs.api.Separator(model="mdx_extra", segment=12)
        self.focus = focus

        self.start = start
        self.end = end
        self.duration = end - start
    
    def download(self, yt_url=None):
        if yt_url:
            filename = YouTube(yt_url).streams.first().download()
        else:
            filename = YouTube(self.yt_url).streams.first().download()
        
        clip = mp.VideoFileClip(filename)
        self.audio_file = "./temp/current_audio.mp3"
        clip.audio.write_audiofile(self.audio_file)
    
    def separate_audio(self):
        origin, separated = self.separator.separate_audio_file(self.audio_file)
        self.separated_audio = separated

        self.audio_data = torch.mean(separated[self.focus], axis=0)

        self.n_samples = len(self.audio_data)
        self.sample_freq = self.separator.samplerate

        time_length = self.n_samples / self.sample_freq
        assert self.duration < time_length, "Configurated duration is too long"

        self.focus_audio_data = self.audio_data[self.start*self.sample_freq:self.end*self.sample_freq]
        self.times = np.linspace(0, self.duration, num=self.duration * self.sample_freq)

    def process_audio(self, window_size=200, threshold_factor=2):
        # Calculate features
        self.stft = np.abs(librosa.stft(self.focus_audio_data.numpy()))
        self.spectral_centroids = librosa.feature.spectral_centroid(y=self.focus_audio_data.numpy(), sr=self.sample_freq)[0]
        self.spectral_rolloff = librosa.feature.spectral_rolloff(y=self.focus_audio_data.numpy(), sr=self.sample_freq)[0]

        # Calculate moving averages and standard deviations
        self.ma_centroids, std_centroids = moving_stats(self.spectral_centroids, window_size)
        self.ma_rolloff, std_rolloff = moving_stats(self.spectral_rolloff, window_size)

        # Detect anomalies when the signal exceeds the moving average by 2 standard deviations
        # This factor determines how many standard deviations to include
        anomaly_points_centroids = np.where(
            (self.spectral_centroids > (self.ma_centroids + threshold_factor * std_centroids))
            # | (self.spectral_centroids < (self.ma_centroids - threshold_factor * std_centroids))
        )[0]
        anomaly_points_rolloff = np.where(
            (self.spectral_rolloff > (self.ma_rolloff + threshold_factor * std_rolloff))
            # | (self.spectral_rolloff < (self.ma_rolloff - threshold_factor * std_rolloff))
        )[0]

        # Merge anomaly points and remove duplicates
        anomaly_points = np.union1d(anomaly_points_centroids, anomaly_points_rolloff)

        timestamps = librosa.frames_to_time(anomaly_points, sr=self.sample_freq)
        self.timestamps = merge_nearby_points(timestamps, .25)

        self.frame_times = librosa.frames_to_time(np.arange(len(self.spectral_centroids)), sr=self.sample_freq)
    
    def plot_audio(self):
        # Plotting
        plt.figure(figsize=(15, 10))

        # Plot audio data
        plt.subplot(3, 1, 1)  # 2 rows, 1 column, first subplot
        plt.plot(self.times, self.focus_audio_data)
        plt.title('Audio Waveform')
        plt.ylabel('Amplitude')
        plt.xlabel('Time (s)')
        # Mark change points on the audio waveform
        for timestamp in self.timestamps:
            plt.axvline(x=timestamp, color='b', linestyle='--')

        # Plot frame_diff data
        plt.subplot(3, 1, 2)  # 2 rows, 1 column, second subplot
        plt.plot(self.frame_times, self.spectral_centroids)  # Adjust frame_times if necessary
        plt.plot(self.frame_times, self.ma_centroids)  # Adjust frame_times if necessary
        plt.title('Spectral Centroids')
        plt.ylabel('Difference')
        plt.xlabel('Time (s)')
        # Mark change points on the frame_diff plot
        for timestamp in self.timestamps:
            plt.axvline(x=timestamp, color='b', linestyle='--')

        # Plot frame_diff data
        plt.subplot(3, 1, 3)  # 2 rows, 1 column, second subplot
        plt.plot(self.frame_times, self.spectral_rolloff)  # Adjust frame_times if necessary
        plt.plot(self.frame_times, self.ma_rolloff)  # Adjust frame_times if necessary
        plt.title('Spectral Rolloff')
        plt.ylabel('Difference')
        plt.xlabel('Time (s)')
        # Mark change points on the frame_diff plot
        for timestamp in self.timestamps:
            plt.axvline(x=timestamp, color='b', linestyle='--')

        plt.tight_layout()  # Adjust layout to not overlap
        plt.savefig("./temp/figure.png")
        plt.show()

    def save_config(self):
        timestamps_data = []
        for i in range(1, len(self.timestamps)):
            d = self.timestamps[i] - self.timestamps[i - 1]
            timestamps_data.append({
                "index": i,
                "duration": float(d),
                "frames": int(round(d * 8, 0)),
                "caption": "",
                "start_timestamp": float(self.timestamps[i - 1]),
                "end_timestamp": float(self.timestamps[i]),
            })

        data = {
            "metadata": {
                "audio_file": str(self.audio_file),
                "duration": int(self.duration),
                "start": int(self.start),
                "end": int(self.end)
            },
            "timestamps": timestamps_data,
        }

        with open("config.yaml", "w") as f:
            yaml.dump(data, f)
        
    def save_separated_audio(self):
        for file, source in self.separated_audio.items():
            print(file, source.shape)
            demucs.api.save_audio(source, f"./temp/{file}.mp3", samplerate=self.separator.samplerate)
    
    def save_sample_video(self):
        # Define duration of the video
        # Define the colors you want to use
        colors = [255, 126, 0]  # Replace with your actual colors

        # List to hold all the clips
        clips = []

        # Create a ColorClip for each segment
        start_ = 0
        for i, time in enumerate(self.timestamps):
            # Calculate the duration for this segment
            segment_duration = time - start_
            
            # Create the clip with the color and duration
            clip = ColorClip(size=(270, 480), color=colors[i % len(colors)], duration=segment_duration)

            # Append the clip to the list of clips
            clips.append(clip)
            
            # Update start time for the next segment
            start_ = time

        # Create the last clip till the end of the video
        clip = ColorClip(size=(270, 480), color=colors[-1], duration=(self.duration - self.start))
        clips.append(clip)

        # Concatenate all clips together
        final_clip = concatenate_videoclips(clips, method="compose")

        final_clip = final_clip.set_duration(self.duration)

        # Add the original audio
        audio = AudioFileClip(self.audio_file)
        audio_segment = audio.subclip(self.start, self.end)
        final_clip = final_clip.set_audio(audio_segment.set_duration(final_clip.duration))

        # Write the result to a file
        final_clip.write_videofile("./temp/output_video.mp4", fps=24)
    