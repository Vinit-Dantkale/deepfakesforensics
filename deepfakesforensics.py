import os
import argparse
from youtube_dl import YoutubeDL
import cv2
from tqdm import tqdm
import sys
#from moviepy.video.io.VideoFileClip import VideoFileClip
import time
import face_recognition
import json

class DeepFakesForensics:
  
  def __init__(self, args):
    self.person_name = args.person_name
    self.real_videos = os.path.join(args.real_videos, self.person_name)
    self.fake_videos = os.path.join(args.fake_videos, self.person_name)
    self.real_frames = os.path.join(args.real_frames, self.person_name)
    self.fake_frames = os.path.join(args.fake_frames, self.person_name)
    self.filter_dir = os.path.join(args.filter_dir, self.person_name)
    self.real_encodings = os.path.join(args.real_encodings, self.person_name)
    self.fake_encodings = os.path.join(args.fake_encodings, self.person_name)
    self.real_alignments = os.path.join(args.real_alignments, self.person_name)
    self.fake_alignments = os.path.join(args.fake_alignments, self.person_name)
    self.fps = 20

  def create_tree(self):

    os.makedirs(self.real_videos)
    os.makedirs(self.fake_videos)
    os.makedirs(self.real_frames)
    os.makedirs(self.fake_frames)
    os.makedirs(self.real_alignments)
    os.makedirs(self.fake_alignments)
    os.makedirs(self.filter_dir)

  def download_videos(self, video_name='obama-talking-fake', url='https://www.youtube.com/watch?v=cQ54GDm1eL0', type='fake'):
    if type.lower() == 'real':  
      options = {
        'format': 'best+[vcodec!*=avc1]',
        'outtmpl': os.path.join(self.real_videos, video_name),
        'merge_output_format' : 'mp4'
      }
    elif type.lower() == 'fake':
      options = {
        'format': 'best+[vcodec!*=avc1]',
        'outtmpl': os.path.join(self.fake_videos, video_name),
        'fps': self.fps,
        'merge_output_format' : 'mp4'
      }
    else:
      raise Exception("Type is invalid")
    with YoutubeDL(options) as ydl:
      _ = ydl.download([url])


  def extract_person_frames(self, video_name, type = 'real'):
    
    video_dir = 'data/videos'
    frame_dir = 'data/frames'
    encoding_dir = 'data/encodings'
  
    if type.lower() == 'real':

      video_dir = self.real_videos
      frame_dir = self.real_frames
      encoding_dir = self.real_encodings

    elif type.lower() == 'fake':

      video_dir = self.fake_videos
      frame_dir = self.fake_frames
      encoding_dir = self.fake_encodings

    else:
      
      raise Exception("Invalid type given")

    # Read video using cv2 library
    video_path = os.path.join(video_dir, video_name)
    capture = cv2.VideoCapture(video_path)

    # Get total number of frame in the video
    num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize counter variables
    frame_num = 0
    unknown_encodings = []
    frame_iter = frame_num

    # Get known image and it's encoding
    known_image = face_recognition.load_image_file(os.path.join(self.filter_dir, self.person_name + ".jpg"))
    known_encoding = face_recognition.face_encodings(known_image)

    # Initial return value is true
    ret = True

    # Start extracting frames
    for frame_num in tqdm(range(num_frames)):
      if ret:
        
        ret, frame = capture.read()

        # turns out there is no need to convert BGR2RGB nowadays
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
          unknown_encoding = face_recognition.face_encodings(frame)[0]
          if face_recognition.compare_faces([known_encoding], unknown_encoding):

            frame_name = os.path.join(frame_dir, video_name.replace('.mp4', '') + "{:05d}.jpg".format(frame_iter))
            cv2.imwrite(frame_name, frame)

            unknown_encodings.append(
              {
                'frame': frame_name,
                'encodings': unknown_encoding
              }
            )

            frame_iter += 1

        except:
          pass

        frame_num += 1
        
      else:

        break
    
    string = "Number of frames with faces extracted: " + frame_iter + " ({}%)".format(frame_iter // frame_num) * 100
    sys.stdout.write(string)
    encoding_file = os.path.join(encoding_dir, video_name.replace('.mp4', '') + "_encodings.json")
    with open(encoding_file, 'w') as fout:
      json.dump(unknown_encodings, fout)
    sys.stdout.write("Wrote alignments to " + encoding_file)

if __name__ == "__main__":

# Create argument parser

  parser = argparse.ArgumentParser()

# Add mandatory arguments to the parser
# Like task, which involves creating tree, downloading videos, 
# extracting, finding alignments and testing 
  parser.add_argument('task', choices=['tree', 'video', 'extract', 'align', 'test'], help="Does stuff")

# Person name will also be mandatory

  parser.add_argument('person_name', type=str, default='obama', help='Name of the person whose video has been faked')

# Directory where videos will be stored

  parser.add_argument('-rv', '--real_videos', action='store', type=str, default='data/videos/real', help='Directory where real videos will be stored')
  parser.add_argument('-fv', '--fake_videos', action='store', default='data/videos/fake', help='Directory where fake videos will be stored')

# Directory where frames will be stored

  parser.add_argument('-rf', '--real_frames', action='store', type=str, default='data/frames/real', help='Directory where frames of the person in real videos will be stored')
  parser.add_argument('-ff', '--fake_frames', action='store', type=str, default='data/frames/fake', help='Directory where frames of the person in fake videos will be stored')

# Directory where the encodings will be stored
  parser.add_argument('-re', '--real_encodings', action='store', type=str, default='data/encodings/real', help='Directory where facial encodings of the person in real videos will be stored')
  parser.add_argument('-fe', '--fake_encodings', action='store', type=str, default='data/encodings/fake', help='Directory where facial encodings of the person in fake videos will be stored')

# Directory where alignments will be stored

  parser.add_argument('-ra', '--real_alignments', action='store', type=str, default='data/align/real', help='Directory where facial alignments of the person in real videos will be stored')
  parser.add_argument('-fa', '--fake_alignments', action='store', type=str, default='data/align/fake', help='Directory where facial alignments of the person in fake videos will be stored')

# Directory where the filter will be stored

  parser.add_argument('-f', '--filter_dir', action='store', type=str, default='data/filter/', help='Directory where the image of a known person is saved for findiing facial encodings in videos')

# Video arguments
  parser.add_argument('-v', '--video_name', action='store', type=str, default='obama-talking-fake', help='With argument video: Title of downloaded video, \nWith argument extract: Title of the video to be extracted')
  parser.add_argument('-u', '--url', action='store', type=str, default='https://www.youtube.com/watch?v=cQ54GDm1eL0', help='URL of the video to be downloaded')

# Type of video [real or fake]

  parser.add_argument('-t', '--type', action='store', type=str, default='fake', help='Video is real or fake')

  args = parser.parse_args()

  forensics = DeepFakesForensics(args)

  if args.task == 'tree':
    forensics.create_tree()
  
  elif args.task == 'video':
    forensics.download_videos(video_name=args.video_name, url=args.url, type=args.type)

  elif args.task == 'extract':
    forensics.extract_person_frames(video_name=args.video_name + ".mp4", type=args.type)

  else:
    raise Exception("Probably not done yet")