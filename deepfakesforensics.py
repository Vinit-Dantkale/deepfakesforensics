import argparse
import os
import cv2
import face_recognition
from tqdm import tqdm
import json

def get_landmarks_fr(frames_dir='data/frames', landmarks_dir='data/landmarks', video_name='video.mp4'):

  image_list = os.listdir(frames_dir)
  landmarks = []

  for image_name in tqdm(image_list):

    image_path = os.path.join(frames_dir, image_name)
  
    image = face_recognition.load_image_file(image_path)
    face_landmarks = face_recognition.face_landmarks(image)
    face_locations = face_recognition.face_locations(image)
    
    landmarks.append(
      {
        'image': image_path,
        'landmarks': face_landmarks,
        'locations': face_locations
      }
    )

  landmarks_file = os.path.join(landmarks_dir, video_name + "_landmarks_fr.json")
  with open(landmarks_file, 'w') as fout:
    json.dump(landmarks, fout)
  


def extract_frames_person(video_path='data/videos', video_name='video.mp4', frame_dir='data/frames', filter_path='data/filter/filter.jpg'): 
   
  capture = cv2.VideoCapture(os.path.join(video_path, video_name))
  ret = True

  # Get total number of frame in the video
  num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
  
  # Initialize counter variables
  frame_num = 0
  frame_iter = frame_num
  
  known_image = face_recognition.load_image_file(filter_path)
  
  known_encoding = face_recognition.face_encodings(known_image)

  for frame_num in tqdm(range(num_frames)):
    
    if ret:
      
      ret, frame = capture.read()
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

      unknown_encoding = face_recognition.face_encodings(frame)
      
      if len(unknown_encoding) > 0:
        unknown_encoding = unknown_encoding[0]
      else:
        continue

      if face_recognition.compare_faces([known_encoding], unknown_encoding):

        frame_name = os.path.join(frame_dir, video_name, video_name.replace('.mp4', '') + "{:05d}.jpg".format(frame_iter))
        cv2.imwrite(frame_name, frame)

        frame_iter += 1

      frame_num += 1
      
    else:
      print("Cannot read next frame. Breaking away")
      break
  
  string = "Number of frames with faces extracted: " + str(frame_iter) + "/" + str(frame_num) + " (" + str((frame_iter / frame_num) * 100) + "%)"
  print(string)
  
class DeepFakesForensics:
  
  def __init__(self, args):
    self.person_name = args.person_name
    self.real_videos = os.path.join(args.real_videos, self.person_name)
    self.fake_videos = os.path.join(args.fake_videos, self.person_name)
    self.real_frames = os.path.join(args.real_frames, self.person_name)
    self.fake_frames = os.path.join(args.fake_frames, self.person_name)
    self.filter_dir = os.path.join(args.filter_dir, self.person_name)
    self.real_landmarks = os.path.join(args.real_landmarks, self.person_name)
    self.fake_landmarks = os.path.join(args.fake_landmarks, self.person_name)
    self.fps = 20

  def create_working_dir(self):
    os.makedirs(self.real_videos)
    os.makedirs(self.fake_videos)
    os.makedirs(self.real_frames)
    os.makedirs(self.fake_frames)
    os.makedirs(self.real_landmarks)
    os.makedirs(self.fake_landmarks)
    os.makedirs(self.filter_dir)

  def download_video(self, video_name='obama-talking-fake', url='https://www.youtube.com/watch?v=cQ54GDm1eL0', type='fake'):
    
    from youtube_dl import YoutubeDL

    video_dir = 'data/videos/real'

    if type.lower() == 'real':
      video_dir = self.real_videos

    elif type.lower() == 'fake':
      video_dir = self.fake_videos

    else:
      raise Exception("IllegalArgumentTypeException: type " + type + " is not a valid type!")

    options = {
      'format': 'best+[vcodec!*=avc1]',
      'outtmpl': os.path.join(video_dir, video_name),
      'merge_output_format' : 'mp4'
    }
  
    with YoutubeDL(options) as ydl:
      _ = ydl.download([url])

  def get_landmarks(self, extractor='fr'):
    
    if extractor != 'fr':
      return
    else:
      
      real_video_frames = os.listdir(self.fake_frames)
      fake_video_frames = os.listdir(self.fake_frames)

      for real_video_frame in real_video_frames:

        real_frame_dir = os.path.join(self.real_frames, real_video_frame)

        get_landmarks_fr(frames_dir=real_frame_dir, landmarks_dir=self.real_landmarks, video_name=real_video_frame)

        
      for fake_video_frame in fake_video_frames:

        fake_frame_dir = os.path.join(self.fake_frames, fake_video_frame)

        get_landmarks_fr(frames_dir=fake_frame_dir, landmarks_dir=self.fake_landmarks, video_name=fake_video_frame)

    
  def extract_frames(self):

    real_videos = os.listdir(self.real_videos)
    fake_videos = os.listdir(self.fake_videos)
    filter_path = os.path.join(self.filter_dir, self.person_name + ".jpg")

    for video in real_videos:

      extract_frames_person(video_path=self.real_videos, video_name=video, frame_dir=self.real_frames, filter_path=filter_path)
    
    for video in fake_videos:

      extract_frames_person(video_path=self.fake_videos, video_name=video, frame_dir=self.fake_frames, filter_path=filter_path)    

if __name__ == "__main__":

  """ Create argument parser"""

  parser = argparse.ArgumentParser()

  """ Add mandatory arguments to the parser"""
# Like task, which involves creating tree, downloading videos, 
# extracting, finding alignments and testing 
  parser.add_argument('task', choices=['tree', 'video', 'extract', 'align', 'test'], help="Choose the task you wish to perform:\n.1. tree - Create a working directory for the person whose video you wish to analyze.\nOptional parameters: {-rv, -fv, -rf, -ff, -rl, -fl, -f}{\n\n2. video - Download a video from youtube by giving it's url")

  """ Person name will also be mandatory"""

  parser.add_argument('person_name', type=str, default='obama', help='Name of the person whose video has been faked')

  """ Directory where videos will be stored"""

  parser.add_argument('-rv', '--real_videos', action='store', type=str, default='data/videos/real', help='Directory where real videos will be stored')
  parser.add_argument('-fv', '--fake_videos', action='store', default='data/videos/fake', help='Directory where fake videos will be stored')

  """ Directory where frames will be stored"""

  parser.add_argument('-rf', '--real_frames', action='store', type=str, default='data/frames/real', help='Directory where frames of the person in real videos will be stored')
  parser.add_argument('-ff', '--fake_frames', action='store', type=str, default='data/frames/fake', help='Directory where frames of the person in fake videos will be stored')

  """ Directory where the landmarks will be stored"""
  parser.add_argument('-rl', '--real_landmarks', action='store', type=str, default='data/landmarks/real', help='Directory where facial landmarks of the person in real videos will be stored')
  parser.add_argument('-fl', '--fake_landmarks', action='store', type=str, default='data/landmarks/fake', help='Directory where facial landmarks of the person in fake videos will be stored')

  """ Which type of landmark tenchnique to be used"""
  parser.add_argument('-lt', '--landmark_technique', action='store', type=str, default='fr', help='Which alignment technique to be used for landmarking. Default: face_recognition library')

  """ Directory where the filter will be stored"""

  parser.add_argument('-f', '--filter_dir', action='store', type=str, default='data/filter/', help='Directory where the image of a known person is saved for findiing facial landmarks in videos')

  """ Video arguments"""
  parser.add_argument('-v', '--video_name', action='store', type=str, default='obama-talking-fake', help='With argument video: Title of downloaded video, \nWith argument extract: Title of the video to be extracted')
  parser.add_argument('-u', '--url', action='store', type=str, default='https://www.youtube.com/watch?v=cQ54GDm1eL0', help='URL of the video to be downloaded')

  """ Type of video [real or fake]"""

  parser.add_argument('-vt', '--video_type', action='store', type=str, default='fake', help='Video is real or fake')

  """ Whether to use GPU or CPU in alignment part"""

  parser.add_argument('-c', '--cpu', action='store_true', default=False, help='Use CPU while finding alignments')

  """ Finding 3d alignments for the face"""

  parser.add_argument('-3d', '--align_3d', action='store_true', default=False, help='Find 3d alignments for the faces')

  args = parser.parse_args()

  forensics = DeepFakesForensics(args)

  if args.task == 'tree':
    forensics.create_working_dir()
  
  elif args.task == 'video':
    forensics.download_video(video_name=args.video_name, url=args.url, type=args.type)

  elif args.task == 'extract':
    forensics.extract_frames()
  
  elif args.task == 'align':
    forensics.get_landmarks()

  else:
    raise Exception("Probably not done yet")