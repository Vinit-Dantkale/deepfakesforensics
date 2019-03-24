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
import face_alignment


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
    os.makedirs(self.real_landmarks)
    os.makedirs(self.fake_landmarks)
    os.makedirs(self.filter_dir)
  
  def get_alignments(self, video_type='fake', cpu=False, align_type='2d'):
    
    frame_dir = 'data/frames'
    align_dir = 'data/align'
    landmarks_type = face_alignment.LandmarksType._2D

    if align_type == '2d':
      landmarks_type = face_alignment.LandmarksType._2D
    elif align_type == '3d':
      landmarks_type = face_alignment.LandmarksType._3D
    else:
      raise Exception("Invalid alignment type!")

    if video_type.lower() == 'fake':
      
      frame_dir = self.fake_frames
      align_dir = self.fake_alignments

    elif video_type.lower() =='real':
      
      frame_dir = self.real_frames
      align_dir = self.real_alignments

    else:
      raise Exception('Invalid Video Type!')

    device = 'cuda'
    if cpu:
      device = 'cpu'
    else:
      device = 'cuda'

    imgs_list = os.listdir(frame_dir)

    fa = face_alignment.FaceAlignment(landmarks_type, flip_input=False, device=device, face_detector='sfd')

    alignment_file = os.path.join(align_dir, self.person_name, video_type, align_type + "_alignments.json")


    fout = open(alignment_file, 'w')

    for img_item in tqdm.tqdm(imgs_list):
      img_path = os.path.join(frame_dir, img_item)
      img = cv2.imread(img_path, cv2.IMREAD_COLOR)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      pred = fa.get_landmarks(img)
      alignment = {
        'frame': img_path,
        'alignments': pred
      }
      json.dump(alignment, fout)
    
    fout.close()
    
    
    
      

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
    landmarks_dir = 'data/landmarks'
  
    if type.lower() == 'real':

      video_dir = self.real_videos
      frame_dir = self.real_frames
      landmarks_dir = self.real_landmarks

    elif type.lower() == 'fake':

      video_dir = self.fake_videos
      frame_dir = self.fake_frames
      landmarks_dir = self.fake_landmarks

    else:
      
      raise Exception("Invalid type given")

    # Read video using cv2 library
    video_path = os.path.join(video_dir, video_name)
    capture = cv2.VideoCapture(video_path)

    # Get total number of frame in the video
    num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize counter variables
    frame_num = 0
    found_landmarks = []
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

        try:
          unknown_encoding = face_recognition.face_encodings(frame)[0]
          if face_recognition.compare_faces([known_encoding], unknown_encoding):

            frame_name = os.path.join(frame_dir, video_name.replace('.mp4', '') + "{:05d}.jpg".format(frame_iter))
            cv2.imwrite(frame_name, frame)

            found_landmarks.append(
              {
                'frame': frame_name,
                'landmarks': face_recognition.face_landmarks(frame),
                'encodings': unknown_encoding,
                'location': face_recognition.face_location(frame)
              }
            )

            frame_iter += 1

        except:
          pass

        frame_num += 1
        
      else:

        break
    
    string = "Number of frames with faces extracted: " + str(frame_iter) + "/" + str(frame_num) + " (" + str((frame_iter / frame_num) * 100) + "%)"
    print(string)
    landmarks_file = os.path.join(landmarks_dir, video_name.replace('.mp4', '') + "_landmarks.json")
    with open(landmarks_file, 'w') as fout:
      json.dump(found_landmarks, fout)
    print("Wrote alignments to " + landmarks_file)

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

# Directory where the landmarks will be stored
  parser.add_argument('-rl', '--real_landmarks', action='store', type=str, default='data/landmarks/real', help='Directory where facial landmarks of the person in real videos will be stored')
  parser.add_argument('-fl', '--fake_landmarks', action='store', type=str, default='data/landmarks/fake', help='Directory where facial landmarks of the person in fake videos will be stored')

# Directory where alignments will be stored

  parser.add_argument('-ra', '--real_alignments', action='store', type=str, default='data/align/real', help='Directory where facial alignments of the person in real videos will be stored')
  parser.add_argument('-fa', '--fake_alignments', action='store', type=str, default='data/align/fake', help='Directory where facial alignments of the person in fake videos will be stored')

# Directory where the filter will be stored

  parser.add_argument('-f', '--filter_dir', action='store', type=str, default='data/filter/', help='Directory where the image of a known person is saved for findiing facial landmarks in videos')

# Video arguments
  parser.add_argument('-v', '--video_name', action='store', type=str, default='obama-talking-fake', help='With argument video: Title of downloaded video, \nWith argument extract: Title of the video to be extracted')
  parser.add_argument('-u', '--url', action='store', type=str, default='https://www.youtube.com/watch?v=cQ54GDm1eL0', help='URL of the video to be downloaded')

# Type of video [real or fake]

  parser.add_argument('-t', '--type', action='store', type=str, default='fake', help='Video is real or fake')

# Whether to use GPU or CPU in alignment part

  parser.add_argument('-c', '--cpu', action='store_true', default=False, help='Use CPU while finding alignments')

# Finding 3d alignments for the face

  parser.add_argument('-3d', '--align_3d', action='store_true', default=False, help='Find 3d alignments for the faces')

  args = parser.parse_args()

  forensics = DeepFakesForensics(args)

  if args.task == 'tree':
    forensics.create_tree()
  
  elif args.task == 'video':
    forensics.download_videos(video_name=args.video_name, url=args.url, type=args.type)

  elif args.task == 'extract':
    forensics.extract_person_frames(video_name=args.video_name + ".mp4", type=args.type)


  elif args.task == 'align':
    if args.align_3d:
      align_type = '3d'
    else:
      align_type = '2d'
    forensics.get_alignments(video_type=args.type, cpu=args.cpu, align_type=align_type)
  else:
    raise Exception("Probably not done yet")