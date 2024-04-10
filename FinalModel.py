import argparse
import numpy as np
import cv2 as cv
from moviepy.editor import VideoFileClip

def str2bool(v):
    if v.lower() in ['on', 'yes', 'true', 'y', 't']:
        return True
    elif v.lower() in ['off', 'no', 'false', 'n', 'f']:
        return False
    else:
        raise NotImplementedError
parser = argparse.ArgumentParser()
parser.add_argument('--image1', '-i1', type=str, help='Path to the input image1. Omit for detecting on default camera.')
parser.add_argument('--image2', '-i2', type=str, help='Path to the input image2. When image1 and image2 parameters given then the program try to find a face on both images and runs face recognition algorithm.')
parser.add_argument('--video', '-v', type=str, help="Path to the vidoes")
parser.add_argument('--scale', '-sc', type=float, default=1.0, help='Scale factor used to resize input video frames.')
parser.add_argument('--face_detection_model', '-fd', type=str, default='face_detection_yunet_2023mar.onnx', help='Path to the face detection model. Download the model at https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet')
parser.add_argument('--face_recognition_model', '-fr', type=str, default='face_recognition_sface_2021dec.onnx', help='Path to the face recognition model. Download the model at https://github.com/opencv/opencv_zoo/tree/master/models/face_recognition_sface')
parser.add_argument('--score_threshold', type=float, default=0.9, help='Filtering out faces of score < score_threshold.')
parser.add_argument('--nms_threshold', type=float, default=0.3, help='Suppress bounding boxes of iou >= nms_threshold.')
parser.add_argument('--top_k', type=int, default=5000, help='Keep top_k bounding boxes before NMS.')
parser.add_argument('--save', '-s', type=str2bool, default=False, help='Set true to save results. This flag is invalid when using camera.')
parser.add_argument("-b", "--blocks", type=int, default=20,help="# of blocks for the pixelated blurring method")
args = parser.parse_args()

def addMosaic(input, faces, mosaicEffect):
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            print('Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'.format(idx, face[0], face[1], face[2], face[3], face[-1]))
            coords = face[:-1].astype(np.int32)
            face_start_x = coords[0]
            face_end_x = coords[0]+coords[2]
            face_start_y = coords[1]
            face_end_y = coords[1]+coords[3]
            size = (int(face[2]),int(face[3]))
            print(size)

            # Add mosaic effect
            if mosaicEffect == 'mosaicEffect1':        
                targetFace = input[face_start_y:face_end_y,face_start_x:face_end_x]
                tempFace = cv.resize(targetFace, (12, 12), interpolation=cv.INTER_LINEAR)
                mosaicedFace = cv.resize(tempFace, (coords[2], coords[3]), interpolation=cv.INTER_NEAREST)
                input[face_start_y:face_end_y,face_start_x:face_end_x] = mosaicedFace
            elif mosaicEffect == 'mosaicEffect2':
                mosaicEffect2 = cv.imread("static\\effect\\mosaicEffect2.png")
                mosaicEffect2 = cv.resize(mosaicEffect2,size)

                # Create a mask to identify non-white pixels in the mosaic effect image
                mask = cv.inRange(mosaicEffect2, np.array([0, 0, 0]), np.array([254, 254, 254]))

                # Extract the target region from the input image
                targetFace = input[face_start_y:face_end_y, face_start_x:face_end_x]

                # Apply the mosaic effect only to non-white pixels in the target region
                mosaicedFace = cv.bitwise_and(mosaicEffect2, mosaicEffect2, mask=mask)

                # Replace the target region in the input image with the mosaiced face
                input[face_start_y:face_end_y, face_start_x:face_end_x][mask != 0] = mosaicedFace[mask != 0]

                # targetFace = input[face_start_y:face_end_y,face_start_x:face_end_x]
                # mosaicedFace = mosaicEffect2
                # input[face_start_y:face_end_y,face_start_x:face_end_x] = mosaicedFace

            else:
                print('error')



def processVideo(inputVideo,processingVideo, finalVideo, photo, mosaicEffect):
    
    # Create the face detection model
    detector = cv.FaceDetectorYN.create(
        args.face_detection_model,
        "",
        (320, 320),
        args.score_threshold,
        args.nms_threshold,
        args.top_k
    )
    
    # Create the face recognition model
    recognizer = cv.FaceRecognizerSF.create(
        args.face_recognition_model,"")
    
    # Set all the video file locations
    inputVideoFile = inputVideo
    outputVideoFile = processingVideo
    finalVideoFile = finalVideo
    cap = cv.VideoCapture(inputVideoFile)

    frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)*args.scale)
    frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)*args.scale)
    fps = cap.get(cv.CAP_PROP_FPS)

    # Read the Photo
    targetFaceImg = cv.imread(cv.samples.findFile(photo))
    targetFaceImgWidth = int(targetFaceImg.shape[1]*args.scale)
    targetFaceImgHeight = int(targetFaceImg.shape[0]*args.scale)
    targetFaceImg = cv.resize(targetFaceImg, (targetFaceImgWidth, targetFaceImgHeight))

    
    size = (frameWidth, frameHeight)     
    editedVideo = cv.VideoWriter(outputVideoFile,  
                        cv.VideoWriter_fourcc(*'mp4v'), 
                        fps, size) 

    while cap.isOpened():
        hasFrame, frame = cap.read()
        if not hasFrame:
            print('No frames grabbed!')
            break
        
        frame = cv.resize(frame, (frameWidth, frameHeight))
        detector.setInputSize([frameWidth, frameHeight])
        faces = detector.detect(frame) # faces is a tuple

        try:
            face1_align = recognizer.alignCrop(frame, faces[1][0])
        except:
            editedVideo.write(frame)
            continue
        
        detector.setInputSize((targetFaceImgWidth, targetFaceImgHeight))
        targetFace = detector.detect(targetFaceImg)
        targetFace_align = recognizer.alignCrop(targetFaceImg, targetFace[1][0])

        # Extract features
        face1_feature = recognizer.feature(face1_align)
        face2_feature = recognizer.feature(targetFace_align)

        # Compare Faces
        cosine_score = recognizer.match(face1_feature, face2_feature, cv.FaceRecognizerSF_FR_COSINE)
        l2_score = recognizer.match(face1_feature, face2_feature, cv.FaceRecognizerSF_FR_NORM_L2)

        if cosine_score >= 0.363 and l2_score <= 1.128:
            addMosaic(frame, faces, mosaicEffect)

        editedVideo.write(frame) 

    cap.release() 
    editedVideo.release()            
    cv.destroyAllWindows()
    
    # Use moviepy to edit the audio
    inputClip = VideoFileClip(inputVideoFile)
    audioClip = inputClip.audio
    outputClip = VideoFileClip(outputVideoFile)
    outputClip = outputClip.set_audio(audioClip) 
    outputClip.write_videofile(finalVideoFile)

    return finalVideoFile

    