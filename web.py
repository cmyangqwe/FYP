from flask import Flask, render_template, request, send_file
import os
from FinalModel import processVideo

app = Flask(__name__, static_url_path='/static')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return '', 404

@app.route('/process_video', methods=['POST'])
def process_video():
    # Check if the POST request has the files part
    if 'video' not in request.files or 'photo' not in request.files:
        return 'No file part'

    videoFile = request.files['video']
    photoFile = request.files['photo']
    mosaicEffect = request.form['mosaic_effect']

    # If the user does not select a file, the browser may send an empty file without a filename
    if videoFile.filename == '' or photoFile.filename == '':
        return 'No selected file'

    # Get filenames without extensions
    video_filename_no_ext = os.path.splitext(videoFile.filename)[0]
    photo_filename_no_ext = os.path.splitext(photoFile.filename)[0]

    # Get file extensions
    video_extension = os.path.splitext(videoFile.filename)[1]
    photo_extension = os.path.splitext(photoFile.filename)[1]

    # Save the uploaded video and photo files
    inputVideoPath = 'uploads/' + videoFile.filename
    inputPhotoPath = 'uploads/' + photoFile.filename
    videoFile.save(inputVideoPath)
    photoFile.save(inputPhotoPath)

    tempVideoFile = 'uploads/temp' + video_extension
    finalVideoFile = 'uploads/final' + video_extension

    # Call the process_video_function with video and photo paths
    outputVideo = processVideo(inputVideoPath, tempVideoFile, finalVideoFile, inputPhotoPath, mosaicEffect)
    
    # Return the processed video to the user for download
    return send_file(outputVideo, as_attachment=True)

if __name__ == '__main__':
    # Create uploads directory if it doesn't exist
    print(1)
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
