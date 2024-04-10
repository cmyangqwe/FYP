document.getElementById('videoUpload').addEventListener('change', function(event) {
    var videoFile = event.target.files[0];
    var videoPreview = document.querySelector('.upload-video-preview');

    var video = document.createElement('video');
    video.src = URL.createObjectURL(videoFile);
    video.controls = true;
    video.autoplay = false;
    videoPreview.innerHTML = '';
    videoPreview.appendChild(video);
});

document.getElementById('photoUpload').addEventListener('change', function(event) {
    var photoFile = event.target.files[0];
    var photoPreview = document.querySelector('.upload-photo-preview');

    var img = document.createElement('img');
    img.src = URL.createObjectURL(photoFile);
    img.alt = 'Uploaded Photo';
    photoPreview.innerHTML = '';
    photoPreview.appendChild(img);
});

document.addEventListener('DOMContentLoaded', function() {
    const effects = document.querySelectorAll('.effect');

    // Set default mosaic effect
    const defaultEffect = 'mosaicEffect1';
    const defaultEffectElement = document.getElementById(defaultEffect);
    defaultEffectElement.classList.add('selected');

    // Set the default mosaic effect in the hidden input field
    document.getElementById('mosaic_effect').value = defaultEffect;

    // Add event listener for mosaic effect selection
    effects.forEach(effect => {
        effect.addEventListener('click', function() {
            // Remove the 'selected' class from all effects
            effects.forEach(effect => {
                effect.classList.remove('selected');
            });

            // Add the 'selected' class to the clicked effect
            this.classList.add('selected'); 
        });
    });

});

function selectEffect(effectName) {
    // Set the value of the hidden input field to the selected effect
    document.getElementById('mosaic_effect').value = effectName;
}

document.addEventListener('DOMContentLoaded', function() {
    // Add event listener for form submission
    document.querySelector('form').addEventListener('submit', function(event) {
        event.preventDefault(); // Prevent default form submission
        
        // Display loading container during video processing
        document.getElementById('loading-container').style.display = 'block';

        // Clear previous output video preview
        document.querySelector('.output-video-preview').innerHTML = '';

        // Create FormData object to send form data to server
        var formData = new FormData(this);

        // Send POST request to server
        fetch('/process_video', {
            method: 'POST',
            body: formData
        })
        .then(response => response.blob()) // Convert response to blob
        .then(blob => {
            // Hide loading container
            document.getElementById('loading-container').style.display = 'none';

            // Show output video preview
            var videoUrl = URL.createObjectURL(blob);
            var outputVideoPreview = document.querySelector('.output-video-preview');
            outputVideoPreview.innerHTML = `<video controls><source src="${videoUrl}" type="video/mp4"></video>`;
            outputVideoPreview.style.display = 'block';

            // Show download button
            var downloadButton = document.querySelector('.download-button');
            downloadButton.href = videoUrl;
            downloadButton.style.display = 'block';
        })
        .catch(error => console.error('Error:', error));
    });
});


