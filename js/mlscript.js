
function preprocessing(img){
    input = img
    input = input.expandDims(0)
    input = input.div(255)
    return input
}


async function predict_video(model){
    const videoElement = document.getElementById('cam')
    videoElement.width = 150
    videoElement.height = 150
    const cam = await tf.data.webcam(videoElement)
    const img = await cam.capture()
    img_processed = preprocessing(img)

    prediction = 0.0
    prediction = model.predict(img_processed);
    prediction = prediction.flatten() // flatten data to one tensor
    prediction = prediction.arraySync() // change tensor into array
    label = tf.where(prediction > 0.5, 1, 0) // 0 Fresh, 1 Rotten
    label = label.arraySync() // change tensor into array

    if (label == 0) {
        confident = 1.0-prediction
        predict_result = 'Fresh Fruit'
    }else{
        confident = prediction
        predict_result = 'Rotten Fruit'
    }     
    confident = confident * 100
    document.getElementById('confident-video').innerText  = confident.toFixed(2) +'%'
    document.getElementById('prediction-video').innerText  = predict_result


    img.dispose();
    tf.nextFrame();
}

async function predict_image(model){
    img = document.getElementById('output_image')
    img = tf.browser.fromPixels(img)
    img_processed = preprocessing(img)
    
    prediction = 0.0
    prediction = model.predict(img_processed)
    prediction = prediction.flatten() // flatten data to one tensor
    prediction = prediction.arraySync() // change tensor into array
    label = tf.where(prediction > 0.5, 1, 0) // 0 Fresh, 1 Rotten
    label = label.arraySync() // change tensor into array

    if (label == 0) {
        confident = 1.0-prediction
        predict_result = 'Fresh Fruit'
    }else{
        confident = prediction
        predict_result = 'Rotten Fruit'
    }
    confident = confident * 100
    document.getElementById('confident').innerText  = confident.toFixed(2) +'%'
    document.getElementById('prediction').innerText  = predict_result
    img_processed.dispose()
}

async function app(){
    //Load the model
    const model = await tf.loadLayersModel('json-model-2/model.json')
    model.compile({
        optimizer: 'adam',
        loss: 'binaryCrossentropy',
    })

    const btn_on_off = document.getElementById('button-on-off')
    btn_on_off.addEventListener('click', function () {
        currentvalue = btn_on_off.innerText
        if(currentvalue == "Capture"){
            btn_on_off.innerText = 'Captured'
            btn_on_off.disabled = true
            setTimeout(function(){
                btn_on_off.innerText = 'Capture'
                btn_on_off.disabled = false
            },1000)
          }else{
            btn_on_off.innerText = 'Capture'
           
          }
          predict_video(model)
    })

    const inputElement = document.getElementById("input_image")
    inputElement.addEventListener("change", function(){
        // add image to preview
        var reader = new FileReader();
        reader.onload = function(){
        var output = document.getElementById('output_image');
        output.src = reader.result;
        }
        reader.readAsDataURL(event.target.files[0]);

        predict_image(model)
    })

}
app()


