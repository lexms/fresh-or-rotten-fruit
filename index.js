const webcamElement = document.getElementById('webcam');
const inputElement = document.getElementById('input_image');

function preprocessing(image){
    inputTensor = tf.browser.fromPixels(image)
    inputTensor = inputTensor.expandDims(0);
    inputScale = inputTensor.div(255)
    //console.log(inputScale)
    return inputScale
}
async function app() {
    alert('Loading model. wait until success message shown');
    // Load the model.
    const model = await tf.loadLayersModel('json-model-2/model.json')
    model.compile({
        optimizer: 'adam',
        loss: 'binaryCrossentropy',
    })      
    alert('Successfully loaded model');


    // Predict via upload
    inputElement.addEventListener('change', function () {
        let prediction = 0.0

        // add image to preview
        var reader = new FileReader();
        reader.onload = function(){
        var output = document.getElementById('output_image');
        output.src = reader.result;
        }
        reader.readAsDataURL(event.target.files[0]);

        data = document.getElementById('output_image')
        input = preprocessing(data)
        prediction = model.predict(input);
        //prediction.print()
        prediction = prediction.flatten()
        prediction = prediction.arraySync()
        label = tf.where(prediction > 0.5, 1, 0) // 0 Fresh, 1 Rotten
        //label.print()
        label =label.arraySync()

        if (label == 0) {
            confident = 1.0-prediction
            predict_result = 'Fresh Fruit'
        }else{
            confident = prediction
            predict_result = 'Rotten Fruit'
        }
        
        confident = confident * 100
        document.getElementById('confident').innerText  = confident.toFixed(2) +'%';
        document.getElementById('prediction').innerText  = predict_result;

        input.dispose();
    });

    //Predict via video
    const webcam = await tf.data.webcam(webcamElement);
    while (true) {
        let prediction = 0.0
        captured = preprocessing(webcamElement)
        prediction = model.predict(captured);
        prediction = prediction.flatten()
        prediction = prediction.arraySync()
        label = tf.where(prediction > 0.5, 1, 0) // 0 Fresh, 1 Rotten
        label =label.arraySync()

        if (label == 0) {
            confident = 1.0-prediction
            predict_result = 'Fresh Fruit'
        }else{
            confident = prediction
            predict_result = 'Rotten Fruit'
        }     
        confident = confident * 100
        document.getElementById('confident-video').innerText  = confident.toFixed(2) +'%';
        document.getElementById('prediction-video').innerText  = predict_result;

        captured.dispose();
        await tf.nextFrame();
  }

}

app();