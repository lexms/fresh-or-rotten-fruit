let net;
const webcamElement = document.getElementById('webcam');
const inputElement = document.getElementById('input_image');

function preprocessing(image){
    inputTensor = tf.browser.fromPixels(image)
    inputTensor = inputTensor.expandDims(0);
    inputScale = inputTensor.div(255)
    console.log(inputScale)
    return inputScale
}
async function app() {
    console.log('Loading model..');
    // Load the model.
    const model = await tf.loadLayersModel('json-model-2/model.json')
    model.compile({
        optimizer: 'adam',
        loss: 'binaryCrossentropy',
    })      
    console.log('Successfully loaded model');



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
        prediction.print()
        prediction = prediction.flatten()
        prediction = prediction.arraySync()
        label = tf.where(prediction > 0.5, 1, 0) // 0 Fresh, 1 Rotten
        label.print()
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


    // Predict via camera

    // while (true){
    //     const webcam = await tf.data.webcam(webcamElement);  
    //     const img = await webcam.capture();
    //     console.log(img)
    //     const tensor = await tf.browser.fromPixels(img) 
        
    //     const prediction = model.predict(tensor);
    //     img.dispose();
    //     await tf.nextFrame();
    // }

}

app();