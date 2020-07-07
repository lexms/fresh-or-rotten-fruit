let net;
const webcamElement = document.getElementById('webcam');
const inputElement = document.getElementById('input_image');
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
        var reader = new FileReader();
        reader.onload = function(){
        var output = document.getElementById('output_image');
        output.src = reader.result;
        }
        reader.readAsDataURL(event.target.files[0]);

        //Preprocessing input
        const imageElement = document.getElementById('output_image');
        inputTensor = tf.browser.fromPixels(imageElement)
        inputTensor = inputTensor.expandDims(0);

        //Preprocessing - Normalize input
        const inputMax = inputTensor.max();
        const inputMin = inputTensor.min();  
        const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
        //normalizedInputs.print()

        prediction = model.predict(normalizedInputs, 10);
        prediction = prediction.flatten()
        prediction = prediction.arraySync()
        label = tf.where(prediction > 0.5, 1, 0)
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

        output.dispose();
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