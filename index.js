
const webcamElement = document.getElementById('webcam');
const inputElement = document.getElementById('input_image');
const buttonElement = document.getElementById('button-on-off');
const labelText = document.getElementById('choose_file');
const sendButton = document.getElementById('send_btn');

function preprocessing(image){
    inputTensor = tf.browser.fromPixels(image)
    inputTensor = inputTensor.expandDims(0);
    inputScale = inputTensor.div(255)
    //console.log(inputScale)
    return inputScale
}
async function app() {

    //alert('Loading model. wait until success message shown');
    const model = await tf.loadLayersModel('../json-model-2/model.json')
    model.compile({
        optimizer: 'adam',
        loss: 'binaryCrossentropy',
    })      
    swal("Success", "Load model successfully, please click OK", "success");

    sendButton.addEventListener('click', function() {
        let prediction = 0.0
        $.LoadingOverlay("show"); //untuk memunculkan
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
        $.LoadingOverlay("hide"); //untuk menutup
    });


    // Predict via upload
    inputElement.addEventListener('change', function (event) {
        // add image to preview
        var reader = new FileReader();
        reader.onload = function(){
            var output = document.getElementById('output_image');
            output.src = reader.result;
        }
        reader.readAsDataURL(event.target.files[0]);
        labelText.innerHTML = event.target.files[0].name;
    });

    //Predict via video
    async function predict_video(){
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

    const webcam = await tf.data.webcam(webcamElement);
    buttonElement.addEventListener('click', function(){
        $.LoadingOverlay("show"); //untuk memunculkan
        currentvalue = buttonElement.innerText
        if(currentvalue == "Capture"){
            buttonElement.innerText = 'Captured'
            buttonElement.disabled = true
            setTimeout(function(){
                buttonElement.innerText = 'Capture'
                buttonElement.disabled = false
            },2000)
          }else{
            buttonElement.innerText = 'Capture'
           
          }
          predict_video()
         $.LoadingOverlay("hide"); //untuk memunculkan
    //     while (true) {
    //         let prediction = 0.0
    //         captured = preprocessing(webcamElement)
    //         prediction = model.predict(captured);
    //         prediction = prediction.flatten()
    //         prediction = prediction.arraySync()
    //         label = tf.where(prediction > 0.5, 1, 0) // 0 Fresh, 1 Rotten
    //         label =label.arraySync()
    
    //         if (label == 0) {
    //             confident = 1.0-prediction
    //             predict_result = 'Fresh Fruit'
    //         }else{
    //             confident = prediction
    //             predict_result = 'Rotten Fruit'
    //         }     
    //         confident = confident * 100
    //         document.getElementById('confident-video').innerText  = confident.toFixed(2) +'%';
    //         document.getElementById('prediction-video').innerText  = predict_result;
    
    //         captured.dispose();
    //         await tf.nextFrame();
    //   }
    })
    
}

app();