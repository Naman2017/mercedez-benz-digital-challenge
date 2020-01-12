Webcam.set({
  width: 480,
  height: 480,
  image_format: 'jpeg',
  jpeg_quality: 90
 });
Webcam.attach( '#my_camera' );


document.getElementById("snap").addEventListener("click",start);
var onGoing=0;

function start(){
  if(onGoing==0){
    onGoing=1;
    document.getElementById("snap").innerHTML="Stop"
    snapshot();
  }
  else{
    onGoing=0;
    document.getElementById("snap").innerHTML="Capture"
  }
}

function snapshot(){
  if(onGoing==0) return;
  var bi;
  Webcam.snap( function(data_uri) {
    bi=data_uri;
  const url="capture";
  const data={
    img : bi
  };
  $.post(url,data,function(v,status){
   console.log(data);
   document.getElementById("cbox").innerHTML=v.val
   snapshot();
  });
} );
  
}