//global variable
top.label = []; //cluster label
top.distanceMatrix = []; //distance matrix between all the clusters

//cache jquery
var $btn_ComputeCluster = $('#btn-ComputeCluster');
var $select_method = $('#select_clusterMethod');
var $select_kNN = $('#select_neighborSize');
var $input_clusterNum = $('#input_clusterNum');

function applySubspaceCluster() {
  //if the server is not ready do nothing
  if( $btn_ComputeCluster.html()!== 'Compute Clustering')
    return;

  //if no words is loaded do nothing
  if(top.words.length === 0) {
    alert("No word is loaded!");
    return;
  }

  $btn_ComputeCluster.html('Computing on the Server...');

  //save parameter to json
  params =  {
        'method':$select_method.val(),
        'numCluster':$input_clusterNum.val()
    };

  //send data to server
  $.getJSON('/_applyCluster', params,
    function(result) {
      if(result.result==='Error') {
        $btn_ComputeCluster.html('Computing Error!');
        return;
      }

      top.label=result.label;
      top.distanceMatrix = result.distanceMatrix;
      top.outlierValue = result.outlierValue;
      //recived data from server
      $btn_ComputeCluster.html('Done Computing');

      //trigger graph layout
      $select_kNN.trigger("change");
    });
}

//reset the botton text if parameter changes
function modifyParameters(){
  if($btn_ComputeCluster.html()==='Done Computing')
    $btn_ComputeCluster.html('Compute Clustering');
}
$select_method.bind('change',modifyParameters);
$input_clusterNum.bind('change',modifyParameters);

//compute on the server
$btn_ComputeCluster.bind('click',applySubspaceCluster);
