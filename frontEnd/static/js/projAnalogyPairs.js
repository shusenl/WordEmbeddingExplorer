//////////////////////// global ///////////////////////////

//UI bindings
$('#btn-ComputeProj').bind('click', projSelectedPairs);
//$('#btn-ComputeProj').bind('click', projSelectedPairsNonDuplicate);

//$('#btn-ComputeVecProj').bind('click', projAnalogyVector);

$('#distTransparencyFlag').bind('click', updatePlot);
$('#neighborHighlightFlag').bind('click', updatePlot)
$('#cosineFilterFlag').bind('click', updatePlot)


$('#btn-clearGroupSelect').bind('click', clearGroupSelection);
$('#btn-clearHighlight').bind('click', clearPairHighlight);
$('#btn-highlightAll').bind('click', highlightPairs);
//slider
$('#baseTranparency').bind('change', updatePlot);
$('#neighborSize').bind('change', updatePlot);
$('#cosineCutoff').bind('change', updatePlot);

//subspace clustering
$('#btn-SubCluster').bind('click', computeSubspaceClustering);
var $subClusterMethod = $('#select_subClusterMethod');
var $subClusterNum = $('#input_subClusterNum');

//histogram
$('#histoBin').bind('change', computeVectorDistanceHistogram);
$('#btn-ComputeVecHisto').bind('click', computeVectorDistanceHistogram);

//projection
var $projectionMethodStr = $('#select_projectionMethod');

//global color
d3Color10 = d3.scale.category10();
//d3Color10 = d3.scale.category20();

// var isDynProj = true;
var isDynProj = false;
function toggleDynProj(){
  isDynProj = !isDynProj;
}

function computeNonDuplicateWords(selectedPairs){
  //handle duplicated point in pairs
  var words = extractWordsFromPairs(selectedPairs);
  var wordSet = {};
  var nonDuplicateWords = [];
  top.originToNonDupliacteIndexMap = Array(words.length);
  top.nonDuplicateToOriginIndexMap = {};
  var index = 0;
  for (var i=0; i<words.length; i++){
      if(wordSet[words[i]] === undefined){
        wordSet[words[i]] = index;
        top.originToNonDupliacteIndexMap[i]=index;
        top.nonDuplicateToOriginIndexMap[index]=[i];
        nonDuplicateWords.push(words[i]);
        index++;
      }
      else{
        top.originToNonDupliacteIndexMap[i]=wordSet[words[i]];
        top.nonDuplicateToOriginIndexMap[wordSet[words[i]]].push(i);//append
      }
  }
  top.nonDuplicateWords = nonDuplicateWords;
}

function embeddingNonDuplicateToOriginal(embedding){
    var mappedEmbedding=[];
    for(var i=0; i<top.originToNonDupliacteIndexMap.length; i++){
        mappedEmbedding.push(embedding[top.originToNonDupliacteIndexMap[i]]);
    }
    return mappedEmbedding;
}
///////////////////// subspace clustering //////////////////////
function computeSubspaceClustering() {
  //save parameter to json
  params =  {
        "wordVecs":buildDataMatrixFromWords(top.nonDuplicateWords),
        "method":$subClusterMethod.val(),
        "numCluster":$subClusterNum.val()
    };
  //send data to server
  postToServer(params, '/_applySubspaceCluster', subspaceCallback);

}

function subspaceCallback(result){
   if(result.result==='Error') {
        alert('Subspace Computing Error!');
        return;
      }

   //handle duplicated point in pairs
   var recoveredLabel = []
   for(var i=0; i<top.originToNonDupliacteIndexMap.length; i++){
      recoveredLabel.push(result.label[top.originToNonDupliacteIndexMap[i]]);
   }

   top.subLabel=recoveredLabel;
   top.subDistanceMatrix = result.distanceMatrix;
   top.subSpacesBasis = result.subspaceBasis;

   //create layout graph interface for exploring subspaces
   updatePlot();
   //create subspace navigation
   //drawSubspaceHUD();
}

//////////////////////// dynamic projection //////////////////////

function server_dynamicProj(startProj, endProj, callback){

    postToServer({
                     "startProj":startProj,
                       "endProj":endProj
                 },
                 '/_dynamicProjection',
                 callback);
}

function dynamicProjectionCallback(result){
    //requestAnimationFrame
    var i=0;
    var start = null;
    top.previousBasis = result.projMatList[result.projMatList.length-1];

    requestAnimationFrame(step);

    function step(){
      var basis = result.projMatList[i];
      top.selectedVecsAfterProj = numeric.dot(top.selectedVecsBeforeProj, basis);
      updatePlot();
      i++;

      var id = requestAnimationFrame(step);
      if(i===result.projMatList.length){
          //update PDF plot at last frame
          queryErrorPDF(300, top.selectedPairs.length*2, $projectionMethodStr.val(),
                        generatePDFHUD, top.selectedVecsAfterProj);
          cancelAnimationFrame(id);
      }
    }
}

/////////////////////////////////////////////////////////////////////

//subspace selection callback
function selectSubspace(d){
    var subspaceIndex = d.group-1;
    console.log("index:"+subspaceIndex);
    var basis = top.subSpacesBasis[subspaceIndex];

    //var isDynProj = true;
    //var isDynProj = false;
    if(isDynProj){
        server_dynamicProj(top.previousBasis, basis, dynamicProjectionCallback)
        top.previousBasis = basis;
    }
    else{
        //console.log("row:"+basis.length);
        //console.log("col:"+basis[0].length);
        var projResult = numeric.dot(top.selectedVecsBeforeProj, basis);
        top.selectedVecsAfterProj = projResult;
        //console.log(top.selectedVecsAfterProj);
        //generatePlot();
        updatePlot();
    }
}

function drawSubspaceHUD(){
   //layout graph
   if(top.subDistanceMatrix){
       var graph = computeGraph(top.subDistanceMatrix, 1);

       var svg = d3.select("#svg-pairProjection");

       //drawing HUD
       svg.select("#subspaceHUD").remove()
       svg//.selectAll("rect")
         .append("rect")
         .attr("id", "subspaceHUD")
         .attr("width", 160)
         .attr("height", 160)
         .attr("x", 0)
         .attr("y", 0)
         .attr("rx", 4)
         .attr("ry", 4)
         .style("fill", "grey")
         .style("opacity", 0.2);

       //var force = d3.layout.force()
       var force = cola.d3adaptor()
            .constraints(graph.constraints)
            .linkDistance(function(link) {return link.value*30.0})
            .avoidOverlaps(true) //cola
            .handleDisconnected(true) //cola
            .size([160, 160]);

        force
            .nodes(graph.nodes)
            .links(graph.links)
            .start();

        var link = svg.selectAll(".link")
            .data(graph.links)
          .enter().append("line")
            .attr("class", "link")
            .style("stroke-width", 3.0)//function(d){ return 30.0*Math.sqrt(d.value); });

        var node = svg.selectAll(".node")
            .data(graph.nodes)
          .enter().append("circle")
            .on("click", selectSubspace)
            .attr("class", "node")
            .attr("r", function(d){return 7.0;})
            .style("fill", function(d){ return d3Color10(d.group-1+2); })//+2 to skip the first two color
            .call(force.drag);

        node.append("title")
            .text(function(d) { return d.name; });

        force.on("tick", function() {
          link.attr("x1", function(d) { return d.source.x; })
              .attr("y1", function(d) { return d.source.y; })
              .attr("x2", function(d) { return d.target.x; })
              .attr("y2", function(d) { return d.target.y; });

          node.attr("cx", function(d) { return d.x; })
              .attr("cy", function(d) { return d.y; });
        });
    }
}

////////////////////////// utility /////////////////////////////
function postToServer(dataObj, url, respCallback){
    $(function()
    {
    //send matrix to server get back principle component
    $.ajax({
          type: "POST",
          url: url,
          // The key needs to match your method's input parameter (case-sensitive).
          data: JSON.stringify(dataObj),
          contentType: "application/json; charset=utf-8",
          dataType: "json",
          success: function(resp) {respCallback(resp);},
          failure: function(errMsg) {alert(errMsg);}
      });
    });
}

function iterateAllActivePairs(callback){ //callback(key, object)
    var analogyGroupNames = Object.keys(top.analogyGroupObject);

    for(var i=0; i<analogyGroupNames.length; i++){
        var analogyGroup = analogyGroupObject[analogyGroupNames[i]];
        var analogyPairs = Object.keys(analogyGroup)
          if(analogyGroup.state.selected) //only if the group is selected
            for(var j=1; j<analogyPairs.length; j++){//start from 1 to avoid state
                callback(analogyPairs[j], analogyGroup);
            }
    }
}

function buildDataMatrixFromPairs(selectedPairs){
    return  extractWordsFromPairs(selectedPairs)
        .map(function (word){return top.wordVecs[word];});
}

function buildDataMatrixFromWords(words){
    return words.map(function (word){return top.wordVecs[word];});
}

function buildDataMatrixFromWordsWithSource(words, wordVecs){
    return words.map(function (word){return wordVecs[word];});
}

function buildVectorDataMatrixFromPairs(selectedPairs){
    return selectedPairs.map(function(p){return p.split('-');})
        .map(function(p){return vecDiff(top.wordVecs[p[0]],top.wordVecs[p[1]]);});
}

function extractWordsFromPairs(selectedPairs){
    return selectedPairs.map(function(p){return p.split('-')})
        .reduce(function(a, b) { return a.concat(b);}, []);
}

//////////////////////////////////////////////////////////////////

////////////////// compute distance ///////////////////
function computePairDistanceDistortion(){
    var dist2D = computePairDistance(top.selectedVecsAfterProj);
    var distHD = computePairDistance(top.selectedVecsBeforeProj);

    var distDiff = [];
    for(var i=0; i<dist2D.length; i++)
        distDiff.push(Math.abs(distHD[i]-dist2D[i])/distHD[i]);
    distDiff = normalize1DArrayDiviedByMax(distDiff);
    return distDiff.map(function(x){return [x,x];})
            .reduce(function(a, b) { return a.concat(b);}, []);
}

function distance(wordVec1, wordVec2){
     var diff = [];
     for(var i=0; i<wordVec1.length; i++){
         diff.push(wordVec1[i] - wordVec2[i])
     }
     return numeric.norm2(diff);
}

function vecDiff(wordVec1, wordVec2){
    var diff = []
    for(var i=0; i<wordVec1.length; i++){
        diff.push(wordVec1[i] - wordVec2[i])
    }
    return diff;
}

function computePairDistance(selectedVecs){
    var pairDist = [];
    for(var i=0; i<selectedVecs.length/2; i++){
        var first = selectedVecs[i*2+0];
        var second = selectedVecs[i*2+1];

        var diff = []
        for(var j=0; j<first.length; j++)
            diff.push(first[j]-second[j]);
        pairDist.push( numeric.norm2(diff) );
    }
    return pairDist;
}

function cosineDistance(vec1, vec2){
    var AB = 0.0;
    var A = 0.0;
    var B = 0.0;

    for(var i=0; i<vec1.length; i++){
        AB += vec1[i]*vec2[i];
        A += vec1[i]*vec1[i];
        B += vec2[i]*vec2[i];
    }
    return 1.0-AB/(Math.sqrt(A)*Math.sqrt(B));
}

function normalize1DArray(X){
    var min = Math.min.apply(null, X);
    var max = Math.max.apply(null, X);
    var normX = X.map(function(x){return (x-min)/(max-min);});
    return normX;
}

function normalize1DArrayDiviedByMax(X){
    var max = Math.max.apply(null, X);
    var normX = X.map(function(x){return x/max;});
    return normX;
}
///////////////////////////////////////////////////////////////////


////////////////////////// html generator /////////////////////////
function buildAnalogyGroupTableFromAnalogyGroupObject(analogyGroupObject){
    var analogyTypeCBTemplate =
      '<li>' +
        '<label class="checkbox inline">' +
            '<input type="checkbox" value="false" class ="CBAnalogyGroup" id="{{type}}">{{type}}'+
        '</label>' +
      '</li>'
    ;
    //'<label class="checkbox"><input type="checkbox" value="" id="cb-{{type}}">{{type}}</label>';

    var checkboxHtml = "";
    var analogyGroupNameArray = Object.keys(analogyGroupObject);
    for(var i=0; i<analogyGroupNameArray.length; i++){
        var groupName = analogyGroupNameArray[i];
        checkboxHtml += Mustache.render(analogyTypeCBTemplate, analogyGroupObject[groupName].state);
    }

    //console.log(checkboxHtml);
    $('#lu_analogyGroup').html(checkboxHtml);

    //setupUI
    //connect analogyGroup checkbox with analogy pair display
    $('#analogyGroupCBArray :checkbox').bind('click',updateAnalogyPairsTable);
}
function switchAnalogyOrder(groupName){
  //console.log(groupName);

  //update the analogy
  for(var key in top.analogyGroupObject[groupName]){
    if(top.analogyGroupObject[groupName].hasOwnProperty(key) && key!=='state'){
      //console.log(key);
      var newKey = key.split('-')[1]+'-'+key.split('-')[0];
      top.analogyGroupObject[groupName][newKey] = top.analogyGroupObject[groupName][key]
      delete top.analogyGroupObject[groupName][key];
      //console.log('add key: ', newKey);
    }
  }

  //update table
  updateAnalogyPairsTable();
}
function updateAnalogyPairsTable(){

    //FIXME not clean
    //remove the selcetdPoint index, since the index is out of date
    top.selectPointIndex = undefined;

    //clear current plot since the selected pairs changed
    clearScatterplot();

    //template build UI
    function buildSubPanels(selectedPairsByGroup, selectedGroupNames){
        var subPanelsHtml = '';
        var subPanelTemplate =
               '<div class="panel panel-default"> '+
                    '<div class="panel-heading"> '+
                        '<h3 class="panel-title">"{{groupName}}" <td>&nbsp;</td>'+
                        '<a class="btn btn-default" onClick="switchAnalogyOrder(\'{{groupName}}\')" id="btn-ComputeProj">Switch Order</a>'+
                        '</h3>'+
                    '</div> '+
                    '<div class="panel-body" id="panel-analogyPairs"> '+
                    //'<div class="container"> '+
                    //'<div id="form-analogyPairs-{{groupName}}"> '+
                       '<fieldset class="PairCBfieldSet" id="analogyPairsCBArray-{{groupName}}"> '+
                            '<div class="CBcolumns"> '+
                              '<lu id="{{groupName}}"> '+
                               '{{{checkboxHtml}}} '+
                              '</lu> '+
                            '</div> '+
                       '</fieldset> '+
                    //'</div> '+
                    //'</div> '+
                    '</div> '+
                '</div> '
        var analogyPairCBTemplate =
        '<li>' +
          '<label class="checkbox">' +
            '<input type="checkbox" {{checked}} value="" class ="CBAnalogyPair" id="{{first}}-{{second}}">{{first}} : {{second}}' +
          '</label>';
        '</li>';

        for(var i=0; i<selectedPairsByGroup.length; i++){
            var pairs = selectedPairsByGroup[i];
            //build subpanel
            var subPanelCB = '';
            //for every pair add a checkbox
            for(var j=0; j<pairs.length; j++){
               //console.log(pairs[j]);
               var pair = pairs[j].split('-');
               var isHighlighted = top.analogyGroupObject[selectedGroupNames[i]][pairs[j]];
               var highlightState = isHighlighted ? "checked=checked":""

               subPanelCB += Mustache.render(analogyPairCBTemplate,
                  {first:pair[0],second:pair[1], checked:highlightState}) + "\n";
            }
            //console.log(subPanelCB)
            var panelsTemplateInput =
                {
                    groupName    : selectedGroupNames[i],
                    checkboxHtml : subPanelCB
                };
            subPanelsHtml += Mustache.render(subPanelTemplate, panelsTemplateInput);
        }
        return subPanelsHtml;
    }

    var analogyGroupCB = $('#analogyGroupCBArray :checkbox');
    //window.temp=analogyGroupCB;

    var selectedPairs = [];
    var selectedGroupNames = [];
    var selectedPairsByGroup = [];
    for(var i=0; i<analogyGroupCB.length; i++){
        //two parent because label, li
        var isChecked = analogyGroupCB[i].checked;
        var cbIndex = $(analogyGroupCB[i]).parent().parent().index();
        var groupName = Object.keys(top.analogyGroupObject)[cbIndex];

        analogyGroupObject[groupName].state.selected=isChecked;
        if(isChecked){
            selectedGroupNames.push(analogyGroupObject[groupName].state.type);
            var pairList = Object.keys(analogyGroupObject[groupName]);
            //remove the type key
            pairList.shift();
            selectedPairs = selectedPairs.concat(pairList);
            selectedPairsByGroup.push(pairList);
        }
    }

    var panel = $('#panel-analogyPairs');
    panel.html( buildSubPanels(selectedPairsByGroup, selectedGroupNames) );

    //setupUI
    $('.PairCBfieldSet :checkbox').each(function(){
        var selectedCB = this;
        $(this).bind("click", function(selectedCB){
          updateAnalogyPairSelection(selectedCB);})
        })
    //save selected pairs
    top.selectedPairs = selectedPairs;
    //update the nonDuplicateWords
    computeNonDuplicateWords(selectedPairs);
}

function updateAnalogyPairSelection(selectedCB){
    var groupName = $(selectedCB.toElement).parent().parent().parent().get(0).id;
    var isChecked = Boolean(selectedCB.toElement.checked);
    var pairName = selectedCB.toElement.id;
    //console.log(groupName+":"+pairName);
    var highlightedFlag = top.analogyGroupObject[groupName][pairName];
    top.analogyGroupObject[groupName][pairName]=!highlightedFlag;
}


///////////////////////////////////////////////////////////////////
function projSelectedPairsNonDuplicate(){
    if(top.wordVecs===undefined)
        return;
    if(top.selectedPairs===undefined || top.selectedPairs.length==0)
        return;

    var highlightedPairs = getHighlightedPairs();
    //console.log(highlightedPairs);

    var dataMatrix = buildDataMatrixFromPairs( top.selectedPairs );
    var noDuplicateDataMatrix = buildDataMatrixFromWords(top.nonDuplicateWords);

    //store the before and after pointSet
    top.selectedVecsBeforeProj = dataMatrix;

    /// compute SVM linear separation
    var binaryLabel = [];
    for(var i=0; i<dataMatrix.length; i++){
        if(i%2 === 0)
            binaryLabel.push(0);
        else
            binaryLabel.push(1);
    }

    var projMatrix;
    if($projectionMethodStr.val() === "PCA"){
        if(highlightedPairs.length===0){
            server_generatePCAprojectionMatrix(noDuplicateDataMatrix, function(p){
                    if(isDynProj && top.previousBasis){
                        projMatrix = p.pc;
                        var basis = numeric.transpose(p.pc);
                        server_dynamicProj(top.previousBasis, basis, dynamicProjectionCallback)
                        top.previousBasis = basis;
                    }else{
                        //build projection matrix
                        var pc = numeric.transpose(p.pc);
                        top.previousBasis = pc;
                        var projResult = numeric.dot(noDuplicateDataMatrix, pc);
                        top.selectedVecsAfterProj = embeddingNonDuplicateToOriginal(projResult);
                        generatePlot();
                    }
                }
            );

        }else{
            console.log("focused PCA")
            //console.log("highlightedPairs:"+highlightedPairs)
            //get projection matrix based on PCA of highlight pairs points
            highlightPairsMatrix = buildDataMatrixFromPairs( highlightedPairs );

            server_generatePCAprojectionMatrix(highlightPairsMatrix,
                function(p){
                    //center data
                    //var pca = new PCA();
                    //var centerDataMatrix = pca.scale(dataMatrix, true, true);
                    //build projection matrix
                    projMatrix = p.pc;
                    var pc = numeric.transpose(p.pc);
                    top.previousBasis = pc;
                    var projResult = numeric.dot(dataMatrix, pc);
                    top.selectedVecsAfterProj = projResult;
                    generatePlot();
                });
         }
    }else if($projectionMethodStr.val() === "SVM+PCA"||
             $projectionMethodStr.val() === "SVM+REG"||
             $projectionMethodStr.val() === "SVMplane"
             ){
        server_generateOptimalAnalogyLinearProjection(dataMatrix, binaryLabel, $projectionMethodStr.val(),
             function(p){

                    //////////////////// compute data rotation ////////////////////
                    /*
                    var reflection = p.reflection;
                    var rotatedData = numeric.dot(allDataMatrix, reflection);
                    var axisPair = [];
                    for(var i=0; i<rotatedData.length; i++){
                        axisPair.push([allWords[i], rotatedData[i][0]]);
                    }
                    axisPair.sort(function (a, b) {
                           if (a[1] < b[1]) return -1;
                           if (a[1] > b[1]) return 1;
                           return 0;
                         })
                    for(var i=0; i<axisPair.length; i++){
                        console.log(axisPair[i][0], axisPair[i][1]);
                    }
                    */
                    ///////////////////////////////////////////////////////////////
                    top.SVMdirectionVec = p.projMat[0];

                    if(isDynProj && top.previousBasis){

                        var basis = numeric.transpose(p.projMat);
                        server_dynamicProj(top.previousBasis, basis, dynamicProjectionCallback)
                        top.previousBasis = basis;
                    }else{
                        projMatrix = p.projMat;
                        //console.log(JSON.stringify(projMatrix));
                        var projMat = numeric.transpose(p.projMat);
                        top.previousBasis = projMat;
                        var projResult = numeric.dot(dataMatrix, projMat);
                        top.selectedVecsAfterProj = projResult;
                        generatePlot();
                    }
        });
    }

    ////
    //save the existing projection
    top.projMatrix = projMatrix;

    //compute cluster statistics on the server
    //server_SVMLinearSeparation(dataMatrix, binaryLabel, processSVMresult)

    return false;
}
///////////////////////////////////////////////////////////////////
function projSelectedPairs(){
    if(top.wordVecs===undefined)
        return;
    if(top.selectedPairs===undefined || top.selectedPairs.length==0)
        return;

    var highlightedPairs = getHighlightedPairs();
    //console.log(highlightedPairs);

    var dataMatrix = buildDataMatrixFromPairs( top.selectedPairs );
    /*
    var allWords, allDataMatrix

    if(top.mostFrequentWords){
        allWords = top.mostFrequentWords;
        allDataMatrix = buildDataMatrixFromWords( allWords, top.mostFrequentWordVecs );
    }else{
        allWords = Object.keys(top.wordVecs);
        allDataMatrix = buildDataMatrixFromWords( allWords, top.wordVecs );
    }
    */

    //store the before and after pointSet
    top.selectedVecsBeforeProj = dataMatrix;

    /// compute SVM linear separation
    var binaryLabel = [];
    for(var i=0; i<dataMatrix.length; i++){
        if(i%2 === 0)
            binaryLabel.push(0);
        else
            binaryLabel.push(1);
    }

    var projMatrix;
    if($projectionMethodStr.val() === "PCA"){
        if(highlightedPairs.length===0){
            //console.log("overall PCA")
            //pca projection, server based is used because the js version does not support
            //smaller number points than dimension
            //server_applyPCAprojection(dataMatrix, function(p){
            server_generatePCAprojectionMatrix(dataMatrix, function(p){
                    //var projResult = p.projResult;
                    //projMatrix = p.projMatrix;
                    //console.log(projResult);
                    //top.selectedVecsAfterProj = projResult;
                    //generatePlot();
                    //center data
                    //var centerDataMatrix = pca.scale(dataMatrix, true, true);
                    if(isDynProj && top.previousBasis){
                        projMatrix = p.pc;
                        var basis = numeric.transpose(p.pc);
                        server_dynamicProj(top.previousBasis, basis, dynamicProjectionCallback)
                        top.previousBasis = basis;
                    }else{
                        //build projection matrix
                        var pc = numeric.transpose(p.pc);
                        top.previousBasis = pc;
                        var projResult = numeric.dot(dataMatrix, pc);
                        top.selectedVecsAfterProj = projResult;
                        generatePlot();
                    }
                }
            );
            //projResult = projectionTSNE( dataMatrix );
            //top.selectedVecsAfterProj = projResult;
            //generatePlot();
            //projResult = projectionPCA( dataMatrix );
            //top.selectedVecsAfterProj = projResult;
            //generatePlot();
        }else{
            console.log("focused PCA")
            //console.log("highlightedPairs:"+highlightedPairs)
            //get projection matrix based on PCA of highlight pairs points
            highlightPairsMatrix = buildDataMatrixFromPairs( highlightedPairs );

            server_generatePCAprojectionMatrix(highlightPairsMatrix,
                function(p){
                    //center data
                    //var pca = new PCA();
                    //var centerDataMatrix = pca.scale(dataMatrix, true, true);
                    //build projection matrix
                    projMatrix = p.pc;
                    var pc = numeric.transpose(p.pc);
                    top.previousBasis = pc;
                    var projResult = numeric.dot(dataMatrix, pc);
                    top.selectedVecsAfterProj = projResult;
                    generatePlot();
                });
         }
    }else if($projectionMethodStr.val() === "SVM+PCA"||
             $projectionMethodStr.val() === "SVM+REG"||
             $projectionMethodStr.val() === "SVMplane"||
             $projectionMethodStr.val() === "LDA"
             ){

            var inputMatrix;
            var inputBinaryLabel;
            if(highlightedPairs.length===0){
              inputMatrix = dataMatrix;
              inputBinaryLabel = binaryLabel;
            }else{
              inputMatrix = buildDataMatrixFromPairs( highlightedPairs );
              inputBinaryLabel = []
              for(var i=0; i<inputMatrix.length; i++){
                if(i%2 === 0)
                  inputBinaryLabel.push(0);
                else
                  inputBinaryLabel.push(1);
              }
            }
        server_generateOptimalAnalogyLinearProjection(inputMatrix, inputBinaryLabel, $projectionMethodStr.val(),
             function(p){
                    top.SVMdirectionVec = p.projMat[0];

                    if(isDynProj && top.previousBasis){
                        var basis = numeric.transpose(p.projMat);
                        server_dynamicProj(top.previousBasis, basis, dynamicProjectionCallback)
                        top.previousBasis = basis;
                    }else{
                        projMatrix = p.projMat;
                        //console.log(JSON.stringify(projMatrix));
                        var projMat = numeric.transpose(p.projMat);
                        top.previousBasis = projMat;
                        var projResult = numeric.dot(dataMatrix, projMat);
                        top.selectedVecsAfterProj = projResult;

                        // var embedding = [];
                        // if($projectionMethodStr.val() === "LDA"){
                        //
                        //   for(var i=0; i<p.embedding.length; i++)
                        //     p.embedding[i] = p.embedding[i][0];
                        //   console.log(p.embedding);
                        //   while(p.embedding.length) embedding.push(p.embedding.splice(0,2));
                        //   top.selectedVecsAfterProj = embedding;
                        //   console.log(embedding)
                        // }

                        generatePlot();
                    }
        });
    }

    ////
    //save the existing projection
    top.projMatrix = projMatrix;

    //compute cluster statistics on the server
    //server_SVMLinearSeparation(dataMatrix, binaryLabel, processSVMresult)

    return false;
}

/////////////////////////////////////////////////////////////////////

//generate projection matrix based on selected points
function server_generatePCAprojectionMatrix(dataMatrix, callback){
    postToServer({"matrix":dataMatrix}, '/_computePCAprojMatrix', callback);
}

function server_generateOptimalAnalogyLinearProjection(dataMatrix, label, optimizationType, callback){
    postToServer({"matrix":dataMatrix, "label":label, "type":optimizationType}, '/_computeAnalogyProjMatrix', callback);
}

function server_applyPCAprojection(dataMatrix, callback){
    postToServer({"matrix":dataMatrix}, '/_computePCA', callback);
}

//compute SVM linear separation plane
function server_SVMLinearSeparation(dataMatrix, label, callback){
    postToServer({"matrix":dataMatrix, "label":label}, '/_computeLinearSVM', callback);
}

//process SVM result from server
function processSVMresult(result){
    var distance = result.distance;
    var classification = result.classification;
    var label = result.label;
    var misclassified = 0;
    var class1Size = result.classSize1;
    var class2Size = result.classSize2;
    top.SVMdirectionVec = result.directionVec;
    //console.log(JSON.stringify(top.SVMdirectionVec));

    var class1 = [];
    var class2 = [];

    //console.log(distance);

    for(var i=0; i<label.length; i++)
    {
        if(classification[i]==0)
            class1.push(distance[i]);
        else
            class2.push(distance[i]);

        if(label[i] !== classification[i])
            misclassified++;
    }

    //console.log(misclassified, class1Size, class2Size);

    //console.log(classification)
    //draw two color histogram
    var classOne = {
      x: class1,
      name: "analogyWord1",
      type: "histogram",
      opacity: 0.5,
      marker: {
      color: 'orange',
      },
    };
    var classTwo = {
      x: class2,
      name: "analogyWord2",
      type: "histogram",
      opacity: 0.6,
      marker: {
        color: 'blue',
        },
    };
    var data = [classOne, classTwo];
    var layout = {barmode: "group"};
    Plotly.newPlot("plotlyDiv", data, layout);
}

//project all points using PCA
function projectionPCA(dataMatrix){
    var pca = new PCA();
    //normalize
    var centerDataMatrix = pca.scale(dataMatrix, true, true);
    var projResult = pca.pca(centerDataMatrix, 2);
    projResult = projResult.map(function(p){return [p[0], p[1]]})
    return projResult;
}

//reprojection t-SNE
function reprojectTSNE(tsne, step, displayFunc, finishCallback){

  k=0;

  function drawEmbedding(){
      var displayRatio=2;
      for(var i=0; i<displayRatio; i++)
         tsne.step();
         var currentResult = tsne.getSolution();
         displayFunc(currentResult);
         k++;

         var id = requestAnimationFrame(drawEmbedding);
         if(k>=step/displayRatio){
            finishCallback(tsne.getSolution());
            cancelAnimationFrame(id);
         }
  }
  requestAnimationFrame(drawEmbedding);
}

//projection t-SNE
function projectionTSNE(dataMatrix, step, displayFunc, finishCallback,
       input2DPos, gains, ystep){
    //default parameter
    if(!step) step = 300;

    var opt = {}
    opt.epsilon = 10; // epsilon is learning rate (10 = default)
    opt.perplexity = 30; // roughly how many neighbors each point influences (30 = default)
    opt.dim = 2; // dimensionality of the embedding (2 = default)

    var tsne = new tsnejs.tSNE(opt); // create a tSNE instance
    top.tSNEmethod = tsne;

    if(input2DPos)
        tsne.initDataRawWithPos(dataMatrix, input2DPos, gains, ystep);
    else
        tsne.initDataRaw(dataMatrix);

    //skip the display for first a few step
    for(var k = 0; k < 10; k++) {
      tsne.step();
    }

    k=0;

    function drawEmbedding(){
        var displayRatio=2;
        //do 10 iteration per display
        for(var i=0; i<displayRatio; i++)
            tsne.step();
        var currentResult = tsne.getSolution();
        displayFunc(currentResult);
        k++;

        var id = requestAnimationFrame(drawEmbedding);
        if(k>=step/displayRatio){
            finishCallback(tsne.getSolution());
            cancelAnimationFrame(id);
        }
    }
    requestAnimationFrame(drawEmbedding);
}

function computeNeighborHighlight(selectedVecs, selectedIndex){
    var neighbor = [];
    var dist = [];
    var wordVec = selectedVecs[selectedIndex];

    for(var i=0; i<selectedVecs.length; i++){
        //dist.push( distance(wordVec, selectedVecs[i]) );
        dist.push( cosineDistance(wordVec, selectedVecs[i]) );
    }
    //find neighbor range
    var sortedDist = dist.slice(0).sort()
    var cutoffIndex = Math.floor(($('#neighborSize').get(0).value / 100.0)*dist.length);
    var cutoffValue = sortedDist[cutoffIndex];
    for(var i=0; i<selectedVecs.length; i++){
        if(dist[i] < cutoffValue)
            neighbor.push(true);
        else
            neighbor.push(false);
    }
    return neighbor;
}

/////////////////////////////////////////////////////////////////
////////////////////////// plot related /////////////////////////

function highlightPairs(){
    iterateAllActivePairs( function (key, object){
            object[key]=true;//clear flag
        }
    );
    //redraw
    updateAnalogyPairsTable();
}

function clearPairHighlight(){
    iterateAllActivePairs( function (key, object){
            object[key]=false;//clear flag
        }
    );
    //redraw
    updateAnalogyPairsTable();
}

function getHighlightedPairs(){
    var pairs = [];
    var analogyGroupNames = Object.keys(top.analogyGroupObject);
    for(var i=0; i<analogyGroupNames.length; i++){
        var analogyGroup = analogyGroupObject[analogyGroupNames[i]];
        var analogyPairs = Object.keys(analogyGroup)
        for(var j=1; j<analogyPairs.length; j++){//start from 1 to avoid state
            if(analogyGroup[analogyPairs[j]]===true)//if true it is highlighted
                pairs.push(analogyPairs[j]);
        }
    }
    //console.log(pairs);
    return pairs;
}

function clearGroupSelection(){
    //clear dynProj
    top.previousBasis = undefined;

    var analogyGroupCBSelected = $('#analogyGroupCBArray :checkbox:checked');
    for(var i=0; i<analogyGroupCBSelected.length; i++){
        analogyGroupCBSelected[i].checked = false;
    }
    //redraw
    buildAnalogyGroupTableFromAnalogyGroupObject(top.analogyGroupObject);
    //clear pairs
    clearPairHighlight();
    //clear proj
    clearScatterplot();
}

function clearScatterplot(){
    //delete svg node for pair projection
    $("#pairProjection").empty();
    //delete svg node for vector proj
    $("#vectorProjection").empty();
    //clear histogram
    $("#vectorHisto").empty();
    //clean up plotly
    $("#plotlyDiv").empty();

    //reset dynamicProjection
    top.previousBasis = undefined;
}

function generatePlot(){
    if(top.selectedVecsAfterProj===undefined ||
       top.selectedVecsBeforeProj===undefined )
       return;
    //update highlighted pairs
    top.highlightedPairSet = new Set(getHighlightedPairs());
    //compute distortion
    top.selectedVecsDistortion = computePairDistanceDistortion();
    //display
    updatePointInfo(top.selectedVecsAfterProj,
                    extractWordsFromPairs(top.selectedPairs),
                    //$('#distTransparencyFlag').get(0).checked,
                    false,
                    //$('#cosineFilterFlag').get(0).checked,
                    $('#neighborHighlightFlag').get(0).checked,
                    top.selectPointIndex );
    D3generatePairProjection();
}

function updatePlot(dummy, index){
    if(top.selectedVecsAfterProj===undefined ||
       top.selectedVecsBeforeProj===undefined ||
       top.selectedVecsDistortion===undefined
       )
       return;

    if(index!==undefined)
        top.selectPointIndex = index;

    //compute distortion
    top.selectedVecsDistortion = computePairDistanceDistortion();

    //display
    updatePointInfo(top.selectedVecsAfterProj,
                    extractWordsFromPairs(top.selectedPairs),
                    false,
                    //$('#distTransparencyFlag').get(0).checked,
                    //$('#cosineFilterFlag').get(0).checked,
                    $('#neighborHighlightFlag').get(0).checked,
                    top.selectPointIndex );

    D3updatePairProjection();
}


//draw 2D scatterplot based on an array of 2D point [[x1,y1], [x2,y2]], and the corresponding words
function updatePointInfo(projResult, words, opacityFlag, neighborHighlightFlag,
    selectedIndex /* only used for neigbhor highting*/){
    //don't draw if no point presented
    if(projResult===undefined)
        return;

    //deep copy array of array
    points2D = $.extend(true, [], projResult);

    //check if the selectedIndex exists
    if(selectedIndex === undefined)
        neighborHighlightFlag = false;
    //distance to all other points
    neighbor = [];
    if(neighborHighlightFlag)
        neighbor = computeNeighborHighlight(top.selectedVecsBeforeProj, selectedIndex);

    var distToSVM = [];

    var cutOffValue = 0.0;
    if(opacityFlag && top.SVMdirectionVec){
        var vectorDataMatrix = buildVectorDataMatrixFromPairs( top.selectedPairs );
        var cutOffRatio = Number($('#cosineCutoff').get(0).value) / 100.0;
        for(var i=0; i<vectorDataMatrix.length; i++){
           distToSVM.push(cosineDistance(top.SVMdirectionVec, vectorDataMatrix[i] ));
           distToSVM.push(cosineDistance(top.SVMdirectionVec, vectorDataMatrix[i] ));
        }
        var range = d3.extent(distToSVM);
        cutOffValue = cutOffRatio*(range[1]-range[0])+range[0];
    }
    var pointPairs = [];
    for(var i=0; i<words.length; i++){
        var opacity = 1.0;
        if(opacityFlag){
            /*
            opacity = 1.0-top.selectedVecsDistortion[i];
            opacity = opacity*opacity;
            var baseOpacity = $('#baseTranparency').get(0).value/100.0;
            opacity = baseOpacity+opacity*(1.0-baseOpacity);
            //opacity = opacity*opacity;
            */
            if(distToSVM[i]>cutOffValue){
              opacity = 0.0;
            }
        }

        ///////////////////// for points2D /////////////////////
        points2D[i].push(words[i]);
        //color indicator
        if(neighborHighlightFlag && neighbor[i])
            points2D[i].push(2); // 2 indicate highlight neighbor
        else
            points2D[i].push(i%2); // 0 or 1
        //point opacity
        points2D[i].push(opacity);

        //update clustering label
        if(top.subLabel){
            if(top.subLabel.length === words.length)
                points2D[i].push(top.subLabel[i]);
        }

        ///////////////////// for pointPairs /////////////////////
        if(i%2===1){
            //determine if need highlighting
            var isHighlighted = top.highlightedPairSet.has(words[i-1]+'-'+words[i]);
            pointPairs.push([points2D[i-1], points2D[i], opacity, isHighlighted]);
        }
    }

    /*
    var points2D = [
                  [ 5,     20,  "Athens", 0, 0.5 ],
                  [ 480,   90, "Swiss", 1, 0.5  ],
                  [ 250,   50, "Greece", 1, 0.5  ],
                  [ 100,   33, "Swiss", 0, 0.5  ],
                  [ 330,   95, "China", 1, 0.5  ],
                  [ 410,   12, "Berlin", 0, 0.5  ],
                  [ 475,   44, "Swiss", 1, 0.5  ],
                  [ 25,    67, "Swiss", 0, 0.5  ],
                  [ 85,    21, "Swiss", 1, 0.5  ],
                  [ 220,   88, "Swiss", 0, 0.5  ]
              ];
    */

    //update figure
    top.points2D = points2D;
    top.pointPairs = pointPairs;
}

/////////////////////////// all d3 related /////////////////////////////
function D3updatePairProjection(){
    var points2D = top.points2D;
    var pointPairs = top.pointPairs;

    var rect = d3.select("#pairProjection").node().getBoundingClientRect()
    //var margin = {top: rect.width/aspectRatio*0.05, bottom: rect.width/aspectRatio*0.05, right: rect.width*0.1, left: rect.width*0.05};
    var margin = {top: 20, bottom: 10, right: 80, left: 15};
    var width = rect.width - margin.left - margin.right;
    //var height = rect.width/aspectRatio - margin.top - margin.bottom;
    var height = width;

    var x = d3.scale.linear()
        .range([margin.left, width]);

    var y = d3.scale.linear()
        .range([margin.top+110, height]);

    x.domain(d3.extent(points2D, function(d) { return d[0]; }));
    y.domain(d3.extent(points2D, function(d) { return d[1]; }));

    //update arrow connecting the pairs
    var svg = d3.select("#svg-pairProjection");

    svg.selectAll("line")
       .data(pointPairs)
       .attr("x1", function(d){return x(d[0][0]);})
       .attr("y1", function(d){return y(d[0][1]);})
       .attr("x2", function(d){return x(d[1][0]);})
       .attr("y2", function(d){return y(d[1][1]);})
       .attr("stroke-width", 2)
       .attr("stroke", function(d) { if(d[3]) return "red"; else return "grey"; })
       .style("opacity", function(d) { return d[2]; });

    //draw points
    svg.selectAll("circle")
       .data(points2D)
       .attr("cx", function(d) { return x(d[0]);})
       .attr("cy", function(d) { return y(d[1]);})
       .attr("r",function(d) { if(d[3]===2)return 8; else return 5;})
       //.style("fill", function(d) { return d3Color10(d[3]); })
       .style("fill", function(d)
       {
         var color = d3Color10(d[3]);
         if(d[5] !== undefined)
           color = d3Color10(d[5]+2);
         return color;
       })//+2 to spkit the first 2 color
       .style("opacity", function(d) { return d[4]; })
       .on("click", updatePlot);

    //add per point label
    svg.selectAll("text")
      .data(points2D)
      .attr("x", function(d) { return x(d[0]);})
      .attr("y", function(d) { return y(d[1])-5;})
      .attr("font-family", "sans-serif")
      .attr("font-size", "15px")
      .attr("fill",  function(d) { return d3Color10(d[3]); })
      .style("opacity", function(d) { return d[4]; });

    //console.log(JSON.stringify(top.selectedVecsAfterProj))
    //////// PDF ////////
    //sampleCount, dataDim, wordCount, callback

}

function D3generatePairProjection(){
    var points2D = top.points2D;
    var pointPairs = top.pointPairs;
    var neighborPoint2D = top.neighborPoint2D;

    //compute data x, y aspect ratio
    var X = points2D.map(function(p){return p[0]});
    var Y = points2D.map(function(p){return p[1]});

    var rangeX = Math.max.apply(null, X)-Math.min.apply(null, X);
    var rangeY = Math.max.apply(null, Y)-Math.min.apply(null, Y);
    var aspectRatio = rangeX/rangeY;

    var rect = d3.select("#pairProjection").node().getBoundingClientRect()
    //var margin = {top: rect.width/aspectRatio*0.05, bottom: rect.width/aspectRatio*0.05, right: rect.width*0.1, left: rect.width*0.05};
    var margin = {top: 20, bottom: 10, right: 80, left: 5};
    var width = rect.width - margin.left - margin.right;
    //var height = rect.width/aspectRatio - margin.top - margin.bottom;
    var height = width;

    if(d3.select("#svg-pairProjection").empty()){
        d3.select("#pairProjection").append("svg").attr("id", "svg-pairProjection")
    }
    else{
        d3.select("#svg-pairProjection").selectAll("*").remove();
    }

    //var svg = d3.select("svg");
    var svg = d3.select("#svg-pairProjection");
        svg
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    //////////////////// add hover event ////////////////////
    d3.select("#svg-pairProjection")
        .on("mousemove", function() {
            var mousePosition = d3.mouse(this);
            highlightLineOnClick(mousePosition, true); //true will also add tooltip
        })
        .on("mouseout", function(){
            cleanTooltip();
        });
    //////////////////////////////////////////////////////////

    var x = d3.scale.linear()
        .range([margin.left, width]);

    var y = d3.scale.linear()
        .range([margin.top+110, height]);

    //determine whether merge the points2D and neighborPoint2D is necessary
    if(neighborPoint2D){
        x.domain(d3.extent(points2D.concat(neighborPoint2D), function(d) { return d[0]; }));
        y.domain(d3.extent(points2D.concat(neighborPoint2D), function(d) { return d[1]; }));
    }
    else{
        x.domain(d3.extent(points2D, function(d) { return d[0]; }));
        y.domain(d3.extent(points2D, function(d) { return d[1]; }));
    }

    //draw arrow connecting the pairs
    svg.selectAll("line")
       .data(pointPairs)
       .enter()
       .append("line")
       .attr("x1", function(d){return x(d[0][0]);})
       .attr("y1", function(d){return y(d[0][1]);})
       .attr("x2", function(d){return x(d[1][0]);})
       .attr("y2", function(d){return y(d[1][1]);})
       .attr("stroke-width", 2)
       .attr("stroke", function(d) { if(d[3]) return "red"; else return "grey"; })
       .style("opacity", function(d) { return d[2]; });

    //draw points
    svg.selectAll("circle")
       .data(points2D)
       .enter()
       .append("circle")
       .attr("cx", function(d) { return x(d[0]);})
       .attr("cy", function(d) { return y(d[1]);})
       .attr("r",function(d) { if(d[3]===2)return 8; else return 5;})
       .style("fill", function(d)
       {
         var color = d3Color10(d[3]);
         if(d[5] !== undefined)
           color = d3Color10(d[5]+2);
         return color;
       })
       //.style("fill", function(d) { return d3Color10(d[5]+2); })//+2 to spkit the first 2 color
       .style("opacity", function(d) { return d[4]; })
       .on("click", updatePlot);

    //add per point label
    svg.selectAll("text")
     .data(points2D)
      .enter()
      .append("text")
      .text(function(d) { return d[2];})
      .attr("x", function(d) { return x(d[0]);})
      .attr("y", function(d) { return y(d[1])-5;})
      .attr("font-family", "sans-serif")
      .attr("font-size", "15px")
      .attr("fill",  function(d) { return d3Color10(d[3]); })
      //.attr("fill",  function(d) { return "#ffbb78"; })
      .style("opacity", function(d) { return d[4]; });

    /////////////////////// add neighbor points ////////////////////
    //draw points
    if(neighborPoint2D){
    svg.selectAll("circle")
       .data(neighborPoint2D)
       .enter()
       .append("circle")
       .attr("cx", function(d) { return x(d[0]);})
       .attr("cy", function(d) { return y(d[1]);})
       .attr("r",function(d) { return 5;})
       .style("fill", function(d)
       {
         var color = d3Color10(5);
         return color;
       });

    svg.selectAll("text")
     .data(neighborPoint2D)
      .enter()
      .append("text")
      .text(function(d) { return d[2];})
      .attr("x", function(d) { return x(d[0]);})
      .attr("y", function(d) { return y(d[1])-5;})
      .attr("font-family", "sans-serif")
      .attr("font-size", "15px")
      .attr("fill",  function(d) { return d3Color10(5); });
    }

    //////// PDF ////////
    //dataDim, wordCount, projMethod, callback
    queryErrorPDF(300, top.selectedPairs.length*2, $projectionMethodStr.val(),
                  generatePDFHUD, /*optional*/top.selectedVecsAfterProj);

}

/////////////////// project neighbor points ///////////////////



///////////////////////////////////////////////////////////////////
////////////////////////Generate Histogram/////////////////////////
///////////////////////////////////////////////////////////////////

function computeVectorDistanceHistogram(){
    var vectorDataMatrix = buildVectorDataMatrixFromPairs( top.selectedPairs );

    top.highlighAnalogyVector = []

    var distValues = [];
    top.distValueLookup = [];
    for(var i=0; i<vectorDataMatrix.length; i++){
        top.highlighAnalogyVector.push(0.0);//init
        for(var j=0; j<i; j++){
            var val = cosineDistance(vectorDataMatrix[i], vectorDataMatrix[j]);
            distValues.push(val);
            top.distValueLookup.push({"value":val, "index":[i, j]});
        }
    }
    var binNum = Number($('#histoBin').val());

    // A formatter for counts.
    var formatCount = d3.format(",.0f");

    var aspectRatio = 2.5;
    var rect = d3.select("#vectorHisto").node().getBoundingClientRect();
    var margin = {top: rect.width/aspectRatio*0.05, bottom: rect.width/aspectRatio*0.05, right: rect.width*0.05, left: rect.width*0.05};
    var width = rect.width - margin.left - margin.right;
    var height = rect.width/aspectRatio - margin.top - margin.bottom;

    var x = d3.scale.linear()
        .domain([0, 1])
        .range([0, width]);

    // Generate a histogram using twenty uniformly-spaced bins.
    var data = d3.layout.histogram()
        .bins(x.ticks(binNum))
        (distValues);

    var y = d3.scale.linear()
        .domain([0, d3.max(data, function(d) { return d.y; })])
        .range([height, 0]);

    var xAxis = d3.svg.axis()
        .scale(x)
        .orient("bottom");

    if(d3.select("#svg-vectorHisto").empty()){
        d3.select("#vectorHisto").append("svg").attr("id", "svg-vectorHisto")
    }
    else{
        d3.select("#svg-vectorHisto").selectAll("*").remove();
    }


    var svg = d3.select("#svg-vectorHisto");
    svg.attr("width", width + margin.left + margin.right)
       .attr("height", height + margin.top + margin.bottom)
       .append("g")
       .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    var bar = svg.selectAll(".bar")
        .data(data)
      .enter().append("g")
        .attr("class", "bar")
        .attr("transform", function(d) { return "translate(" + x(d.x) + "," + y(d.y) + ")"; });

    bar.append("rect")
        .attr("x", 0)
        .attr("width", x(data[0].dx) -3)
        .attr("height", function(d) { return height - y(d.y); })
        .on("click",
        function(d)
        {
           //clear previous selection
           for(var i=0; i<top.highlighAnalogyVector.length; i++)
              top.highlighAnalogyVector[i] = 0.0;

           var dmin = d.x;
           var dmax = d.x+d.dx;
           for (var i=0; i<distValueLookup.length; i++){
               if(distValueLookup[i].value>dmin && distValueLookup[i].value<dmax){
                  //update the highlight
                  top.highlighAnalogyVector[distValueLookup[i].index[0]] = 1.0;
                  top.highlighAnalogyVector[distValueLookup[i].index[1]] = 1.0;
               }
           }
           //console.log(top.highlighAnalogyVector);
           reDrawVectorPairs();
           //select bins
        });
    bar.append("text")
        .attr("dy", ".50em")
        .attr("y", 6)
        .attr("x", x(data[0].dx) / 2)
        .attr("text-anchor", "middle")
        .text(function(d) { return formatCount(d.y); });

    svg.append("g")
        .attr("class", "x axis")
        .attr("transform", "translate(0," + height + ")")
        .call(xAxis);

}


//////////////////// mouse overline highlight ////////////////////////
// http://bl.ocks.org/mostaphaRoudsari/b4e090bb50146d88aec4

function cleanTooltip(){
	// removes any object under #tooltip is
	d3.select("#svg-pairProjection").selectAll("#tooltip")
    	.remove();
}

function addTooltip(clicked, clickedCenPts){

	// sdd tooltip to multiple clicked lines
    var clickedDataSet = [];

    // get all the values into a single list
    var text = clicked[0];
  	var x = clickedCenPts[0][0];
  	var y = clickedCenPts[0][1];
  	clickedDataSet.push([x, y, text]);

	// add rectangles
	var fontSize = 14;
	var padding = 2;
	var rectHeight = fontSize + 2 * padding; //based on font size
    var svg = d3.select("#svg-pairProjection");

	svg.selectAll("rect[id='tooltip']")
        	.data(clickedDataSet).enter()
        	.append("rect")
        	.attr("x", function(d) { return d[0] - d[2].length * 5;})
			.attr("y", function(d) { return d[1] - rectHeight + 2 * padding; })
			.attr("rx", "2")
			.attr("ry", "2")
			.attr("id", "tooltip")
			.attr("fill", "grey")
			.attr("opacity", 0.9)
			.attr("width", function(d){return d[2].length * 10;})
			.attr("height", rectHeight);

	// add text on top of rectangle
	svg.selectAll("text[id='tooltip']")
    	.data(clickedDataSet).enter()
    		.append("text")
			.attr("x", function(d) { return d[0];})
			.attr("y", function(d) { return d[1]; })
			.attr("id", "tooltip")
			.attr("fill", "white")
			.attr("text-anchor", "middle")
			.attr("font-size", fontSize)
        	.text( function (d){ return d[2];})
}

function highlightLineOnClick(mouseClick, drawTooltip){
	var clicked = [];
    var clickedCenPts = [];
	clickedData = getClickedLines(mouseClick);

	if (clickedData && clickedData[0].length!=0){

		clicked = clickedData[0];
    	clickedCenPts = clickedData[1];

		if (drawTooltip){
			// clean if anything is there
			cleanTooltip();
	    	// add tooltip
	    	addTooltip(clicked, clickedCenPts);
		}

	}
};

function getClickedLines(mouseClick){
  var clicked = [];
  var clickedCenPts = [];

  var svg = d3.select("#svg-pairProjection");
  var circles = svg.selectAll("circle")[0];
  //get point position
  var points2D = [];
  for(var i=0; i<circles.length; i++){
      var point = [];
      point.push(circles[i].getAttribute("cx"));
      point.push(circles[i].getAttribute("cy"));
      point.push(top.points2D[i][2]);//add word
      points2D.push(point);
  }

  // find the line
  for(var i=0; i<points2D.length/2; i++){
    if(isOnLine(points2D[i*2+0], points2D[i*2+1], mouseClick, 5)){
      clicked.push(points2D[i*2+0][2]+':'+points2D[i*2+1][2]);
      clickedCenPts.push([(Number(points2D[i*2+0][0])+Number(points2D[i*2+1][0]))/2.0,
                          (Number(points2D[i*2+0][1])+Number(points2D[i*2+1][1]))/2.0]); // for tooltip
    }
  }

  return [clicked, clickedCenPts]
}

function dist2(x1, y1, x2, y2) {return Math.sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2));}

function isOnLine(startPt, endPt, testPt, tol){
	// check if test point is close enough to a line
	// between startPt and var mousePosition = d3.mouse(this);			 . close enough means smaller than tolerance
	var x0 = testPt[0];
	var	y0 = testPt[1];
	var x1 = startPt[0];
	var	y1 = startPt[1];
	var x2 = endPt[0];
	var	y2 = endPt[1];
	var Dx = x2 - x1;
	var Dy = y2 - y1;
	var delta = Math.abs(Dy*x0 - Dx*y0 - x1*y2+x2*y1)/Math.sqrt(Math.pow(Dx, 2) + Math.pow(Dy, 2));
	//console.log(delta);

	if (delta <= tol){
	   var lineDist = dist2(x1,y1,x2,y2);
       var p1Dist = dist2(x0,y0,x1,y1);
	   var p2Dist = dist2(x0,y0,x2,y2);
	   if(p1Dist<lineDist && p2Dist<lineDist)
	       return true;
	}
	return false;
}
