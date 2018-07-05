var $tSNEstep = $('#input_tSNEstep');
var $HDNeighborShown = $('#HDNeighborShown');
var $rankingDistortionkNN = $('#input_tSNEdistortionNeighborSize');
//color
tSNEd3Color20 = d3.scale.category20();

//buttom
d3.select('#btn-ComputetSNE-Server').on('click', applytSNEServer);
d3.select('#btn-ComputetSNE').on('click',applytSNE);
d3.select('#btn-ComputetSNE-Reoptimize').on('click', reoptimizetSNE);
d3.select('#btn-ComputetSNE-FocusedOptimize').on('click', focusedOptimizetSNE);
d3.select('#btn-tSNEcluster').on('click', tSNEcomputeClustering);
d3.select('#btn-tSNECompareWordVec').on('click', tSNECompareWordVec);
//per-point action buttom
d3.select('#btn-PerPoint-SplitWords').on('click', perPointSplitWords);

//textbox
d3.select('#input_tSNEwordSearch').on('input',searchWord);

//checkbox
d3.select('#tSNEColorBarFlag').on('change', d3tSNEplot);
d3.select('#tSNEToolTipFlag').on('change', d3tSNEplot);
d3.select('#tSNEHDNeighborFlag').on('change', d3tSNEplot);

//dropbox
d3.select('#tSNEcolorMapScheme').on('change', computeColorMap);
d3.select('#input_tSNEdistortionNeighborSize').on('change', computeColorMap);

//slider
d3.select('#HDNeighborShown').on('change', function()
	{
        if(top.istSNEtoolTipActive){
           cleantSNEtooltip(); 
           top.istSNEtoolTipActive = false;
        }
        var index = top.tSNEselectedPointIndex;
        var svgPoint = d3.selectAll(".dot")[0][index];
		addtSNEtooltip(Number(svgPoint.getAttribute('cx')), Number(svgPoint.getAttribute('cy')),
                           top.tSNEselectedPointText, top.tSNEselectedPointIndex);
	});

////// on server //////
function computetSNEonServer(dataMatrix, callback){
    postToServer({"matrix":dataMatrix}, '/_computetSNE', callback);
}

function computeKernelDensityonServer(dataMatrix, callback){
	postToServer({"matrix":dataMatrix, "kernelType":'gaussian', "kernelSize":1.5}, '/_computeKDE', callback);
}

function computePerPointMeasureOnServer(dataMatrix, neighborSize, measureType, callback){
	postToServer({"matrix":dataMatrix, "neighborSize":neighborSize, "measureType":measureType}, 
	 '/_computePerPointMeasure', callback);	 
}

///////////////////// WordVec comparison //////////////////////
function tSNECompareWordVec(){
	//load the words
	queryServerForVectors(top.tSNEwords, $('#select_tSNECompareWordVec').val(),
    function(result){
        //top.tSNEwordFrequencyRank = result['frequencyRank'];
        top.tSNEcompareWordVecs = result['wordVec'];
        $('#btn-tSNECompareWordVec').html('Compare Word Vectors Loaded');      
        $('#tSNEComparelist').text('Word Count:'+top.tSNEwords.length);

        top.tSNEdataMatrixCompare = buildDataMatrixFromWords(top.tSNEwords, top.tSNEcompareWordVecs);

        //switch to comparison colormap
        document.getElementById('tSNEcolorMapScheme').value = 'EmbedComparison';

        //compute distortion difference
        computeColorMap();
      }
    );
}

///////////////////// t-SNE computation ////////////////////////
function updatetSNEpartialplot(subsetEmbeddingResult){

	var x = d3.scale.linear()
		.range([top.tSNEmargin.left, top.tSNEwidth]);

	var y = d3.scale.linear()
		.range([top.tSNEmargin.top, top.tSNEwidth]);//width==height

	x.domain(d3.extent(top.tSNEembeddingResult, function(d) { return d[0]; }));
	y.domain(d3.extent(top.tSNEembeddingResult, function(d) { return d[1]; }));


	var circles = d3.select("#svg-tSNEembedding").selectAll(".dot")[0];
    var index = 0;
	for(var i=0; i<top.tSNElassoSelection.length; i++){
		if(top.tSNElassoSelection[i]){
			//update point position
		  	top.tSNEpoints2D[i][0] = subsetEmbeddingResult[index][0];
		  	top.tSNEpoints2D[i][1] = subsetEmbeddingResult[index][1];
			//update display point position
			circles[i].setAttribute("cx", x(subsetEmbeddingResult[index][0]));
			circles[i].setAttribute("cy", y(subsetEmbeddingResult[index][1]));
			index++;
		}
	}
	//TODO: update colormap
}
function updatetSNEplot(embeddingResult){
	if(top.tSNEpoints2D === undefined){
		//init 2D points
		generateWordEmbeddingPlot(embeddingResult);
    }
	else{ //just update the point location
	    //update tooltip
        top.tSNEembeddingResult = embeddingResult;
		//update 2d point
	  	for(var i=0; i<top.tSNEpoints2D.length; i++){
		  	top.tSNEpoints2D[i][0] = embeddingResult[i][0];
		  	top.tSNEpoints2D[i][1] = embeddingResult[i][1];
	  	}

	    //update distortion measure
	    computeColorMap();
	    /*

		var x = d3.scale.linear()
			.range([top.tSNEmargin.left, top.tSNEwidth]);

		var y = d3.scale.linear()
			.range([top.tSNEmargin.top, top.tSNEwidth]);//width==height

		x.domain(d3.extent(embeddingResult, function(d) { return d[0]; }));
		y.domain(d3.extent(embeddingResult, function(d) { return d[1]; }));
		
		//update 2d point
	  	for(var i=0; i<top.tSNEpoints2D.length; i++){
		  	top.tSNEpoints2D[i][0] = embeddingResult[i][0];
		  	top.tSNEpoints2D[i][1] = embeddingResult[i][1];
		  	if(top.tSNEperPointDistortion)
		  	   top.tSNEpoints2D[i][4] = top.tSNEperPointDistortion[i];
	  	}

		//D3 update
	 	var min = Math.min.apply(null, top.tSNEperPointDistortion);
        var max = Math.max.apply(null, top.tSNEperPointDistortion);

		top.tSNEdistortionColorMap = d3.scale.linear().domain([min, max]).range(colorMap);
		d3.select("#svg-tSNEembedding").selectAll(".dot")
			   .data(top.tSNEpoints2D)
			   .attr("cx", function(d) { return x(d[0]);})
			   .attr("cy", function(d) { return y(d[1]);})
			   .style("fill", function(d) 
                { 
         			if(d.length>4)
           				return top.tSNEdistortionColorMap(d[4]);
         			else
           				return d3Color10(0);         
       			})
	    //console.log(d3.select("#svg-tSNEembedding").selectAll("circle")[0][0]);
        */
	    
	}
}

function applytSNEServer(){
  if(top.tSNEwordVecs)
  {
     var dataMatrix = buildDataMatrixFromWords(top.tSNEwords, top.tSNEwordVecs);
     top.tSNEdataMatrix = dataMatrix;

     cleantSNEtooltip();

     computetSNEonServer(dataMatrix, function(p){
	   top.tSNEembeddingResult = p.projResult;
	   generateWordEmbeddingPlot(top.tSNEembeddingResult);

	   //compute the distortion based on current selection
	   computeColorMap();
	   var neighborSize = Number($rankingDistortionkNN.val());
	   //compute rank distortion
	   top.LDNeighbor = computePointNeighbor(top.tSNEembeddingResult, neighborSize,  cosineDistance);
	   top.HDNeighbor = computePointNeighbor(top.tSNEdataMatrix, neighborSize,  cosineDistance);	   
     });
  }
}

function applytSNE(){
  if(top.tSNEwordVecs)
  {
    //console.log(top.tSNEwords);    
    var dataMatrix = buildDataMatrixFromWords(top.tSNEwords, top.tSNEwordVecs);
    top.tSNEdataMatrix = dataMatrix;
    var step = Number($tSNEstep.val());

    cleantSNEtooltip();
    
    //var projResult = projectionTSNE(dataMatrix, step, updatetSNEplot);
    projectionTSNE(dataMatrix, step, updatetSNEplot, function(projResult){
       //compute the distortion based on current selection
	   top.tSNEembeddingResult = projResult;		   
	   //generateWordEmbeddingPlot(top.tSNEembeddingResult);
       computeColorMap();
       
	   var neighborSize = Number($rankingDistortionkNN.val());
	   //compute rank distortion
	   top.LDNeighbor = computePointNeighbor(top.tSNEembeddingResult, neighborSize,  cosineDistance);
	   top.HDNeighbor = computePointNeighbor(top.tSNEdataMatrix, neighborSize,  cosineDistance);
    });
   
  }
}

///////////////////////// tSNE of the subset of the words ////////////////////
function focusedOptimizetSNE(){
	if(top.tSNElassoSelection){
	   if(top.tSNElassoSelection.indexOf(true) !== -1){
	   	 var subsetWords = [];
	   	 var subsetEmbeddingResult = [];
	   	 for(var i=0; i<top.tSNElassoSelection.length; i++){
	   	 	if(top.tSNElassoSelection[i]){ //if the word is selected
	   	 		subsetWords.push(top.tSNEwords[i]);
	   	 		subsetEmbeddingResult.push(top.tSNEembeddingResult[i]);
	   	 	}
	   	 }
	   	 var subDataMatrix = buildDataMatrixFromWords(subsetWords, top.tSNEwordVecs);
	   	 cleantSNEtooltip();
         /*
	   	 if(top.tSNEmethod){
	   	 	var gains = top.tSNEmethod.getGains(top.tSNElassoSelection);
	   	 	var ystep = top.tSNEmethod.getyStep(top.tSNElassoSelection);

		 	projectionTSNE(subDataMatrix, 200, updatetSNEpartialplot, function(projResult){
		   	//compute the distortion based on current selection
		   	//top.tSNEembeddingResult = projResult;		   
		   	//generateWordEmbeddingPlot(top.tSNEembeddingResult);
		   	//computeColorMap();       	
		 	}, subsetEmbeddingResult, gains, ystep);
		 }else{*/
		 	projectionTSNE(subDataMatrix, 200, updatetSNEpartialplot, function(projResult){
		   	//compute the distortion based on current selection
		   	//top.tSNEembeddingResult = projResult;		   
		   	//generateWordEmbeddingPlot(top.tSNEembeddingResult);
		   	//computeColorMap();       	
		 	}, subsetEmbeddingResult);

		 //}
	   }
	}else{
		//project with intial position
		 projectionTSNE(top.tSNEdataMatrix, 200, updatetSNEplot, function(projResult){
		   //compute the distortion based on current selection
		   top.tSNEembeddingResult = projResult;		   
		   //generateWordEmbeddingPlot(top.tSNEembeddingResult);
		   computeColorMap();       	
		 });			
	}
}
/////////////////////////////////////////////////////////////////////////////
function reoptimizetSNE(){
	if(top.tSNEmethod)
	{
		//var weight = squareMatrix(top.tSNEwords.length * top.tSNEwords.length, 1.0);

		if(top.tSNEselectedPointIndex){
			var index = top.tSNEselectedPointIndex;
			//compute distance to point
			var smallest = Number.MAX_VALUE;
			top.tSNEdistanceToCurrentPoint = [];
			var dist = 0;
			for(var i=0; i<top.tSNEdataMatrix.length; i++){
				if(index !== i)
				{
					dist = cosineDistance(top.tSNEdataMatrix[index], top.tSNEdataMatrix[i]);
					if(dist < smallest)
					   smallest = dist;
				}
				top.tSNEdistanceToCurrentPoint.push(dist);
			}
         	//update for the current point
	        top.tSNEdistanceToCurrentPoint[index] = smallest;
		  
		    var weight = top.tSNEdistanceToCurrentPoint;
            var min = Math.min.apply(null, weight);
            var max = Math.max.apply(null, weight);
            //var indices =  findIndicesOfMin(weight, 50);
            //var weight = Array.apply(null, Array(weight.length)).map(function() { return 0.03 });
            //for(var i=0; i<indices.length; i++){
            //	weight[indices[i]] = 1.0;
            //}
            //weight = weight.map(function(p){return p*100;});
            //*
            weight = weight.map(function(p)
              {
              	var val = 1.0-p/max;              	
              	return val*val;
              });
            //*/
            //var min = Math.min.apply(null, weight);
            //var max = Math.max.apply(null, weight);
            //console.log(min, max);
		    top.tSNEperPointDistortion = weight;
		    updatetSNEpoints2D();

            //top.tSNEmethod.initSolution();
		    top.tSNEmethod.updateWeight(weight);
		    top.tSNEmethod.setFocus(index);
		}

		//*		
		reprojectTSNE(top.tSNEmethod, 300, updatetSNEplot, function(projResult){
           top.tSNEembeddingResult = projResult;		   
	       computeColorMap();       	
        });
        //*/
	}
}

/*
function squareMatrix(n, val){
    if(typeof(n)==='undefined' || isNaN(n)) { return []; }
    if(typeof ArrayBuffer === 'undefined') {
      // lacking browser support
      var arr = new Array(n);
      for(var i=0;i<n;i++) { arr[i] = val; }
      return arr;
    } else {
      var array = new Float64Array(n); // typed arrays are faster
      for(var i=0;i<n;i++) { array[i] = val; }
      return array;
    }
}
*/

//////////////////////////////// Clustering /////////////////////////////////
function tSNEcomputeClustering() {
  //compute cluster only when embedding is computed
  if(top.tSNEembeddingResult === undefined) return;

  //save parameter to json
  params =  {
        "wordVecs":top.tSNEdataMatrix,
        "method":document.getElementById('tSNE_select_clusterMethod').value,
        "numCluster":Number(document.getElementById('tSNE_input_clusterNum').value)
    };

  //send data to server
  postToServer(params, '/_applySubspaceCluster', function(result){
   if(result.result==='Error') {
        alert('Clustering Computing Error!');
        return;
      }

   top.tSNEclusterLabel = result.label;

   top.tSNELabel = top.tSNEclusterLabel;
   //top.subDistanceMatrix = result.distanceMatrix;
   //top.subSpacesBasis = result.subspaceBasis;
   document.getElementById('tSNEcolorMapScheme').value = 'ClassLabel';

   //updateplot
   d3tSNEplot();   
  });
}

//////////////////////////////// Distortion /////////////////////////////////
function computeColorMap(){
  if(top.tSNEembeddingResult)
  {
	var colormapOption = d3.select("#tSNEcolorMapScheme").node();
    var colormapType = colormapOption.options[colormapOption.selectedIndex].value;

    if(colormapType === "KDE"){
        //async
	    computeKernelDensityonServer(top.tSNEdataMatrix, function(result){
	       	 top.tSNEperPointDistortion = result.density;
             //updateplot
    	     updatetSNEpoints2D();
	       });
	}
	else if(colormapType === "RankingError"){
	    var neighborSize = Number($rankingDistortionkNN.val());
		//compute rank distortion
		top.LDNeighbor = computePointNeighbor(top.tSNEembeddingResult, neighborSize,  cosineDistance);
		top.HDNeighbor = computePointNeighbor(top.tSNEdataMatrix, neighborSize,  cosineDistance);

		top.tSNEperPointDistortion = rankingDistortion(top.HDNeighbor, top.LDNeighbor);
		//normalize distortion range
		var minDistortion = Math.min.apply(null, top.tSNEperPointDistortion);
		var maxDistortion = Math.max.apply(null, top.tSNEperPointDistortion);
		//normalize
		//top.tSNEperPointDistortion = top.tSNEperPointDistortion.map(
		//	function(x) { return (x/maxDistortion)*(x/maxDistortion); }
		//  );
	    //updateplot
    	updatetSNEpoints2D();		  
    }
    else if(colormapType === "tSNEcost"){

    	if(top.tSNEmethod){
    		var perpointCost = top.tSNEmethod.getPerPointCost();
    	}else{
    		var perpointCost = tSNEperPointCost(top.tSNEdataMatrix, top.tSNEembeddingResult);
    	}
    	
    	top.tSNEperPointDistortion = [];
		for(var i=0; i<perpointCost.length; i++){
			top.tSNEperPointDistortion.push(Math.sqrt(Math.abs(perpointCost[i])));
		}    		
    	//updateplot
    	updatetSNEpoints2D();    	
    }
    else if(colormapType === "Outliers"){
    	var NN = computePointNeighbor(top.tSNEdataMatrix, 2,  cosineDistance);
    	top.tSNEperPointDistortion = [];
    	for(var i=0; i<NN.length; i++){
			top.tSNEperPointDistortion.push(NN[i][1].distance);
    	}
	    //updateplot
    	updatetSNEpoints2D();
    }
    else if(colormapType === "FrequencyRank"){

    	if(top.tSNEwordFrequencyRank && 
    	   top.tSNEwordFrequencyRank.length === top.tSNEperPointDistortion.length){
    	   	 top.tSNEperPointDistortion = [];
			 for(var i=0; i<top.tSNEwordFrequencyRank.length; i++){
				top.tSNEperPointDistortion.push(
					Math.sqrt(Math.sqrt(Math.sqrt(top.tSNEwordFrequencyRank[i])))
				);
			 }    	   
    	   }
    		
	    //updateplot
    	updatetSNEpoints2D();    	
    }
    else if(colormapType === "Outliers/FreqRank"){

    	var NN = computePointNeighbor(top.tSNEdataMatrix, 2,  cosineDistance);

        if(top.tSNEwordFrequencyRank && 
    	   top.tSNEwordFrequencyRank.length === top.tSNEperPointDistortion.length){
    	   	 top.tSNEperPointDistortion = [];
			 for(var i=0; i<top.tSNEwordFrequencyRank.length; i++){
				top.tSNEperPointDistortion.push(
    	        (NN[i][1].distance*NN[i][1].distance)/
				(0.1+Math.log(top.tSNEwordFrequencyRank[i]))
				);
			 }    	   
    	   }
    		
	    //updateplot
    	updatetSNEpoints2D();    	
    }
    else if(colormapType === "EmbedComparison"){
	    var neighborSize = Number($rankingDistortionkNN.val());
		//compute rank distortion
		top.HDNeighborCompare = computePointNeighbor(top.tSNEdataMatrixCompare, neighborSize,  cosineDistance);
		top.HDNeighbor = computePointNeighbor(top.tSNEdataMatrix, neighborSize,  cosineDistance);

		top.tSNEperPointDistortion = rankingDistortion(top.HDNeighbor, top.HDNeighborCompare);
        for(var i=0; i<top.tSNEwordFrequencyRank.length; i++){
        	top.tSNEperPointDistortion[i] = top.tSNEperPointDistortion[i]
				/(Math.log(top.tSNEwordFrequencyRank[i]));
		}  
	    //updateplot
    	updatetSNEpoints2D();    
    }
    else if(colormapType === "NeighborShape"){

    	var neighborSize = Number($rankingDistortionkNN.val());
    	computePerPointMeasureOnServer(top.tSNEdataMatrix, neighborSize, "multiSense", function(result){
    	   top.tSNEperPointDistortion = result.perPointMeasure;
    	   //updateplot
    	   updatetSNEpoints2D();
    	});
    	/*
		top.HDNeighbor = computePointNeighbor(top.tSNEdataMatrix, neighborSize,  cosineDistance);

		var N = top.tSNEdataMatrix.length;

		top.tSNEperPointDistortion = [];
		//for every point
		for(var i=0; i<N; i++){
			var longestPair = 0.0;
			//find the longest pairwise distance in a neighborhood
			for(var j=1; j<neighborSize-1; j++){
				var index1 = top.HDNeighbor[i][j].index;
				for(var k=1; k<neighborSize-1; k++){
					var index2 = top.HDNeighbor[i][k].index;
					if(i>j){
						var dist = cosineDistance(top.tSNEdataMatrix[index1], top.tSNEdataMatrix[index2]);
						if(dist>longestPair)
						   longestPair = dist;
					}
				}
			}
			top.tSNEperPointDistortion.push(longestPair);
		}
	    //updateplot
    	updatetSNEpoints2D();
    	*/
    }
    else if(colormapType === "ClassLabel"){
        top.tSNELabel = top.tSNEclusterLabel;
    	d3tSNEplot();
    }
    else if(colormapType === "WordNetDist"){
    	if(top.tSNEselectedPointIndex){
			var wordPairs = [];
			var referenceWord = top.tSNEwords[top.tSNEselectedPointIndex];
			for(var i=0; i<top.tSNEwords.length; i++){
				wordPairs.push([referenceWord, top.tSNEwords[i]]);
			}

			postToServer({"pairs":wordPairs}, '/_queryWordNetDistance', function(result){
				top.tSNEperPointDistortion = result.pairDistance;
				updatetSNEpoints2D(); 
			});    		
    	}
    }
    else if(colormapType === "WordNetLabel"){
    	top.tSNEwordNetLabel = [];
    	//generate label from dict
    	for(var i=0; i<top.tSNEwords.length; i++){
    		top.tSNEwordNetLabel.push(top.wordNetLabel[top.tSNEwords[i]]);
    	}
    	
		//top.tSNELabel = tSNEwordNetLabel;
		d3tSNEplot();
    }
  }
}

///////////////////////////// word search //////////////////////////////
function searchWord(){	
	//console.log(d3.select('#input_tSNEwordSearch').node().value);
	if(top.tSNEpoints2D){
		var word = d3.select('#input_tSNEwordSearch').node().value;
		for(var i=0; i<top.tSNEpoints2D.length; i++){
			if(top.tSNEpoints2D[i][2] === word ||
			   top.tSNEpoints2D[i][2] === word.toLowerCase() ||
			   top.tSNEpoints2D[i][2].toLowerCase() === word){
                    if(top.tSNEpoints2D[i].length<4)
                    	return;
					if(top.istSNEtoolTipActive){
					   cleantSNEtooltip();
					   top.istSNEtoolTipActive = false;
					}
					//update selection index
					top.tSNEselectedPointIndex = i;

					var svgPoint = d3.selectAll(".dot")[0][i];

					addtSNEtooltip(Number(svgPoint.getAttribute('cx')),
					               Number(svgPoint.getAttribute('cy')),
					               top.tSNEpoints2D[i][2],
					               top.tSNEpoints2D[i][3]);
			   }
		}
	}
}

///////////////////////////// for display /////////////////////////////////

function generateWordEmbeddingPlot(embeddingResult){
  //deep copy array of array
  top.tSNEpoints2D = JSON.parse(JSON.stringify(embeddingResult))
  //top.tSNEpoints2D = $.extend(true, [], embeddingResult);  
  for(var i=0; i<top.tSNEpoints2D.length; i++){
      top.tSNEpoints2D[i].push(top.tSNEwords[i]);
      top.tSNEpoints2D[i].push(i);
  }
  d3tSNEplot();
}

function updatetSNEpoints2D(){
  if(top.tSNEpoints2D){
    if(top.tSNEpoints2D[0].length === 4){
      for(var i=0; i<top.tSNEpoints2D.length; i++){
          top.tSNEpoints2D[i].push(top.tSNEperPointDistortion[i]);
      }
    }else
    {
      for(var i=0; i<top.tSNEpoints2D.length; i++){
          top.tSNEpoints2D[i][4]=top.tSNEperPointDistortion[i];
      }
    }
  }
  d3tSNEplot();
}

function clearEmbedding(){
    if(d3.select("#svg-tSNEembedding").empty()){
        d3.select("#tSNEembedding").append("svg").attr("id", "svg-tSNEembedding")    
    }
    else{
        d3.select("#svg-tSNEembedding").selectAll("*").remove();
    }
}

function updateDistortionDisplay(index){
	//compute distance to point
	var smallest = Number.MAX_VALUE;
	top.tSNEdistanceToCurrentPoint = [];
	var dist = 0;
	for(var i=0; i<top.tSNEdataMatrix.length; i++){
		if(index !== i){
			dist = cosineDistance(top.tSNEdataMatrix[index], top.tSNEdataMatrix[i]);
			if(dist < smallest)
			   smallest = dist;

		}
		top.tSNEdistanceToCurrentPoint.push(dist);
	}
	//update for the current point
	top.tSNEdistanceToCurrentPoint[index] = smallest;

	//update point2D
    for(var i=0; i<top.tSNEpoints2D.length; i++){
        top.tSNEpoints2D[i][4]=top.tSNEdistanceToCurrentPoint[i];
    }	
	d3tSNEplot(top.tSNEdistanceToCurrentPoint, ["yellow", "green"]);
}

////////////////// drawing code ////////////////////

function d3tSNEplot(value, colorMap){

    value = value || top.tSNEperPointDistortion;
    colorMap = colorMap || ["yellow", "red"]

    //get the mapping function
    var points2D = top.tSNEpoints2D;

    //compute data x, y aspect ratio
    var X = points2D.map(function(p){return p[0]});
    var Y = points2D.map(function(p){return p[1]});

    var rangeX = Math.max.apply(null, X)-Math.min.apply(null, X);
    var rangeY = Math.max.apply(null, Y)-Math.min.apply(null, Y);
    var aspectRatio = rangeX/rangeY;

    var rect = d3.select("#tSNEembedding").node().getBoundingClientRect()
    var margin = {top: 30, bottom: 10, right: 30, left: 30};    
    var width = rect.width - margin.left - margin.right;
    //var height = rect.width/aspectRatio - margin.top - margin.bottom;
    var height = width;
    top.tSNEmargin = margin;
    top.tSNEwidth = width;


    //add color scale and colormap    
    var min = Math.min.apply(null, value);
    var max = Math.max.apply(null, value);
    top.tSNEdistortionColorMap = d3.scale.linear().domain([min, max]).range(colorMap);      

    if(d3.select("#svg-tSNEembedding").empty()){
        d3.select("#tSNEembedding").append("svg").attr("id", "svg-tSNEembedding")    
    }
    else{
        d3.select("#svg-tSNEembedding").selectAll("*").remove();
    }

    //var svg = d3.select("svg");
    var svg = d3.select("#svg-tSNEembedding");
        svg
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    svg.on("click", function(){
				if(top.istSNEtoolTipActive){
				   cleantSNEtooltip(); 
				   top.istSNEtoolTipActive = false;
				}        		
				updatetSNEpoints2D();
        	}
        );

    var x = d3.scale.linear()
        .range([margin.left, width]);

    var y = d3.scale.linear()
        .range([margin.top, height]);

    x.domain(d3.extent(points2D, function(d) { return d[0]; }));
    y.domain(d3.extent(points2D, function(d) { return d[1]; }));

    //lasso ////////////////////////////////////////////////

	// Create the area where the lasso event can be triggered
	var lasso_area = svg.append("rect")
						  .attr("width",width)
						  .attr("height",height)
						  .style("opacity",0);

	// Define the lasso
	var lasso = d3.lasso()
		  .closePathDistance(2000) // max distance for the lasso loop to be closed
		  .closePathSelect(true) // can items be selected by closing the path?
		  .hoverSelect(true) // can items by selected by hovering over them?
		  .area(lasso_area) // area where the lasso can be started
		  //.on("start", function(){
		       //lasso.items().classed({"not_possible":true,"selected":false}); // style as not possible
		       //lasso.items().classed({"possible":true}); // style as not possible

		  //}) // lasso start function
		  .on("draw", function(){
			  // Style the possible dots
			  lasso.items().filter(function(d) {return d.possible===true})
				.classed({"not_selected":false,"selected":true});

			  // Style the not possible dot
			  lasso.items().filter(function(d) {return d.possible===false})
				.classed({"not_selected":true,"selected":false});
           }) // lasso draw function
		  .on("end", function(){
		      top.tSNElassoSelection = [];
			  for(var i=0; i<lasso.items()[0].length;i++){
			  	top.tSNElassoSelection.push(
			  		lasso.items()[0][i].getAttribute('class') === 'dot selected');
			  }
           }); // lasso end function
		  //.on("end",lasso_draw); // lasso end function
	// Init the lasso on the svg:g that contains the dots
	svg.call(lasso);
    ////////////////////////////////////////////////////////

    //////////////////////// drag //////////////////////////

    var drag = d3.behavior.drag()
        .on("drag", function(d, i) {
        	d3.select(this).attr("cx", d.x = d3.event.x).attr("cy", d.y = d3.event.y);
        	if (this.nextSibling) this.parentNode.appendChild(this);
        	var dataX = x.invert(d.x);
        	var dataY = y.invert(d.y);
        	top.tSNEpoints2D[i][0] = dataX;
        	top.tSNEpoints2D[i][1] = dataY;
        	top.tSNEembeddingResult[i][0] = dataX;
        	top.tSNEembeddingResult[i][1] = dataY; 
        })
        .on("dragend", function(d,i){
        	computeColorMap();
        });
    ////////////////////////////////////////////////////////

    //draw points///////////////////////////////////////////
    svg.selectAll(".dot")
       .data(points2D)
     .enter().append("circle")
       .attr("id",function(d,i) {return "dot_" + i;}) // added
       .attr("class", function(d, i){
       	  if(top.tSNElassoSelection){
       	  	 if(top.tSNElassoSelection.indexOf(true)===-1)
       	  	     return 'dot'; //if no thing is selected
       	  	 else{
				 if(top.tSNElassoSelection[i])
					return 'dot';
				 else
					return 'dot not_selected';       	  	 	
       	  	 }
       	  }else{
       	  	 return 'dot';
       	  }
       })       
       .attr("cx", function(d) { return x(d[0]);})
       .attr("cy", function(d) { return y(d[1]);})
       .attr("r",function(d) { return 7;})
       .style("fill", function(d,i) 
       { 
         //////////////// for single label ////////////////
         if( d3.select('#tSNEcolorMapScheme').node().value ==='ClassLabel'
           && top.tSNELabel){
           	 if(top.tSNELabel[i]===-1)
           	 	return "black";//for DBSCAN
           	 else
           	 	return tSNEd3Color20(top.tSNELabel[i]);
         }
         //////////////// for multilabel ////////////////
         else if(d3.select('#tSNEcolorMapScheme').node().value ==='WordNetLabel'){
         	var labelList = top.tSNEwordNetLabel[i];
         	if(labelList.length===1){
         		if(labelList[0]===-1)
           	 		return "black";
           	 	else
           	 		return tSNEd3Color20(labelList[0]);
				
         	}else{
				return "black";
         	}
         }
         else{
			 if(d.length>4)
			   return top.tSNEdistortionColorMap(d[4]);
			 else
			   return d3Color10(0);         
         }
       })
       .on("click", function(d){
       	    if (d3.event.defaultPrevented) return; // dragged
            if(top.istSNEtoolTipActive){
               cleantSNEtooltip(); 
               top.istSNEtoolTipActive = false;
            }
            updateDistortionDisplay(d[3]); //pass in index
            d3.event.stopPropagation();
       })
       .on("mousemove", function(d) {            
            if(top.istSNEtoolTipActive){
               cleantSNEtooltip(); 
               top.istSNEtoolTipActive = false;
            }		

			top.tSNEselectedPointText = d[2];
			//store the hovered index
			top.tSNEselectedPointIndex = d[3];

            addtSNEtooltip(x(d[0]),y(d[1]), d[2], d[3]);
            d3.event.stopPropagation();            
       })
       .on("mouseout", function(){
            //cleantSNEtooltip();  
       })
       .call(drag);

	lasso.items(d3.selectAll(".dot"));
     
 	//add or remove colormap
 	if(d3.select('#tSNEcolorMapScheme').node().value !== 'ClassLabel' &&
 	   d3.select('#tSNEcolorMapScheme').node().value !== 'WordNetLabel'){
 	   	
		if(d3.select('#tSNEColorBarFlag').node().checked){
			if(top.tSNEperPointDistortion)
				addColorBarIntSNEdistortionPlot([min, max], colorMap);
		}else{
			//remove the colormap
			d3.select("#tSNEcolorMapBar").remove();    	
		}
    }
}

function isPointNotOverlapTo(point, listOfPoints, threshold){
	if (threshold == undefined)
		threshold = 30;
	var isClose = true;

	for(var i=0; i<listOfPoints.length; i++){
		if(Math.abs(point[1] - listOfPoints[i][1])<20 &&
		   distance(point, listOfPoints[i])<threshold)
			return false;
	}
	return isClose;
}

/////////////////////////// Tool tip ///////////////////////////////
function addtSNEtooltip(x, y, text, index){  
  top.istSNEtoolTipActive = true;
  var svg = d3.select("#svg-tSNEembedding");
  
  //add tooltip words
  var clickedDataSet = [];
  clickedDataSet.push([x, y, text]);

  //draw lines
  var neighborRanking = [[text, 0, 0.0]];

  var numOfNeighbors = ($HDNeighborShown.val()/100.0)*Number($rankingDistortionkNN.val());
  if(top.LDNeighbor && $('#tSNEHDNeighborFlag').get(0).checked)
  {
    var circles = svg.selectAll(".dot")[0];    
    var lines = [];
    var hIndex, lIndex;
    var HDlineDist = [];
    var HDpoint2DList = [];
    top.tooltipLineIndex = []; //store the line point index
    top.tooltipTextIndex = [];
    for(var i=1; i<numOfNeighbors; i++){
    
      hIndex = top.HDNeighbor[index][i].index;
      var distToCurrent = top.HDNeighbor[index][i].distance;
      var HDNeighbor2Dpoint = [Number(circles[hIndex].getAttribute("cx")), 
                               Number(circles[hIndex].getAttribute("cy"))];
      lines.push([[x, y], 
                  HDNeighbor2Dpoint,
                  (1.0-i/numOfNeighbors)*0.8+0.2, //drawing line width
                  1]);
      
      top.tooltipLineIndex.push(hIndex);

      //store point info for determining the tooltip
      HDlineDist.push(distance([x,y], HDNeighbor2Dpoint));
      HDpoint2DList.push([x,y])

      //add tooltip list for HD neighbors
      neighborRanking.push([top.tSNEpoints2D[hIndex][2], i, distToCurrent]);

      if(isPointNotOverlapTo(HDNeighbor2Dpoint, HDpoint2DList)){
      	    
         top.tooltipTextIndex.push(hIndex);
      	    
         clickedDataSet.push( [HDNeighbor2Dpoint[0], 
                               HDNeighbor2Dpoint[1], 
                               top.tSNEpoints2D[hIndex][2]]);
         HDpoint2DList.push(HDNeighbor2Dpoint); //storing existing tooltip location
      }
    }

    svg.selectAll("line[id='tooltip']")  
       .data(lines)
       .enter()
       .append("line")       
       .attr("x1", function(d){return d[0][0];})
       .attr("y1", function(d){return d[0][1];})
       .attr("x2", function(d){return d[1][0];})
       .attr("y2", function(d){return d[1][1];})
       .attr("stroke-width", function(d){return d[2]*3;})
       //.attr("stroke-width", 3.0)
       //.attr("stroke", function(d) { if(d[3]) return "blue"; else return "green"; });
       .attr("stroke", function(){return "blue";});
       //.style("opacity", function(d) { return d[2]; });

  }else{
    for(var i=1; i<numOfNeighbors; i++){
      if(top.HDNeighbor){
		  var hIndex = top.HDNeighbor[index][i].index;
		  var distToCurrent = top.HDNeighbor[index][i].distance;
		  neighborRanking.push([top.tSNEpoints2D[hIndex][2], i, distToCurrent]);
      }
    }
  }

  ///////////////// word tooltip ///////////////
	
  if(d3.select("#tSNEToolTipFlag").node().checked){
	  // add rectangles
	  var fontSize = 14;
	  var padding = 2;
	  var rectHeight = fontSize + 2 * padding; //based on font size
	  var yOffset = -8;

	  svg.selectAll("rect[id='tooltip']")
			.data(clickedDataSet).enter()
				.append("rect")
				.attr("x", function(d) { return d[0] - d[2].length * 5;})
				.attr("y", function(d) { return d[1] - rectHeight + 2 * padding + yOffset; })
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
				.attr("y", function(d) { return d[1]+yOffset; })
				.attr("id", "tooltip")
				.attr("fill", "white")
				.attr("text-anchor", "middle")
				.attr("font-size", fontSize)
				.text( function (d){ return d[2];})


	  var maxDist = neighborRanking[neighborRanking.length-1][2];

	  //console.log(maxDist);  
	  neighborRanking = neighborRanking.map(
		function(d){return [d[0], d[1], d[2], (1.0-d[2]/maxDist)*0.4+0.6];});
	  //console.log(neighborRanking);
      var format = d3.format(".2f");
	  var rect = d3.select("#tSNEembedding").node().getBoundingClientRect();
	  svg.select("rect[id='tooltipNeighborBackground']").remove();
	  svg.append("rect")
				.attr("x", function() { return rect.width-150; })
				.attr("y", function(d) { return 0; })
				.attr("rx", function() { return 5; })
				.attr("ry", function(d) { return 5; })
				.attr("width", function() { return 130; })
				.attr("height", function(d) { return 20 + neighborRanking.length*15; })
				.attr("fill", "#cccccc")
				.attr("opacity", 0.85)
				.attr("id", "tooltipNeighborBackground");
	        
	  svg.selectAll("text[id='tooltipNeighbor']")
			.data(neighborRanking).enter()
				.append("text")
				.attr("x", function() { return rect.width-140; })
				.attr("y", function(d) { return 20 + d[1]*15; })
				.attr("id", "tooltipNeighbor")
				.attr("fill", "black")
				.attr("text-anchor", "left")
				.attr("font-size", fontSize)
				//.attr("opacity", function (d){ return d[3];})
				.text( function (d,i){ 
				  if(i===0)
				    return d[0];
				  else
					return d[0]+' ('+format(d[2])+')';
			     });

      //////////////// show word comparison //////////////
      if(top.HDNeighborCompare){
          var neighborRanking = [];
          for(var i=0; i<numOfNeighbors; i++){
			var hIndex = top.HDNeighborCompare[index][i].index;
			var distToCurrent = top.HDNeighborCompare[index][i].distance;
			neighborRanking.push([top.tSNEpoints2D[hIndex][2], i, distToCurrent]);          	
          }
		  var maxDist = neighborRanking[neighborRanking.length-1][2];

		  //console.log(maxDist);  
		  neighborRanking = neighborRanking.map(
			function(d){return [d[0], d[1], d[2], (1.0-d[2]/maxDist)*0.4+0.6];});
		  //console.log(neighborRanking);
		  var format = d3.format(".2f");
		  var rect = d3.select("#tSNEembedding").node().getBoundingClientRect();
		  svg.selectAll("text[id='tooltipNeighborCompare']")
				.data(neighborRanking).enter()
					.append("text")
					.attr("x", function() { return rect.width-250; })
					.attr("y", function(d) { return 15 + d[1]*15; })
					.attr("id", "tooltipNeighbor")
					.attr("fill", "black")
					.attr("text-anchor", "left")
					.attr("font-size", fontSize)
					.attr("opacity", function (d){ return d[3];})
					.text( function (d,i){ 
					  if(i===0)
						return d[0];
					  else
						return d[0]+' ('+format(d[2])+')';
					 });      	
      }
  }//tooltip text ended

  //update colormap if required
  //if(document.getElementById('tSNEcolorMapScheme').value === 'WordNetDist'){
  //	computeColorMap();
  //}

}

function cleantSNEtooltip(){
	// removes any object under #tooltip is
	d3.select("#svg-tSNEembedding").selectAll("#tooltip").remove();
	d3.select("#svg-tSNEembedding").selectAll("line").remove();
	d3.select("#svg-tSNEembedding").selectAll("#tooltipNeighbor").remove();	
	d3.select("#svg-tSNEembedding").selectAll("#tooltipNeighborCompare").remove();	
}


///// colormap /////
function colorMaptSNE(colorMap, d){
   if(d.length>4)
    return colorMap(d[4]);
   else
    return d3Color10(0);
}

function addColorBarIntSNEdistortionPlot(domain, color){
    var svg = d3.select("#svg-tSNEembedding");
    //remove the colormap
    d3.select("#tSNEcolorMapBar").remove();

    g = svg.append("g")
           .attr("id", "tSNEcolorMapBar")
           .attr("transform","translate(10,10)")
           .classed("colorbar",true);
           
    cb = colorBar()
           .color(d3.scale.linear().domain(domain).range(color))
           .size(200)
           .lineWidth(40)           
           .precision(4);

    g.call(cb);
}

///////////////////// split words ///////////////////////
function perPointSplitWords(){	
    var selectPointIndex = top.tSNEselectedPointIndex;
    var neighborSize = ($HDNeighborShown.val()/100.0)*Number($rankingDistortionkNN.val());

    var dataMatrix = JSON.parse(JSON.stringify(top.tSNEdataMatrix));
    var embedding = JSON.parse(JSON.stringify(top.tSNEembeddingResult));
	splitWords(dataMatrix, embedding, selectPointIndex, neighborSize);

}


