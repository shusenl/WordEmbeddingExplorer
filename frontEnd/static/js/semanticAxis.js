d3.select('#btn-semanticAxis-project').on('click', computeSemanticProjection);

d3.select('#axis_input').on('change', computeSinglePairSemanticAxisMinMax);
//TODO implement fixed axis definition
d3.select('#axis_input_x').on('change', function() {
    console.log("testX");
});
d3.select('#axis_input_y').on('change', function() {
    console.log("testY");
});

///////////////// computation ////////////////////
function getSinglePairAxis(id) {
    var wordList = d3.select(id).node().value.split(":");
    //lookup word1
    var wordVec1 = top.wordVecs[wordList[0]];
    if (wordVec1 === undefined)
        wordVec1 = top.mostFrequentWordVecs[wordList[0]];
    //lookup word2
    var wordVec2 = top.wordVecs[wordList[1]];
    if (wordVec2 === undefined)
        wordVec2 = top.mostFrequentWordVecs[wordList[1]];
    var directVec = [];
    for (var i = 0; i < wordVec1.length; i++) {
        directVec.push(wordVec1[i] - wordVec2[i]);
    }
    return {
        "analogyPair": wordList,
        "directVec": directVec
    };
}

function computeSinglePairSemanticAxisMinMax() {
    var singleAnalogyData = getSinglePairAxis("#axis_input");

    postToServer({
        'directVec': singleAnalogyData["directVec"]
    }, '/_computeRotationToVector', function(p) {
        computeSemanticAxisMinMax(p.reflection, singleAnalogyData["analogyPair"]);
    });
}

function buildingAnalogyLinearProjectionQueryData(groupName) {
    var pairListFull = Object.keys(top.analogyGroupObject[groupName]).slice(1);
    //slice pairList
    //var pairList = pairListFull.slice(0, 3);
    var pairList = pairListFull;
    //console.log(pairList);
    var selectedPairs = [];
    selectedPairs = selectedPairs.concat(pairList);
    var wordToExclude = extractWordsFromPairs(selectedPairs);
    var dataMatrix = buildDataMatrixFromPairs(selectedPairs);
    /// compute SVM linear separation
    var binaryLabel = [];
    for (var i = 0; i < dataMatrix.length; i++) {
        if (i % 2 === 0)
            binaryLabel.push(0);
        else
            binaryLabel.push(1);
    }
    return {
        "dataMatrix": dataMatrix,
        "binaryLabel": binaryLabel,
        "wordToExclude": wordToExclude
    };
}

function computeAnalogyGroupSemanticAxisMinMax() {
    var groupName = top.semanticAxis_currentAxis;
    var queryData = buildingAnalogyLinearProjectionQueryData(groupName);

    server_generateOptimalAnalogyLinearProjection(queryData['dataMatrix'], queryData['binaryLabel'], 'SVM+PCA', function(p) {
        computeSemanticAxisMinMax(p.reflection, queryData['wordToExclude']);
    });
}

//global!!! FIXME
var allWords = []
  , allDataMatrix = [];

function computeSemanticAxis(reflection) {

    //compute only when word vectors are empty
    if (allWords.length === 0 || allDataMatrix.length === 0) {
        if (top.mostFrequentWords) {
            allWords = top.mostFrequentWords;
            allDataMatrix = buildDataMatrixFromWordsWithSource(allWords, top.mostFrequentWordVecs);
        } else {
            allWords = Object.keys(top.wordVecs);
            allDataMatrix = buildDataMatrixFromWordsWithSource(allWords, top.wordVecs);
        }
    }

    //speed bottleneck here
    //var t0 = performance.now();
    var rotatedData = numeric.dot(allDataMatrix, reflection);
    //var t1 = performance.now();
    //console.log( 'sec:', (t1-t0)/1000);

    var axisPair = [];
    for (var i = 0; i < rotatedData.length; i++) {
        axisPair.push([allWords[i], rotatedData[i][0]]);
    }
    //*
    axisPair.sort(function(a, b) {
        if (a[1] < b[1])
            return -1;
        if (a[1] > b[1])
            return 1;
        return 0;
    })
    //*/
    return axisPair;
}

function computeSemanticAxisMinMax(reflection, wordToExclude) {
    var wordToExcludeSet = new Set(wordToExclude);
    for (var i = 0; i < wordToExclude.length; i++) {
        wordToExcludeSet.add(wordToExclude[i].toLowerCase());
    }

    var axisPair = computeSemanticAxis(reflection);

    var topCount = 25;
    var data = [];

    var concept1TopWords = [];
    var concept2TopWords = [];
    for (var i = 0; i < axisPair.length; i++) {
        if (i < topCount || i > axisPair.length - 1 - topCount) {
            if (wordToExcludeSet.has(axisPair[i][0]))
                data.push(['\(' + axisPair[i][0] + '\)', axisPair[i][1]]);
            else
                data.push([axisPair[i][0], axisPair[i][1]]);
        }
        if (i < topCount)
            concept1TopWords.push(axisPair[i][0]);
        if (i > axisPair.length - 1 - topCount)
            concept2TopWords.push(axisPair[i][0]);
    }
    displaySemanticAxisMinMax(data);
    //discoverAnalogyRelationship(concept1TopWords, concept2TopWords);
}

function computeSemanticAxisRange(rotation, data) {//inspect first dimension
}

var testData = [['word1', 0.7], ['word2', 0.1], ['word3', -0.3], ['word4', -0.5]];
////////////////// display //////////////////
function displaySemanticAxisMinMax(inputData) {
    var data;
    if (inputData === undefined)
        data = testData;
    else
        data = inputData;
    //var aspectRatio = 0.45;
    var aspectRatio = 0.8;
    //cleanup existing drawing
    if (!d3.select("#panel_semanticAxisMinMax").empty()) {
        d3.select("#panel_semanticAxisMinMax").selectAll("*").remove();
    }
    var rect = d3.select("#panel_semanticAxisMinMax").node().getBoundingClientRect();
    var margin = {
        top: rect.width / aspectRatio * 0.00,
        bottom: rect.width / aspectRatio * 0.05,
        right: rect.width * 0.05,
        left: rect.width * 0.02
    };
    var width = rect.width - margin.left - margin.right;
    var height = rect.width / aspectRatio - margin.top - margin.bottom;
    var x = d3.scale.linear().range([0, width]);
    var y = d3.scale.ordinal().rangeRoundBands([0, height], 0.1);
    var xAxis = d3.svg.axis().scale(x).orient("bottom").tickSize(3);
    var yAxis = d3.svg.axis().scale(y).orient("left").tickSize(0).tickPadding(6);

    var svg = d3.select("#panel_semanticAxisMinMax").append("svg").attr("width", width + margin.left + margin.right).attr("height", height + margin.top + margin.bottom).append("g").attr("transform", "translate(" + margin.left + "," + margin.top + ")");
    y.domain(data.map(function(d) {
        return d[0];
    }));
    x.domain(d3.extent(data, function(d) {
        return d[1];
    })).nice();
    svg.selectAll(".bar").data(data).enter().append("rect").attr("class", function(d) {
        return "bar bar--" + (d[1] < 0 ? "negative" : "positive");
    }).attr("x", function(d) {
        return x(Math.min(0, d[1]));
    }).attr("y", function(d) {
        return y(d[0]);
    }).attr("width", function(d) {
        return Math.abs(x(d[1]) - x(0));
    }).attr("height", y.rangeBand());
    svg.append("g").attr("class", "x axis")
                   .attr("transform", "translate(0," + height + ")")                   
                   .call(xAxis);
                     
    svg.append("g").attr("class", "y axis")
                   .attr("transform", "translate(" + x(0) + ",0)")
                   .call(yAxis);

    //increase tick font size
    //d3.selectAll(".tick > text").style("font-size","40px");
}

function computeSemanticProjection() {
    var groupNameX = top.semanticAxis_currentAxis_X;
    var groupNameY = top.semanticAxis_currentAxis_Y;

    var queryDataX = buildingAnalogyLinearProjectionQueryData(groupNameX);
    var queryDataY = buildingAnalogyLinearProjectionQueryData(groupNameY);

    server_generateOptimalAnalogyLinearProjection(queryDataX['dataMatrix'], queryDataX['binaryLabel'], 'SVM+PCA', function(p) {
        var axis_pair_x = computeSemanticAxis(p.reflection);
        server_generateOptimalAnalogyLinearProjection(queryDataY['dataMatrix'], queryDataY['binaryLabel'], 'SVM+PCA', function(p) {
            var axis_pair_y = computeSemanticAxis(p.reflection);
            //generate the projection
            generateSemanticAxisPlot(axis_pair_x, axis_pair_y, 8000);
        });
    });
}

////////////////////////////// 2D Plot //////////////////////////////
function generateSemanticAxisPlot(axis_pair_x, axis_pair_y, topCount) {
    var selectedWords = {};
    console.assert(axis_pair_x.length === axis_pair_y.length);

    //find the extreme case words
    for (var i = 0; i < axis_pair_x.length; i++) {
        if (i < topCount || i > axis_pair_x.length - 1 - topCount) {
            selectedWords[axis_pair_x[i][0]] = [0, 0];
            selectedWords[axis_pair_y[i][0]] = [0, 0];
        }
    }
    //lookup coordinate
    for (var i = 0; i < axis_pair_x.length; i++) {
        if (selectedWords.hasOwnProperty(axis_pair_x[i][0])) {
            selectedWords[axis_pair_x[i][0]][0] = axis_pair_x[i][1];
        }
        if (selectedWords.hasOwnProperty(axis_pair_y[i][0])) {
            selectedWords[axis_pair_y[i][0]][1] = axis_pair_y[i][1];
        }
    }
    var plotData = []
    Object.keys(selectedWords).forEach(function(key) {
        plotData.push([key, selectedWords[key][0], selectedWords[key][1]]);
    });

    //console.log(plotData);
    var aspectRatio = 0.8;
    //cleanup existing drawing
    if (!d3.select("#panel_semanticAxis_projection").empty()) {
        d3.select("#panel_semanticAxis_projection").selectAll("*").remove();
    }
    var rect = d3.select("#panel_semanticAxis_projection").node().getBoundingClientRect();
    var margin = {
        top: rect.width / aspectRatio * 0.05,
        bottom: rect.width / aspectRatio * 0.15,
        right: rect.width * 0.05,
        left: rect.width * 0.05
    };
    var width = rect.width - margin.left - margin.right;
    var height = rect.width / aspectRatio - margin.top - margin.bottom;
    var svg = d3.select("#panel_semanticAxis_projection").append("svg").attr("width", width + margin.left + margin.right).attr("height", height + margin.top + margin.bottom).append("g").attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    // define range map
    var x = d3.scale.linear().range([0, width]);
    var y = d3.scale.linear().range([height, 0]);
    x.domain(d3.extent(plotData.map(p=>p[1])));
    y.domain(d3.extent(plotData.map(p=>p[2])));
    var xAxis = d3.svg.axis().scale(x).orient("bottom");
    var yAxis = d3.svg.axis().scale(y).orient("left");

    // x-axis
    svg.append("g").attr("class", "axis")
                   .attr("transform", "translate(0," + y(0) + ")")
                   .call(xAxis)
                   .append("text")
                   .attr("class", "label")
                   .attr("x", width)
                   .attr("y", -6)
                   .style("font-size","15px")
                   .style("text-anchor", "end")
                   .text(top.semanticAxis_currentAxis_X);

    // y-axis
    svg.append("g").attr("class", "axis")
                   .attr("transform", "translate(" + x(0) + ",0)")
                   .call(yAxis)
                   .append("text")
                   .attr("class", "label")
                   .attr("transform", "rotate(-90)")
                   .attr("y", 6)
                   .attr("dy", ".71em")
                   .style("font-size","15px")
                   .style("text-anchor", "end")
                   .text(top.semanticAxis_currentAxis_Y);

    // add the tooltip area to the webpage
    var tooltip = d3.select("#panel_semanticAxis_projection").append("div")
    .attr("class", "tooltip")
    .style("opacity", 0);
    
    // draw dots
    svg.selectAll(".dot")
                .data(plotData)
                 .enter()
                 .append("circle")
                 .attr("class", "dot")
                 .attr("r", 3.5)
                 .attr("opacity",0.2)
                 .attr("cx", function(d) {
        return x(d[1]);
    }).attr("cy", function(d) {
        return y(d[2]);
    })
    //.style("fill", function(d) { return color(cValue(d));});
    //*
      .on("mouseover", function(d) {
          tooltip
               .style("opacity", 1.0)
               .style("color", "red")
               .style("font-size","20px");

          tooltip.html(d[0])               
                 .style("left", (d3.event.pageX + 5) + "px")
                 .style("top", (d3.event.pageY - 15) + "px");
      })
      .on("mouseout", function(d) {
          tooltip.style("opacity", 0);
      });
      //*/

}

////////////////////////////// experiments //////////////////////////////
function discoverAnalogyRelationship(group1, group2) {
    var allDirectionVec = [];
    var allPairWordVec = [];
    for (var i = 0; i < group1.length; i++)
        for (var j = 0; j < group2.length; j++) {
            if (j > i) {
                var pair = group1[i] + ':' + group2[j];
                allDirectionVec.push([vecDiff(top.mostFrequentWordVecs[group1[i]], top.mostFrequentWordVecs[group2[j]]), pair]);
                //allPairWordVec.push([top.mostFrequentWordVecs[group1[i]], top.mostFrequentWordVecs[group2[j]]], [group1[i],group2[j]]);
            }
        }
    var allPairFound = [];
    var allQuadFound = [];
    //*
    for (var i = 0; i < allDirectionVec.length; i++) {
        for (var j = 0; j < allDirectionVec.length; j++) {
            if (j > i) {
                var words = allDirectionVec[i][1].split(':').concat(allDirectionVec[j][1].split(':'));
                var wordSet = new Set(words);
                if (wordSet.size !== 4)
                    continue;
                var dist = cosineDistance(allDirectionVec[i][0], allDirectionVec[j][0]);
                if (dist < 0.35) {
                    //if( Math.abs(allDirectionVec[i][2]-allDirectionVec[j][2]) < 0.02)
                    //    console.log(allDirectionVec[i][1]+'->'+allDirectionVec[j][1], dist);
                    allPairFound.push([allDirectionVec[i][1] + '->' + allDirectionVec[j][1], dist])
                }
            }
        }
    }
    //*/
    /*
    var referencePair = ['boy','girl'];
    for (var i = 0; i < allDirectionVec.length; i++) {
        var words = allDirectionVec[i][1].split(':').concat(referencePair);
        var wordSet = new Set(words);
        //if (wordSet.size !== 4)
        //    continue;
        var dist = cosineDistance(allDirectionVec[i][0], 
             vecDiff(top.mostFrequentWordVecs[referencePair[0]],top.mostFrequentWordVecs[referencePair[1]]));
        //if(dist<0.25){
        //if (Math.abs(allDirectionVec[i][2] - allDirectionVec[j][2]) < 0.02)
            //    console.log(allDirectionVec[i][1]+'->'+allDirectionVec[j][1], dist);
            allPairFound.push([allDirectionVec[i][1], dist])
        //}
    }
    */
    console.log("before sorting, len:", allPairFound.length);

    allPairFound.sort(function(a, b) {
        if (a[1] < b[1])
            return -1;
        if (a[1] > b[1])
            return 1;
        return 0;
    })

    console.log(allPairFound);
}
