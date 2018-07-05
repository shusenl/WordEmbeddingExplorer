function queryErrorPDF(dataDim, wordCount, projMethod, callback, embeddingPos){
    if(projMethod==='SVMplane'){
      var svg = d3.select("#svg-pairProjection");
      svg.selectAll("g").remove();
      return;
    }

    //embedding is optional
    if(embeddingPos === undefined)
       embeddingPos = [];

    var embeddingDim = 2;
    postToServer(
        {
          "dataDim": dataDim,
          "wordCount": wordCount,
          "projMethod": projMethod,
          "embedding":embeddingPos
        }, '/_queryRandomAnalogyProjectionErrorPDF', callback);
}

//handle pure drawing code
function drawPDFHUD(lineData){

}

//handle plot generation
function generatePDFHUD(result){
  if(jQuery.isEmptyObject(result))
    return;
  //random pdf capture the baseline case
  //query histo can be a pdf or a single value (non-bootstrap case)
  var histo = result.histo[0];
  var bins = result.histo[1];
  var error = result.error;
  var projError = result.projError;

  histo.push(0);
  histo.unshift(0);
  var binSize=bins[1]-bins[0];
  bins.push(bins[bins.length-1]+binSize);
  //console.log(histo, bins, error, projError);
  console.log(error, projError);
  var dataY = histo;
  var lineData = histo.map((data, index)=>[bins[index], data]);

  var width = 200;
  var height = 100;
  var x = d3.scale.linear()
        .domain(d3.extent(bins.concat([projError])))
        .range([10, width]);

  var y = d3.scale.linear()
        .domain(d3.extent(dataY))
        .range([height, 0]);

  var lineGenerator = d3.svg.line()
                        .x(function(d) { return x(d[0]); })
                        .y(function(d) { return y(d[1]); })
                        .interpolate("linear");

  if(dataY.length && dataY.length>1){
     var svg = d3.select("#svg-pairProjection");

    //drawing HUD
    svg.selectAll("g").remove();

    var group = svg.append("g").attr("id", "PDFHUD");
    group.append("rect")
           .attr("width", width)
           .attr("height", height)
           .attr("x", 10)
           .attr("y", 0)
           .attr("rx", 4)
           .attr("ry", 4)
           .style("fill", "grey")
           .style("opacity", 0.2);

    group.append("path")
           .attr("d", lineGenerator(lineData))
           .attr("stroke", "blue")
           .attr("stroke", "blue")
           .attr("fill", "rgba(0,0,200,0.2)")
           //.attr("opacity",0.4);

    group.append("line")          // attach a line
            .style("stroke", "red")  // colour the line
            .style("stroke-dasharray", ("3, 3"))
            .attr("x1", x(projError))     // x position of the first end of the line
            .attr("y1", 0)      // y position of the first end of the line
            .attr("x2", x(projError))     // x position of the second end of the line
            .attr("y2", height);    // y position of the second end of the line

    var trianglePoints1 = x(projError)-7+", "+ (height+14)+' '
                           +(x(projError)+7)+", "+ (height+14)+' '
                           +x(projError)+", "+height;

    group.append('polyline')
            .attr('points', trianglePoints1)
            .style('fill', 'red')
            .append('text', projError);

    /*
    group.append("line")          // attach a line
            .style("stroke", "blue")  // colour the line
            .style("stroke-dasharray", ("3, 3"))
            .attr("x1", x(error))     // x position of the first end of the line
            .attr("y1", 0)      // y position of the first end of the line
            .attr("x2", x(error))     // x position of the second end of the line
            .attr("y2", height);    // y position of the second end of the line

    var trianglePoints2 = x(error)-7+", "+ (height+14)+' '
                           +(x(error)+7)+", "+ (height+14)+' '
                           +x(error)+", "+height;

    group.append('polyline')
            .attr('points', trianglePoints2)
            .style('fill', 'blue')
            .append('text', error);
     */
    }
}
