//global variable
top.words = [];

// $('#btn-uploadCSVFile').click(
// function(){
//   $('#uploadCSVFile').click();
//
// })

$('#btn-pairData').click(
function(){
  $('#analogyFiles').click();

})

// $('#uploadCSVFile').change(loadUserCSVFile)
// $('#analogyFiles').change(handleFileSelect(extractAnalogyPairFromText))

////////////////// load user CSV file to server //////////////////
// function loadUserCSVFile(selectFileStream){
//   console.log(selectFileStream);
// }

/////////////////// load file / process files / send words to server //////////////////////

function handleFileSelect(extractMethod)
{
  return function(evt){
      var files = evt.target.files; // FileList object
      // files is a FileList of File objects. List some properties.
      var output = [];
      //only load the first one
      fileName = files[0];
      readFile(fileName, extractMethod);
  }
}

//read the text file, parse it, and send to server
function readFile(f, extractMethod){
  var reader = new FileReader();
  reader.onload = function(e){
    extractMethod(reader.result);
  };
  reader.readAsText(f);
}

function queryServerForVectors(words, dbString, callback){
  $(function()
  {
    //var dbString = $('#select_wordEmbeddingDatabase').val();
    var dbName = dbString.split("-")[0];
    var collectionName = dbString.split("-")[1];
    //send words to server
    $.ajax({
          type: "POST",
          url: '/_fetchWordVectors',
          // The key needs to match your method's input parameter (case-sensitive).
          data: JSON.stringify({"words":words, "dbName":dbName, "collectionName":collectionName}),
          contentType: "application/json; charset=utf-8",
          dataType: "json",
          success: function(resp) {callback(resp)},
          failure: function(errMsg) {alert(errMsg);}
      });
  });
}

/////////////////////////// For Exploring Analogy Pairs /////////////////////////
function extractAnalogyPairFromText(text){
  var pairCount = 0;
  var wordsObj = {};
  var wordLabel = {};
  //analogyGroup: key are analogy pair, value are not important
  var groups = text.split(":")
  var analogyGroupObject = {};
  for(var i=0; i<groups.length; i++){
    var analogyGroup = {};
    if(groups[i].length!==0){
      var wordList = groups[i].trim().split(/\s|\n/)
      //console.log(wordList);
      for(var j=0; j<wordList.length-1; j++)
        if(j===0)
          analogyGroup['state'] = {type:wordList[j], selected:false};
          //analogyGroup['type'] = wordList[j];
        else{
          //add pairs
          analogyGroup[wordList[j]+'-'+wordList[j+1]] = false;
          //false/true indicate if the pairs is highlighted
          j++;
        }
      //save individual words in words array
      for(var j=1; j<wordList.length; j++){
          wordsObj[wordList[j]]=0;

          if(wordLabel[wordList[j]]===undefined)
            if(i<=2)
                wordLabel[wordList[j]]=0;
            else
                wordLabel[wordList[j]]=i-2;
      }

      pairCount += Object.keys(analogyGroup).length-1;//first key is reserve for storing state
      //analogyGroupArray.push(analogyGroup);
      analogyGroupObject[analogyGroup.state.type] = analogyGroup;
    }
  }
  //generate label for input file
  //console.log(wordLabel)

  $('#pair-list').text(
      'Analogy Pairs: '+pairCount + '\t' +
      //'Analogy Group: '+analogyGroupArray.length
      'Analogy Group: ' + Object.keys(analogyGroupObject).length
    )
  var analogySize = [];
  for(var key in analogyGroupObject){
    if(analogyGroupObject.hasOwnProperty(key)){
      analogySize.push(Object.keys(analogyGroupObject[key]).length-1);
    }
  }
  // console.log(analogySize);

  //global variable
  top.analogyGroupObject = analogyGroupObject;

  var words = Object.keys(wordsObj);

  //updateUI
  $('#btn-examplePairData').html('Loading From Server ...');
  queryServerForVectors(words, $('#select_wordEmbeddingDatabase').val(), function(result)
    {
        top.wordFrequency = result['frequencyRank'];
        var wordVecs = result['wordVec'];
        //store the returned vectors
        top.wordVecs = wordVecs;

        //temp
        top.tSNEwords = words;
        top.tSNEwordVecs = wordVecs;
        //update UI
        buildAnalogyGroupTableFromAnalogyGroupObject(analogyGroupObject);
        //updateUI
        $('#btn-examplePairData').html('Example Data Loaded');
        //console.log(cosineDistance(wordVecs['cat'], wordVecs['cats']));

        //experiments/////////////////////////////////////
        /*
        top.analogyVecs = [];
        for (var groupName in analogyGroupObject) {
        if (analogyGroupObject.hasOwnProperty(groupName)) {
            var pairList = Object.keys(analogyGroupObject[groupName]);
            //remove the type key
            pairList.shift();
            //selectedPairsByGroup.push(pairList);
            var vectorAverageDist = computeAverageCosineDistanceOfAnalogyVectors(pairList);
            //var conceptAverageDist = computeAnalogyDistanceRatio(pairList);
            //var conceptStatus = computeAnalogyDistanceRatio_perPairAverage(pairList);
            var conceptStatus = computeAnalogyStatus(pairList);

            //console.log(groupName, conceptStatus, vectorAverageDist);
            //console.log(conceptAverageDist,vectorAverageDist);

            //compute analogy direction
            var dataMatrix = buildDataMatrixFromPairs( pairList );

            var binaryLabel = [];
            for(var i=0; i<dataMatrix.length; i++){
                if(i%2 === 0)
                    binaryLabel.push(0);
                else
                    binaryLabel.push(1);
            }
            server_generateOptimalAnalogyLinearProjection(dataMatrix, binaryLabel, "Analogy",
              function(p){
                top.analogyVecs.push(p.projMat[0]);
              } );
          }
        }
        //*/

       //temp: load most frequent words
       if(document.getElementById("cb_mostFrequentData").checked){
       //*
         var dbString = $('#select_wordEmbeddingDatabase').val()
         jQuery.get("static/exampleData/google-10000-no-swears.txt", function(words){
            var wordList = words.split("\n");
            queryServerForVectors(wordList, dbString, function(result){
               top.mostFrequentWordVecs = result['wordVec'];
               top.mostFrequentWords = wordList;
               $('#btn-examplePairData').html('Most Frequent Word Loaded');
            });
         })
       }
       //*/

       /////////////////////// for semantic axis ////////////////////////
       //update analogy group selection
       var groupNames=[];
       for (var groupName in analogyGroupObject) {
         groupNames.push(groupName);
       }

       //update min max ui
       d3.select('#semanticAxis-axis-selector')
        .selectAll('li')
        .data(groupNames)
        .enter()
        .append('li')
        .attr('id', function(d){return 'axis'+d;})
        .append('a')
        .text(function(d) { return d;})
        .on('click', function(d)
          {
            top.semanticAxis_currentAxis = d;
            d3.select('#btn-axis').node().innerHTML = "Axis "+"("+d+")"+" <span class=\"caret\"></span>";
            computeAnalogyGroupSemanticAxisMinMax();
          } );

       //update Axis X, and Axis Y
       d3.select('#semanticAxis-axis-X')
        .selectAll('li')
        .data(groupNames)
        .enter()
        .append('li')
        .attr('id', function(d){return 'axis'+d;})
        .append('a')
        .text(function(d) { return d;})
        .on('click', function(d)
           {
             top.semanticAxis_currentAxis_X = d;
             d3.select('#btn-axis-x').node().innerHTML = "Axis-X "+"("+d+")"+" <span class=\"caret\"></span>";
           } );

       d3.select('#semanticAxis-axis-Y')
        .selectAll('li')
        .data(groupNames)
        .enter()
        .append('li')
        .attr('id', function(d){return 'axis'+d;})
        .append('a')
        .text(function(d) { return d;})
        .on('click', function(d)
          {
            top.semanticAxis_currentAxis_Y = d;
            d3.select('#btn-axis-y').node().innerHTML = "Axis-Y "+"("+d+")"+" <span class=\"caret\"></span>";
          } );
    });

}

///////////////////// tSNE explainer related /////////////////////
//this the general function that extract wordVec from text and server
function extractWordVecs(text){
  //remove words contain things other than charactor
  words = text.match(/\S+/g);//.match()
  wordSet = new Set();
  for(var i=0; i<words.length; i++)
    if(/^[a-zA-Z]+$/.test(words[i])){
      wordSet.add(words[i]);
    }

  //to do remove the global variable
  top.tSNEwords = [];
  for( var word of wordSet)
    top.tSNEwords.push(word);
  //return top.uniqueWords;
  //updateUI
  $('#btn-tSNEexampleData').html('Loading From Server ...');

  queryServerForVectors(top.tSNEwords, $('#select_wordEmbeddingDatabase_tSNE').val(),
    function(result){
        top.tSNEwordFrequencyRank = result['frequencyRank'];
        var wordVecs = result['wordVec'];
        top.tSNEwordVecs = wordVecs;
        $('#btn-tSNEexampleData').html('Example Data Loaded');
        $('#tSNElist').text('Word Count:'+top.tSNEwords.length);
      }
    );
}

//this is used for updating server wordVecs
function extractWordFromText(text){
  /////////////processing the text//////////////
  //remove words contain things other than charactor
  words = text.match(/\S+/g);//.match()
  wordSet = new Set();
  for(var i=0; i<words.length; i++)
    if(/^[a-zA-Z]+$/.test(words[i])){
      wordSet.add(words[i]);
    }

  top.words = [];
  for( var word of wordSet)
    top.words.push(word);

  updateWordVecOnServer(top.words, addWordsToList);
}

function addWordsToList(resp) {
    $('#list').text('Word Count:'+resp.words_count);
}

function updateWordVecOnServer(words, respCallback){
  var dbString = $('#select_wordEmbeddingDatabase').val();
  var dbName = dbString.split("-")[0];
  var collectionName = dbString.split("-")[1];
  //jQuery
  $(function()
  {
    //send words to server
    $.ajax({
          type: "POST",
          url: '/_lookupVector',
          // The key needs to match your method's input parameter (case-sensitive).
          data: JSON.stringify({"words":words, "dbName":dbName, "collectionName":collectionName}),
          contentType: "application/json; charset=utf-8",
          dataType: "json",
          success: respCallback,
          failure: function(errMsg) {alert(errMsg);}
      });
  });
}

function loadDefaultFile(fileName, processingFunction){
  $.get(fileName,function(data) {
   if (data === "ON" || data === "OFF") {
     alert("file reading error");
   }else {
     processingFunction(data);
   }
  });
}


function loadDefaultWordFile(processingFunction){
  return function() {
    var exampleData = d3.select('#select_exampleData').node().value;
    if(exampleData ==='AnalogyPairs')
      loadDefaultFile('/static/exampleData/analogyWords.txt', processingFunction);
    else if(exampleData === 'MostFrequent')
      loadDefaultFile('/static/exampleData/words800.txt', processingFunction);
    else //default
      loadDefaultFile('/static/exampleData/analogyWords.txt', processingFunction);
  }
}

function loadDefaultAnalogyPairsFile(processingFunction){
  return function() {
    loadDefaultFile('/static/exampleData/analogyPairs.txt', processingFunction);
  }
}

//for analogyGroup exploration
$('#analogyFiles').bind('change', handleFileSelect(extractAnalogyPairFromText));
$('#btn-examplePairData').bind('click', loadDefaultAnalogyPairsFile(extractAnalogyPairFromText));

//for tSNE explainer
$('#loadWords').bind('change', handleFileSelect(extractWordVecs));
$('#btn-tSNEexampleData').bind('click', loadDefaultWordFile(extractWordVecs));
