////////////////////////// tSNE cost ///////////////////////////
// utilitity that creates contiguous vector of zeros of size n
function zeros(n) {
    if(typeof(n)==='undefined' || isNaN(n)) { return []; }
    if(typeof ArrayBuffer === 'undefined') {
      // lacking browser support
      var arr = new Array(n);
      for(var i=0;i<n;i++) { arr[i]= 0; }
      return arr;
    } else {
      return new Float64Array(n); // typed arrays are faster
    }
}
// compute L2 distance between two vectors
var L2 = function(x1, x2) {
  var D = x1.length;
  var d = 0;
  for(var i=0;i<D;i++) {
    var x1i = x1[i];
    var x2i = x2[i];
    d += (x1i-x2i)*(x1i-x2i);
  }
  return d;
}
// compute pairwise distance in all vectors in X
function xtod(X) {
  var N = X.length;
  var dist = zeros(N * N); // allocate contiguous array
  for(var i=0;i<N;i++) {
    for(var j=i+1;j<N;j++) {
      var d = L2(X[i], X[j]);
      dist[i*N+j] = d;
      dist[j*N+i] = d;
    }
  }
  return dist;
}

// compute (p_{i|j} + p_{j|i})/(2n)
function d2p(D, perplexity, tol) {
    var Nf = Math.sqrt(D.length); // this better be an integer
    var N = Math.floor(Nf);
    //assert(N === Nf, "D should have square number of elements.");
    var Htarget = Math.log(perplexity); // target entropy of distribution
    var P = zeros(N * N); // temporary probability matrix

    var prow = zeros(N); // a temporary storage compartment
    for(var i=0;i<N;i++) {
      var betamin = -Infinity;
      var betamax = Infinity;
      var beta = 1; // initial value of precision
      var done = false;
      var maxtries = 50;

      // perform binary search to find a suitable precision beta
      // so that the entropy of the distribution is appropriate
      var num = 0;
      while(!done) {
        //debugger;

        // compute entropy and kernel row with beta precision
        var psum = 0.0;
        for(var j=0;j<N;j++) {
          var pj = Math.exp(- D[i*N+j] * beta);
          if(i===j) { pj = 0; } // we dont care about diagonals
          prow[j] = pj;
          psum += pj;
        }
        // normalize p and compute entropy
        var Hhere = 0.0;
        for(var j=0;j<N;j++) {
          var pj = prow[j] / psum;
          prow[j] = pj;
          if(pj > 1e-7) Hhere -= pj * Math.log(pj);
        }

        // adjust beta based on result
        if(Hhere > Htarget) {
          // entropy was too high (distribution too diffuse)
          // so we need to increase the precision for more peaky distribution
          betamin = beta; // move up the bounds
          if(betamax === Infinity) { beta = beta * 2; }
          else { beta = (beta + betamax) / 2; }

        } else {
          // converse case. make distrubtion less peaky
          betamax = beta;
          if(betamin === -Infinity) { beta = beta / 2; }
          else { beta = (beta + betamin) / 2; }
        }

        // stopping conditions: too many tries or got a good precision
        num++;
        if(Math.abs(Hhere - Htarget) < tol) { done = true; }
        if(num >= maxtries) { done = true; }
      }

      // console.log('data point ' + i + ' gets precision ' + beta + ' after ' + num + ' binary search steps.');
      // copy over the final prow to P at row i
      for(var j=0;j<N;j++) { P[i*N+j] = prow[j]; }

    } // end loop over examples i

    // symmetrize P and normalize it to sum to 1 over all ij
    var Pout = zeros(N * N);
    var N2 = N*2;
    for(var i=0;i<N;i++) {
      for(var j=0;j<N;j++) {
        Pout[i*N+j] = Math.max((P[i*N+j] + P[j*N+i])/N2, 1e-100);
      }
    }

    return Pout;
  }

function computeQ(Y){
  var N, dim;
  N = Y.length;
  dim = Y[0].length;

  var pmul = this.iter < 100 ? 4 : 1; // trick that helps with local optima

  // compute current Q distribution, unnormalized first
  var Qu = zeros(N * N);
  var qsum = 0.0;
  for(var i=0;i<N;i++) {
    for(var j=i+1;j<N;j++) {
      var dsum = 0.0;
      for(var d=0;d<dim;d++) {
        var dhere = Y[i][d] - Y[j][d];
        dsum += dhere * dhere;
      }
      var qu = 1.0 / (1.0 + dsum); // Student t-distribution
      Qu[i*N+j] = qu;
      Qu[j*N+i] = qu;
      qsum += 2 * qu;
    }
  }
  // normalize Q distribution to sum to 1
  var NN = N*N;
  var Q = zeros(NN);
  for(var q=0;q<NN;q++) {
    Q[q] = Math.max(Qu[q] / qsum, 1e-100);
  }
  return Q;
}

function tSNEperPointCost(X, Y, perplexity){
   if(perplexity === undefined)
      perplexity = 40;

   var dists = xtod(X); // convert X to distances using gaussian kernel

   //compute P, Q
   var P = d2p(dists, perplexity, 1e-4); // attach to object
   var Q = computeQ(Y);

   var N = X.length;
   var perPointCost = new Array(N).fill(0);
   for(var i=0; i<N; i++)
     for(var j=0; j<N; j++){
      perPointCost[i] += P[i*N+j] * Math.log(P[i*N+j]/Q[i*N+j]);
   }
   return perPointCost;
}


////////////////////////// Ranking distortion ///////////////////////////

function computePointNeighbor(dataMatrix, numOfNeighbor, distFunc){
    var pointNeighbor = [];
    var kNN = numOfNeighbor;
    for(var i=0; i<dataMatrix.length; i++){
        pointNeighbor.push(new Array(kNN).fill({index:0, distance:Number.MAX_SAFE_INTEGER}));

        for(var j=0; j<dataMatrix.length; j++){
            var dist =  distFunc(dataMatrix[i], dataMatrix[j]);
            if(pointNeighbor[i][kNN-1].distance > dist)
                pointNeighbor[i][kNN-1] = {index:j, distance:dist};

            for(var k=kNN-1; k>0; k--){
                if(pointNeighbor[i][k-1].distance > pointNeighbor[i][k].distance){//swap
                    var temp = pointNeighbor[i][k-1];
                    pointNeighbor[i][k-1] = pointNeighbor[i][k];
                    pointNeighbor[i][k] = temp;
                }
            }
        }
    }
    return pointNeighbor;
}

function rankingDistortion(HDpointNeighbor, LDpointNeighhor, type){
    var distortion = [];
    var perPointDistortion;
    if(type === undefined){
        type = "difference";
        //type = "rankDifference";
    }

    if(type === "difference"){
        for(var i=0; i<HDpointNeighbor.length; i++){
            var set = {};
            for(var j=0; j<HDpointNeighbor[i].length; j++)
                set[HDpointNeighbor[i][j].index] = true;
            perPointDistortion = 0;
            for(var j=0; j<HDpointNeighbor[i].length; j++)
                if(set[LDpointNeighhor[i][j].index]!==true)
                    perPointDistortion = perPointDistortion + 1;
            distortion.push(perPointDistortion);
        }
    }else if(type === "rankDifference"){
        var maxNeighborSize = HDpointNeighbor[0].length;
        for(var i=0; i<HDpointNeighbor.length; i++){

            //store the HD neighbor index
            var HDrank = {};
            for(var j=0; j<HDpointNeighbor[i].length; j++)
                HDrank[HDpointNeighbor[i][j].index] = j;

            perPointDistortion = 0;

            for(var j=0; j<HDpointNeighbor[i].length; j++)
                if(HDrank[LDpointNeighhor[i][j].index] !== undefined){
                    perPointDistortion = perPointDistortion +
                       Math.abs(HDrank[LDpointNeighhor[i][j].index] - j)
                }else{
                    perPointDistortion = perPointDistortion + maxNeighborSize;
                }
            distortion.push(perPointDistortion);
        }
    }
    return distortion;
}
