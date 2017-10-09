#include "MultiRover.h"

MultiRover::MultiRover(vector<double> w, size_t numSteps, size_t numPop, size_t
		       numPOIs, Fitness f, size_t rovs, int c, AgentType t)
  : world(w), nSteps(numSteps), nPop(numPop), nPOIs(numPOIs), nRovers(rovs),
    coupling(c), fitness(f), outputEvals(false), outputTrajs(false),
    outputQury(false), outputBlf(false), gPOIObs(false), type(t), verbose(true),
    biasStart(true) {

  initRovers();
}

MultiRover::MultiRover(vector<double> w, size_t numSteps, size_t numPop, size_t
		       numPOIs, Fitness f, size_t rovs, int c,
		       vector< vector<NeuralNet> > nets, vector<vector<size_t>> inds,
		       bool controlled)
  : world(w), nSteps(numSteps), nPop(numPop), nPOIs(numPOIs), nRovers(rovs),
    coupling(c), fitness(f), outputEvals(false), outputTrajs(false),
    outputQury(false), outputBlf(false), gPOIObs(false), verbose(true),
    biasStart(true) {

  size_t nOut = inds.size();
  for (size_t i = 0; i < nRovers; i++) {
    vector<NeuralNet> netsAgent;
    for (size_t j = 0; j < nets.size(); j++) {
      int netI = nets[j].size() == rovs ? i : 0;
      netsAgent.push_back(nets[j][netI]);
    }
    if (controlled) {
      roverTeam.push_back(new Controlled(nSteps, nPop, fitness, netsAgent, inds, nOut));
      type = AgentType::C;
    } else {
      roverTeam.push_back(new NeuralRover(nSteps, nPop, fitness, netsAgent, inds, nOut));
      type = AgentType::M;
    }
  }
}

MultiRover::~MultiRover() {
  for (size_t i = 0; i < nRovers; i++){
    delete roverTeam[i] ;
    roverTeam[i] = 0 ;
  }
  if (outputEvals)
    evalFile.close() ;
  if (outputTrajs){
    trajFile.close() ;
    POIFile.close() ;
  }
}

void MultiRover::initRovers() {
  roverTeam.clear();
  
  for (size_t i = 0; i < nRovers; i++) {
    if (type == AgentType::R) {
      roverTeam.push_back(new Rover(nSteps, nPop, fitness));
    } else if (type == AgentType::P) {
      roverTeam.push_back(new OnlyPOIRover(nSteps, nPop, fitness));
      setCoupling(1);
    } else if (type == AgentType::A) {
      roverTeam.push_back(new TeamFormingAgent(nSteps, nPop, fitness, coupling));
    } else if (type == AgentType::E) {
      roverTeam.push_back(new ExploringAgent(nSteps, nPop, fitness, coupling));
    }
  }
}
void MultiRover::InitialiseEpoch(){
  double rangeX = world[1] - world[0] ;
  double rangeY = world[3] - world[2] ;
  
  initialXYs.clear() ;
  initialPsis.clear() ;
  for (size_t i = 0; i < nRovers; i++){
    Vector2d initialXY ;
    if (type == AgentType::A || !biasStart) {
      initialXY(0) = rand_interval(world[0],world[1]);
      initialXY(1) = rand_interval(world[2],world[3]);
    } else {
      initialXY(0) = rand_interval(world[0]+rangeX/3.0,world[1]-rangeX/3.0);
      initialXY(1) = rand_interval(world[2]+rangeY/3.0,world[3]-rangeX/3.0);
    }
    double initialPsi = rand_interval(-PI,PI) ;
    initialXYs.push_back(initialXY) ;
    initialPsis.push_back(initialPsi) ;
  }
  
  // POI locations and values in global frame (restricted to within outer regions of 9 grid)
  POIs.clear() ;
  for (size_t p = 0; p < nPOIs; p++){
    Vector2d xy ;
    double x, y ;
    bool accept = false ;
    while (!accept){
      x = rand_interval(world[0],world[1]) ;
      y = rand_interval(world[2],world[3]) ;
      
      accept = !(x > world[0]+rangeX/3.0 &&
		 x < world[1]-rangeX/3.0 &&
		 y > world[2]+rangeY/3.0 &&
		 y < world[3]-rangeX/3.0);
    }
    
    xy(0) = x ; // x location
    xy(1) = y ; // y location
    double v = rand_interval(1,10) ; // value
    POIs.push_back(Target(xy, v, coupling));
  }
}

void MultiRover::InitialiseEpochFromVectors(vector<Target> targets,
					    vector<Vector2d> xys) {
  initialXYs = xys;
  POIs = targets;

  initialPsis.clear();
  for (size_t i = 0; i < nRovers; i++) {
    double initialPsi = rand_interval(-PI,PI);
    initialPsis.push_back(initialPsi);
  }
}


vector< vector<size_t> > MultiRover::RandomiseTeams(size_t n){
  vector< vector<size_t> > teams ;
  vector<size_t> order ;
  for (size_t i = 0; i < n; i++) {
    order.push_back(i) ;
  }
  
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count() ;

  for (size_t j = 0; j < nRovers; j++){
    shuffle (order.begin(), order.end(), std::default_random_engine(seed)) ;
    teams.push_back(order) ;
  }
  
  return teams ;
}

void MultiRover::SimulateEpoch(bool train){
  size_t teamSize ;

  teamSize = train ? 2*nPop : nPop;
    
  // each row is the population for a single agent
  vector< vector<size_t> > teams = RandomiseTeams(teamSize) ; 

  if (outputTrajs) {
    printPOIs();
  }
  
  double maxEval = 0.0 ;

  Env env(world, roverTeam, POIs, teamSize);
  for (size_t i = 0; i < teamSize; i++){ // looping across the columns of 'teams'
    // Initialise world and reset rovers and POIs
    vector<Vector2d> jointState ;
    for (size_t j = 0; j < nRovers; j++){
      roverTeam[j]->InitialiseNewLearningEpoch(POIs,initialXYs[j],initialPsis[j]) ;
      jointState.push_back(initialXYs[j]) ;
    }
    
    if (outputTrajs && i == teamSize - 1) {
      printJointState(jointState);
      toggleAgentOutput(true);
    }
    
    tempPOIs = POIs;
    
    for (size_t t = 0; t < nSteps; t++) {
      vector<Vector2d> newJointState ;
      double G = 0.0 ;
      if (type == AgentType::C) {
	std::cout << *this << std::endl;
      }
      for (size_t j = 0; j < nRovers; j++){ // looping down the rows of 'teams'
        Vector2d xy = roverTeam[j]->ExecuteNNControlPolicy(teams[j][i],jointState) ;
        newJointState.push_back(xy);
	agentObserves(xy, t);
      }

      if (fitness == Fitness::D) {
	calculateStepwiseG();

	for (auto& rov : roverTeam) {
	  rov->DifferenceEvaluationFunction(newJointState, G);
	}
      }
      
      jointState = newJointState;
      newJointState.clear();
      
      if (outputTrajs) {
        printJointState(jointState);
      }
    }
    
    double eval = calculateG();
    resetDomain();
    maxEval = max(eval, maxEval);
    
    // Assign fitness
    for (size_t j = 0; j < nRovers; j++) {
      roverTeam[j]->SetEpochPerformance(eval, teams[j][i]) ;
    }
    
    if (outputEvals) {
      evalFile << eval << "," ;
    }
  }
  
  if (outputEvals) {
    evalFile << std::endl;
  }

  if (verbose) {
    std::cout << "max achieved value: " << maxEval << "..." << std::endl;
  }
}

void MultiRover::EvolvePolicies(bool init){
  std::cout << "MultiRover:EvolvePolicies:start" << std::endl;
  for (auto& rov : roverTeam) {
    std::cout << "MultiRover:EvolvePolicies:loop" << std::endl;
    rov->EvolvePolicies(init);
    std::cout << "MultiRover:EvolvePolicies:loop:neuro" << rov->GetNEPopulation() << std::endl;
  }
  std::cout << "MultiRover:EvolvePolicies:end" << std::endl;
}

void MultiRover::ResetEpochEvals(){
  for (auto& rov : roverTeam) {
    rov->ResetEpochEvals();
  }
}

// Wrapper for writing epoch evaluations to specified files
void MultiRover::OutputPerformance(std::string fileName){
  if (evalFile.is_open()) {
    evalFile.close();
  }
  
  evalFile.open(fileName.c_str(),std::ios::app);
  
  outputEvals = true;
}

// Wrapper for writing final trajectories to specified files
void MultiRover::OutputTrajectories(std::string tFile, std::string poiFile,
				    std::string tChoiceFile) {
  if (trajFile.is_open()) {
    trajFile.close();
  }
  trajFile.open(tFile.c_str(),std::ios::app) ;
  
  if (POIFile.is_open()) {
    POIFile.close();
  }
  POIFile.open(poiFile.c_str(),std::ios::app) ;

  for (auto& rov : roverTeam) {
    rov->openOutputFile(tChoiceFile);
  }
  
  outputTrajs = true ;
}

// Wrapper for writing final control policies to specified file
void MultiRover::OutputControlPolicies(std::string nnFile) {
  for (auto const& rover : roverTeam) {
    rover->OutputNNs(nnFile);
  }
}

void MultiRover::ExecutePolicies(string readFile, string storeTraj,
				 string storePOI, string storeEval,
				 size_t numIn, size_t numOut, size_t numHidden) {

  loadNNs(readFile, numIn, numHidden, numOut);
  

  if (getVerbose()) {
    std::cout << "Initialising test world..." << std::endl;
  }
  
  InitialiseEpoch() ;
  OutputPerformance(storeEval) ;
  OutputTrajectories(storeTraj, storePOI, "d") ;
  ResetEpochEvals() ;
  SimulateEpoch(false) ; // simulate in test mode
}

void MultiRover::loadNNs(string nnFile, size_t numIn, size_t numHidden,
			 size_t numOut) {
  if (getVerbose()) {
    std::cout << "Reading in neural network policies from " << nnFile
	      << std::endl;
  }

  vector<NeuralNet*> loadedNN = NeuralNet::loadNNFromFile(nnFile, numIn, numHidden,
							  numOut);

  int k = 0;
  for (auto& rov : roverTeam) {
    NeuroEvo* rovNE = rov->GetNEPopulation();
    for (size_t i = 0; i < getNPop(); i++) {
      rovNE->GetNNIndex(i)->SetWeights(loadedNN[k]->GetWeightsA(),
				       loadedNN[k]->GetWeightsB());
      k++;
    }
  }

  for (size_t i = 0; i< loadedNN.size(); i++) {
    delete loadedNN[i];
    loadedNN[i] = 0;
  }
}

void MultiRover::loadNNsNeuralRover(vector<string> nnFiles, vector<size_t> nIns,
				    vector<size_t> nHiddens, vector<size_t> nOuts,
				    vector<vector<size_t>> inds) {

  vector< vector<NeuralNet*> > nets;
  for (size_t i = 0; i < nnFiles.size(); i++) {
    size_t nIn = nIns.size() == nnFiles.size() ? i : 0;
    size_t nH = nHiddens.size() == nnFiles.size() ? i : 0;
    size_t nOut = nOuts.size() == nnFiles.size() ? i : 0;
    vector<NeuralNet*> subNets = NeuralNet::loadNNFromFile(nnFiles[i], nIns[nIn],
							   nHiddens[nH], nOuts[nOut]);
    nets.push_back(subNets);
  }

  roverTeam.clear();
  
  size_t nOut = inds.size();
  for (size_t i = 0; i < getNRovers(); i++) {
    vector<NeuralNet> netsAgent;
    for (auto& net : nets) {
      int netI = net.size() == getNRovers() ? i : 0;
      netsAgent.push_back(*net[netI]);
    }
    roverTeam.push_back(new NeuralRover(getNSteps(), getNPop(), getFitness(),
					netsAgent, inds, nOut));
  }
}

vector<NeuralNet*> MultiRover::getNNTeam() {
  std::vector<NeuralNet*> nets;
  for (auto& rov : roverTeam) {
    nets.push_back(rov->GetNEPopulation()->GetNNIndex(0));
  }
  return nets;
}

void MultiRover::printPOIs() {
  for (auto const& poi : POIs) {
    POIFile << poi << std::endl;
  }
}
 
void MultiRover::printJointState(const vector<Vector2d> jointState) {
  for (const auto& vec : jointState) {
    trajFile << vec(0) << "," << vec(1) << ",";
  }
  trajFile << std::endl;
}

void MultiRover::toggleAgentOutput(bool toggle) {
  for (auto& rov : roverTeam) {
    rov->setOutputBool(toggle);
  }
}

void MultiRover::agentObserves(Vector2d xy, int time) {
  if (type == AgentType::A || type == AgentType::E) {
    for (const auto& other : roverTeam) {
      Vector2d otherLoc = other->getCurrentXY();
      if (!(otherLoc(0) == xy(0) && otherLoc(1) == xy(1))) {
	TeamFormingAgent* tA = (TeamFormingAgent*) other;
	tA->ObserveTarget(xy, time);
      }
    }
  } else {
    for (auto& poi : POIs) {
      poi.ObserveTarget(xy, time);
    }
 
    if (fitness == Fitness::D) {
      for (auto& tempPoi : tempPOIs) {
 	tempPoi.ObserveTarget(xy, time);
      }
    }
  }
}

void MultiRover::resetDomain() {
  resetAgents();
  resetPOIs();
}

void MultiRover::resetAgents() {
  if (type == AgentType::A || type == AgentType::E) {
    for (auto& rov : roverTeam) {
      TeamFormingAgent* tA = (TeamFormingAgent*) rov;
      tA->ResetTarget();
    }
  }
}

void MultiRover::resetPOIs() {
  for (auto& poi : POIs) {
    poi.ResetTarget();
  }
}
double MultiRover::calculatePOIValue(const Target& poi) {
  return poi.IsObserved() ? (poi.GetValue() / max(poi.GetNearestObs(), 1.0)) : 0.0;
}

double MultiRover::calculateG() {
  double G = 0.0;

  for (auto& rover : roverTeam) {
    G += rover->getReward();
  }

  for (auto& poi : POIs) {
    G += calculatePOIValue(poi);
  }

  return G;
}

double MultiRover::calculateStepwiseG() {
  double G = 0.0;
  
  for (const auto& poi : tempPOIs) {
    G += calculatePOIValue(poi);
  }
  
  return G;
}

std::ostream& operator<<(std::ostream &strm, const MultiRover &d) {
  double height = d.world[3] - d.world[2];
  double width  = d.world[1] - d.world[0];

  int h = floor(height);
  int w = floor(width);
  
  double vals[h][w];
  for (int i = 0; i < h; i++) {
    for (int j = 0; j < w; j++) {
      vals[i][j] = 0;
    }
  }

  for (const auto& poi : d.POIs) {
    Vector2d loc = poi.GetLocation();
    int fh = (int) (floor (loc(0)));
    int fw = (int) (floor (loc(1)));
    int ph = (fh + h) % h;
    int pw = (fw + w) % w;

    vals[ph][pw] += poi.GetValue();
    strm << poi << std::endl;
  }
  
  for (const auto& rov : d.roverTeam) {
    strm << *rov << std::endl;
  }

  for (int i = 0; i < h; i++) {
    for (int j = 0; j < w; j++) {
      strm << vals[i][j] << " ";
    }
    strm << std::endl;
  }
  
  return strm;
}
