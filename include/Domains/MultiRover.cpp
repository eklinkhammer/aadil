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
  // for (size_t i = 0; i < nRovers; i++){
  //   delete roverTeam[i] ;
  //   roverTeam[i] = 0 ;
  // }
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

vector<Env*> MultiRover::createEnvs(vector< vector< Agent* > > agents,
				    vector<string> labels, size_t teamSize) {
  vector<Env*> envs;
  vector< vector< Agent* > > dupAgents = duplicateAll(agents);
  
  for (size_t i = 0; i < agents.size(); i++) {
    envs.push_back(new Env(world, dupAgents[i], POIs, teamSize, labels[i]));
  }

  return envs;
}

vector< vector< Agent* > > MultiRover::duplicateAll(vector< vector< Agent* > > agents) {
  size_t maxsize = 0;
  for (const auto& i : agents) {
    if (i.size() > maxsize) maxsize = i.size();
  }

  vector< vector< Agent* > > allAgents;
  for (auto& i : agents) {
    allAgents.push_back(duplicate(i, maxsize));
  }

  return allAgents;
}

vector< Agent* > MultiRover::duplicate(vector< Agent* > agents, size_t n) {
  if (agents.size() == n) return agents;

  for (size_t i = 0; i < n - agents.size(); i=i) {
    agents.push_back(agents[i]->copyAgent());
  }

  return agents;
}

void MultiRover::simulateWithAlignment(bool train, vector< vector< Agent* >> agents,
				       vector< string > labels) {
  size_t teamSize = train ? 2*nPop : nPop;
  std::cout << "Creating envs: " << std::endl;
  vector<Env*> envs = createEnvs(agents, labels, teamSize);

  vector< vector<size_t> > teams = RandomiseTeams(teamSize);

  vector< State > jointState ;
  vector< size_t > teamIndex;
  for (size_t i = 0; i < teamSize; i++) {
    for (size_t j = 0; j < nRovers; j++) {
      State s(initialXYs[j], initialPsis[j]);
      jointState.push_back(s);
      teamIndex.push_back(teams[j][i]);
    }
  }
  
  Env env(world, roverTeam, POIs, teamSize);
  env.init(jointState, teamIndex);

  for (auto& e : envs) {
    e->init(jointState, teamIndex);
  }

  vector< State > bestMove;
  double bestResult = 0;
  
  for (size_t t = 0; t < nSteps; t++) {
    for (auto& e : envs) {
      std::cout << "Using Env: " << e->getID() << std::endl;
      std::cout << "Last step: " << e->latestStepReward() << std::endl;
      double estimate = e->estimateRewardOfStep(e->nextStep());
      std::cout << "Estimate of following: " << estimate << std::endl;
      std::cout << "Estimate margin: " << estimate - e->currentReward() << std::endl;
    }
  }
}

void MultiRover::simulateWithAlignment(bool train, vector<Env*> envs) {
  size_t teamSize = train ? 2*nPop : nPop;
    
  // grab first network from each sim
  vector< size_t > index;
  for (size_t i = 0; i < nRovers; i++) {
    index.push_back(0);
  }
  
  Env* env = createSim(teamSize);
  for (auto& e : envs) {
    e->init(env->getCurrentStates(), index);
    e->setTargetLocations(POIs);
  }

  //envs.push_back(env);

  double equalCount = 0.0;
  double moveCount = 0.0;
  for (size_t t = 0; t < nSteps; t++) {
    double maxLast = 0;
    double maxEstimateMargin = 0;
    double maxEstimateActual = env->currentReward();
    std::cout << "Step: " << t << " Current Reward: " << maxEstimateActual << std::endl;
    std::cout << *this << std::endl;
    // std::cout << "Env current states:" << std::endl;
    // for (auto& s : env->getCurrentStates()) {
    // 	std::cout << s << " ";
    // }
    // std::cout << std::endl;
    std::cout << "ID        LastStep  NextStep  NextG" << std::endl;

    // I need three because nextStep has super weird behavior
    // I can't use an index because nextStep is not pure (??????)
    vector< State > nextM;
    vector< State > nextR;
    vector< State > nextE;
    vector< State > nextG;
    string nextMS;
    string nextRS;
    string nextES;
    string nextGS;

    auto rng = std::default_random_engine {};
    std::shuffle(std::begin(envs), std::end(envs), rng);

    string name;
    vector< State > next;
    for (size_t i = 0; i < envs.size(); i++) {

      
      Env* e = envs[i];
      name = e->getID();
      double lastR = e->latestStepReward();
      next = e->nextStep();
      double estimate = e->estimateRewardOfStep(next);
      double estimateActual = env->estimateRewardOfStep(next);
      double estimateMargin = estimate - e->currentReward();
      printf("%.*s\t%.2f\t%.2f\t%.2f\n", 10, (name + "          ").c_str(), lastR, estimateMargin, estimateActual);

      
      if (lastR > maxLast) {
	maxLast = lastR;
	nextR = next;
	nextRS = name;
      } 

      // if (estimate > maxEstimate) {
      // 	maxEstimate = estimate;
      // 	nextE = next;
      // 	nextES = name;
      // }

      if (estimateMargin > maxEstimateMargin) {
	maxEstimateMargin = estimateMargin;
	nextM = next;
	nextMS = name;
      }

      if (estimateActual > maxEstimateActual) {
	maxEstimateActual = estimateActual;
	nextG = next;
	nextGS = name;
      }
    }
    
    vector< State > nextMove;
    string choice;
    string choiceNoG;
    if (nextM.size() > 0) {
      nextMove = nextM;
      choice = nextMS;
      choiceNoG = choice;
    } else if (nextR.size() > 0) {
      nextMove = nextR;
      choice = nextRS;
      choiceNoG = choice;
    // } else if (nextE.size() > 0) {
    //   nextMove = nextE;
    //   choice = nextES;
    } else {
      nextMove = next;
      choice = "default pick of " + name;
      choiceNoG = choice;
    }

    if (nextG.size() > 0) {
      nextMove = nextG;
      choice = nextGS;
    }

    moveCount++;
    std::cout << "The pick with access to G is: " << choice << std::endl;
    std::cout << "The pick without access is: " << choiceNoG << std::endl;
    if (choice.compare(choiceNoG) != 0) {
      std::cout << "The choices were not equal." << std::endl;
    } else {
      equalCount++;
    }
    if (rand() % 100 < 3) {
      nextMove = next;
      choice = "random pick of " + name;
    }

    // std::cout << "Choice nextMove: " << std::endl;
    // for (auto& s : nextMove) {
    //   std::cout << s << " ";
    // }
    std::cout << "Choice nextMove: " << choice << std::endl;
    
    for (auto& e : envs) {
      e->applyStep(nextMove);
    }

    env->applyStep(nextMove);
  }

  std::cout << "Final alignment percentage: " << equalCount / moveCount << std::endl;
  std::cout << "Final reward: " << env->currentReward() << std::endl;
}

double MultiRover::runSim(Env* env) {
  for (size_t t = 0; t < nSteps; t++) {
    vector< State > jointState = env->step();

    if (outputTrajs) {
      printJointState(jointState);
    }
  }

  return env->currentReward();
}

Env* MultiRover::createSim(size_t teamSize) {
  Env* env = new Env(world, roverTeam, POIs, teamSize);

  vector< State > initState;
  vector< size_t > netPerAgent;
  for (size_t j = 0; j < nRovers; j++) {
    State s(initialXYs[j], initialPsis[j]);
    initState.push_back(s);
  }

  env->init(initState, netPerAgent);

  return env;
}
void MultiRover::SimulateEpoch(bool train){
  size_t teamSize = train ? 2*nPop : nPop;
    
  // each row is the population for a single agent
  vector< vector<size_t> > teams = RandomiseTeams(teamSize) ; 

  if (outputTrajs) {
    printPOIs();
  }

  Env* env = createSim(teamSize);
  
  double maxEval = 0.0 ;
  vector< size_t > netEachAgentUses;
  
  for (size_t i = 0; i < teamSize; i++) { // looping across the columns of 'teams'
    // Initialise world and reset rovers and POIs
    vector< State > jointState ;
    netEachAgentUses.clear();
    
    for (size_t j = 0; j < nRovers; j++) {
      netEachAgentUses.push_back(teams[j][i]);
    }

    env->init(netEachAgentUses);
    
    if (outputTrajs && i == teamSize - 1) {
      printJointState(jointState);
      toggleAgentOutput(true);
    }
    
    double eval = runSim(env);
    env->reset();
    maxEval = max(eval, maxEval);
    
    // Assign fitness
    for (size_t j = 0; j < nRovers; j++) {
      roverTeam[j]->SetEpochPerformance(eval, teams[j][i]) ;
    }
    
    if (outputEvals) {
      evalFile << eval << "," ;
    }
  }

  delete env;
  if (outputEvals) {
    evalFile << std::endl;
  }

  if (verbose) {
    std::cout << "max achieved value: " << maxEval << "..." << std::endl;
  }
}

void MultiRover::EvolvePolicies(bool init){
  for (auto& rov : roverTeam) {
    rov->EvolvePolicies(init);
  }
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

// Wrapper for writing POMDP actions to specified file
void MultiRover::OutputQueries(char * A){
  // Filename to write to stored in A
  std::stringstream fileName ;
  fileName << A ;
  if (quryFile.is_open())
    quryFile.close() ;
  quryFile.open(fileName.str().c_str(),std::ios::app) ;
  
  outputQury = true ;
}

// Wrapper for writing POMDP beliefs to specified file
void MultiRover::OutputBeliefs(char * A){
	// Filename to write to stored in A
	std::stringstream fileName ;
  fileName << A ;
  if (blfFile.is_open())
    blfFile.close() ;
  blfFile.open(fileName.str().c_str(),std::ios::app) ;
  
  outputBlf = true ;
}

// Wrapper for writing running average of stepwiseD to specified file
void MultiRover::OutputAverageStepwise(char * A){
	// Filename to write to stored in A
	std::stringstream fileName ;
  fileName << A ;
  if (avgStepRFile.is_open())
    avgStepRFile.close() ;
  avgStepRFile.open(fileName.str().c_str(),std::ios::app) ;
  
  outputAvgStepR = true ;
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
void MultiRover::ExecutePolicies(string expFile, string novFile,
				 string storeTraj, string storePOI,
				 string storeEval, size_t numIn, size_t numOut,
				 size_t numHidden){

  if (verbose) {
    std::cout << "Reading out " << nPop << " expert NN control policies for"
	      << " each rover to test..." << std::endl;
  }
  
  vector<NeuralNet *> expLoadedNN = NeuralNet::loadNNFromFile(expFile,numIn,
							      numHidden,numOut);

  if (verbose) {
    std::cout << "Reading out " << nPop << " novice NN control policies "
	      << "for each rover to test..." << std::endl;
  }
  
  vector<NeuralNet *> novLoadedNN = NeuralNet::loadNNFromFile(novFile, numIn,
							      numHidden,numOut);
  
  // Assign control policies to rovers ;
  int k = 0 ;
  for (size_t i = 0; i < nRovers; i++){
    NeuroEvo * rovNE = roverTeam[i]->GetNEPopulation() ;
    if (k == 0){
      for (size_t j = 0; j < nPop; j++){
        rovNE->GetNNIndex(j)->SetWeights(novLoadedNN[k]->GetWeightsA(),novLoadedNN[k]->GetWeightsB()) ;
        k++ ;
      }
    }
    else {
      for (size_t j = 0; j < nPop; j++){
        rovNE->GetNNIndex(j)->SetWeights(expLoadedNN[k]->GetWeightsA(),expLoadedNN[k]->GetWeightsB()) ;
        k++ ;
      }
    }
  }
  
  // Initialise test world
  std::cout << "Initialising test world...\n" ;
  InitialiseEpoch() ;
  OutputPerformance(storeEval) ;
  OutputTrajectories(storeTraj, storePOI, "d") ;
  ResetEpochEvals() ;
  SimulateEpoch(false) ; // simulate in test mode
  
  for (size_t i = 0; i < expLoadedNN.size(); i++){
    delete expLoadedNN[i] ;
    expLoadedNN[i] = 0 ;
  }
  for (size_t i = 0; i < novLoadedNN.size(); i++){
    delete novLoadedNN[i] ;
    novLoadedNN[i] = 0 ;
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
 
void MultiRover::printJointState(const vector<State> jointState) {
  for (const auto& s : jointState) {
    Vector2d vec = s.pos();
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

double MultiRover::calculatePOIValue(const Target& poi) {
  return poi.IsObserved() ? (poi.GetValue() / max(poi.GetNearestObs(), 1.0)) : 0.0;
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

  string output[h][w];
  
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

    if (fh < 0 || fw < 0 || fw > width || fh > height) {
      continue;
    }
    // Look up 6.042. This is probably not needed. I forget why I put it in
    int ph = (fh + h) % h;
    int pw = (fw + w) % w;

    vals[ph][pw] += poi.GetValue();
    //strm << poi << std::endl;
  }
  
  for (const auto& rov : d.roverTeam) {
    //strm << *rov << std::endl;

    Vector2d loc = rov->getCurrentXY();
    int fh = (int) (floor (loc(0)));
    int fw = (int) (floor (loc(1)));
    if (fh < 0 || fw < 0 || fw > width || fh > height) {
      continue;
    }
    int ph = (fh + h) % h;
    int pw = (fw + w) % w;

    output[ph][pw] += "A";
  }

  for (int i = 0; i < h; i++) {
    for (int j = 0; j < w; j++) {
      //strm << vals[i][j] << " ";
    }
    //strm << std::endl;
  }

  for (int i = 0; i < h; i++) {
    for (int j = 0; j < w; j++) {
      if (vals[i][j] > 0) {
	output[i][j] += "P";
      }
      string out = output[i][j];
      out = (" " + out + " - ").substr(1,3);
      strm << out;
    }
    strm << std::endl;
  }
  
  return strm;
}
